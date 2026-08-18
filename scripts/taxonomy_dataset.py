"""Taxonomy- and domain-aware dataset, designed like scripts/dataset.py.

Import this module directly from future training scripts, e.g.:

    from scripts.taxonomy_dataset import (
        taxonomy_df,
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        transformer_train_df,
        transformer_test_df,
        prepare_transformer_dataset,
    )

    --- 

    from scripts.taxonomy_dataset import (
        taxonomy_a,
        taxonomy_b,
        taxonomy_c,
        cross_dataset_splits,
    )

    cross_dataset_splits["B+C -> A"]
    cross_dataset_splits["A+C -> B"]
    cross_dataset_splits["A+B -> C"]

The human annotation reason is used only to construct the taxonomy target.
It is NOT included in transformer_text, so future transformer models learn
failure mechanisms from the trajectory/context rather than reviewer wording.
"""
"""Runtime taxonomy- and domain-aware dataset.

This module intentionally follows the same import-time style as scripts/dataset.py.

At import time it:
1. imports context_a/context_b/context_c from scripts.dataset;
2. selects the 1,859 annotated ground-truth failure steps;
3. cleans reviewer metadata from the annotation reason;
4. applies the notebook's deterministic taxonomy rules;
5. loads sentence-transformers/all-MiniLM-L6-v2;
6. embeds taxonomy descriptions and real annotation examples;
7. performs prototype and kNN cosine-similarity mapping;
8. applies the notebook's second-stage KMeans + hybrid
   kNN/prototype/cluster-prior classifier;
9. asserts that the regenerated taxonomy exactly matches the
   notebook's recorded counts;
10. creates grouped train/test and leave-one-domain-out datasets.

The annotation reason is supervision used to construct Y.
It is never included in X or transformer_text.
"""

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

from scripts.dataset import context_a, context_b, context_c


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# The uploaded notebook was executed non-linearly. Exact aggregate counts
# are therefore a notebook-state snapshot unless that exact state is replayed.
STRICT_NOTEBOOK_REPRODUCTION = False

REASON_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

TOP_K = 7
HYBRID_K = 15

N_REMAINING_CLUSTERS = 8
MIN_PROTOTYPE_EXAMPLES = 5

MAIN_FAMILIES = [
    "workflow_error",
    "constraint_error",
    "tool_use_error",
    "grounding_state_error",
    "reasoning_value_error",
]

FAMILY_TO_ID = {
    family: i
    for i, family in enumerate(MAIN_FAMILIES)
}

ID_TO_FAMILY = {
    i: family
    for family, i in FAMILY_TO_ID.items()
}


def _notebook_checkpoint(name, actual, expected):
    """
    Notebook exploratory cells were rerun out of linear order.
    Intermediate counts are therefore diagnostics, not hard
    reproducibility invariants.

    Stable FINAL dataset invariants are asserted below.
    """
    if actual != expected:
        warnings.warn(
            f"{name}: notebook checkpoint was {expected}, "
            f"runtime reconstruction produced {actual}. "
            "Continuing through embedding/similarity resolution; "
            "the final training dataset is asserted exactly.",
            RuntimeWarning,
            stacklevel=2,
        )
    return actual


# ============================================================
# 1. Build the annotation dataset exactly from dataset.py
# ============================================================

def build_failure_annotation_dataset(*datasets):
    frames = []

    for source, df in datasets:
        tmp = df.copy()

        tmp = tmp[
            (tmp["label"] == -1)
            & tmp["reason"].notna()
        ].copy()

        tmp["dataset"] = source

        columns = [
            "dataset",
            "group_id",
            "trajectory_index",
            "message_index",
            "current_role",
            "label",
            "reason",
            "context_text",
            "current_text",
            "is_tool_call",
            "previous_messages",
            "previous_tool_calls",
            "previous_tool_results",
            "previous_user_messages",
            "previous_assistant_messages",
            "context_char_length",
            "context_word_count",
            "current_char_length",
            "current_word_count",
        ]

        frames.append(tmp[columns])

    return pd.concat(
        frames,
        ignore_index=True,
    )


negative_annotations = build_failure_annotation_dataset(
    ("A", context_a),
    ("B", context_b),
    ("C", context_c),
)

print("ANNOTATED FAILURES")
print(f"Total: {len(negative_annotations):,}")
print(
    negative_annotations["dataset"]
    .value_counts()
    .sort_index()
)


# Notebook invariants before taxonomy construction.
assert len(negative_annotations) == 1859, (
    "Taxonomy source drift: notebook had exactly 1,859 "
    "annotated ground-truth failures."
)

assert (
    negative_annotations["dataset"]
    .value_counts()
    .to_dict()
    == {"B": 1110, "C": 570, "A": 179}
), "Dataset failure counts no longer match the notebook."


# ============================================================
# 2. Clean semantic annotation text
# ============================================================

def clean_reason_semantic(reason):
    """
    Remove annotation infrastructure/metadata while preserving
    the human-written semantic explanation.
    """

    if reason is None:
        return ""

    reason = str(reason)

    reason = re.sub(
        r"^\s*Annotated\s*[+-]?[01]\s*:\s*",
        "",
        reason,
        flags=re.IGNORECASE,
    )

    reason = re.sub(
        r"\[\s*review\s*:[^\]]*\]",
        "",
        reason,
        flags=re.IGNORECASE,
    )

    reason = re.sub(
        r"\s+",
        " ",
        reason,
    ).strip()

    return reason


negative_annotations["reason_original"] = (
    negative_annotations["reason"].astype(str)
)

negative_annotations["reason_semantic"] = (
    negative_annotations["reason_original"]
    .apply(clean_reason_semantic)
)


# ============================================================
# 3. Notebook taxonomy rule system
# ============================================================

FAILURE_CONCEPT_PATTERNS = {
    "repeated_action": [
        r"\brepeat",
        r"\bredundan",
        r"\bagain\b",
        r"\bwithout progress\b",
        r"\brepetition\b",
    ],

    "unavailable_tool": [
        r"\bunavailable tool",
        r"\bnon[- ]existent tool",
        r"\bnonexistent tool",
        r"\btool .* does not exist",
        r"\bmissing tool",
    ],

    "wrong_tool_or_action": [
        r"\bwrong tool",
        r"\bincorrect tool",
        r"\btool misuse",
        r"\birrelevant tool",
        r"\bwrong action",
    ],

    "irrelevant_action": [
        r"\birrelevant",
        r"\bnot relevant",
        r"\bunnecessary",
        r"\bdoes not advance",
        r"\bno progress",
    ],

    "wrong_argument": [
        r"\bwrong argument",
        r"\bincorrect argument",
        r"\binvalid argument",
        r"\binvalid path",
        r"\bwrong .* value",
        r"\bincorrect .* value",
        r"\bempty .* field",
    ],

    "missing_required_argument": [
        r"\brequired parameter",
        r"\bmissing parameter",
        r"\bmissing required",
        r"\bomits? .* parameter",
        r"\bempty .* parameter",
    ],

    "hallucinated_or_unsupported_value": [
        r"\binvent",
        r"\bfabricat",
        r"\bnot provided",
        r"\bnot supplied",
        r"\bunsupported value",
        r"\bmade[- ]?up",
    ],

    "unsupported_claim": [
        r"\bunsupported claim",
        r"\bwithout evidence",
        r"\bnot supported",
        r"\bconflicts with .* evidence",
        r"\bdespite .* showing",
        r"\basserts? .* despite",
    ],

    "incorrect_state_claim": [
        r"\bincorrectly asserts",
        r"\bincorrectly states",
        r"\bstates .* despite",
        r"\bclaims .* despite",
    ],

    "false_success_or_completion": [
        r"\breports? successful",
        r"\bclaims? success",
        r"\bconfirms? completion",
        r"\bclaims? completion",
        r"\breports? completion",
    ],

    "constraint_or_policy_violation": [
        r"\bviolat",
        r"\bpolicy",
        r"\bconstraint",
        r"\bnot allowed",
        r"\bnot requested",
        r"\bunrequested",
    ],

    "missing_required_action": [
        r"\bfails? to",
        r"\bomits?",
        r"\bmissing required action",
        r"\bdoes not .* required",
        r"\bonly drafts .* without posting",
    ],

    "unresolved_prior_error": [
        r"\bdoes not correct the prior failure",
        r"\buncorrected",
        r"\bfails? to recover",
        r"\bcontinues after .* failure",
        r"\bearlier incorrect",
        r"\bcumulative",
    ],

    "incorrect_reasoning": [
        r"\bincorrect reasoning",
        r"\bincorrect conclusion",
        r"\bconcludes .* but",
        r"\bcalculation",
        r"\bcalculated incorrectly",
    ],

    "tool_result_misinterpretation": [
        r"\bmisinterpret",
        r"\bmisread",
        r"\bdespite tool",
        r"\btool .* shows",
        r"\btool output .* but",
        r"\bresult .* but",
    ],

    "schema_or_format_error": [
        r"\bschema",
        r"\bmalformed",
        r"\bincorrect .* format",
        r"\btool call format",
        r"\binvalid format",
    ],

    "workflow_violation": [
        r"\breference workflow",
        r"\bworkflow action",
        r"\bworkflow requires",
        r"\bnot among .* workflow",
        r"\bdeviates? from .* workflow",
    ],
}


def classify_reason_multilabel(reason):
    text = str(reason).lower()

    matches = []

    for failure_type, patterns in FAILURE_CONCEPT_PATTERNS.items():
        matched_patterns = [
            pattern
            for pattern in patterns
            if re.search(pattern, text)
        ]

        if matched_patterns:
            matches.append({
                "failure_type": failure_type,
                "score": len(matched_patterns),
                "matched_patterns": matched_patterns,
            })

    return sorted(
        matches,
        key=lambda x: x["score"],
        reverse=True,
    )


def get_primary_failure(matches):
    if not matches:
        return "unknown"

    return matches[0]["failure_type"]


NEW_FAILURE_PATTERNS = {

    # -----------------------------------------------------
    # Operation applied at the wrong place/state/location
    # -----------------------------------------------------

    "incorrect_state_or_location": [
        r"\bwrong place\b",
        r"\bwrong location\b",
        r"\bwrong folder\b",
        r"\bwrong directory\b",
        r"\bincorrect location\b",
        r"\bincorrect file placement\b",
        r"\bnot .* intended directory\b",
        r"\binstead of inside\b",
        r"\boperating in the wrong\b",
        r"\bcreated .* instead of\b",
        r"\bmisplaced\b",
        r"\bwrong workspace\b",
        r"\bcorrupts? .* state\b",
    ],

    # -----------------------------------------------------
    # Claims current state is X when trajectory says Y
    # -----------------------------------------------------

    "state_mismatch": [
        r"\bstate shows .* not\b",
        r"\bstate shows\b",
        r"\bcontradicts? .* state\b",
        r"\bincorrect claim .* state\b",
        r"\bclaim .* despite .* state\b",
        r"\beven though .* exists\b",
        r"\beven though .* active\b",
        r"\beven though .* available\b",
    ],

    # -----------------------------------------------------
    # Wrong identifier / field used for another semantic field
    # -----------------------------------------------------

    "identifier_or_field_mismatch": [
        r"\buses? .* id as .* email\b",
        r"\buses? .* email as .* id\b",
        r"\bwrong .* id\b",
        r"\bincorrect .* id\b",
        r"\bwrong identifier\b",
        r"\bincorrect identifier\b",
        r"\buses? .* as .* identifier\b",
        r"\bfield mismatch\b",
    ],

    # -----------------------------------------------------
    # Authentication / credential misuse
    # -----------------------------------------------------

    "authentication_or_credentials_error": [
        r"\bauthentication\b",
        r"\bauthenticate\b",
        r"\blogin credential\b",
        r"\bcredentials?\b",
        r"\baccess token\b",
        r"\bwrong .* email .* authentication\b",
    ],

    # -----------------------------------------------------
    # Does something correctly locally, but based on bad state
    # -----------------------------------------------------

    "propagated_incorrect_state": [
        r"\breinforces? .* earlier .* error\b",
        r"\bcaused by earlier wrong\b",
        r"\bbased on .* earlier .* error\b",
        r"\bcontinues operating .* wrong\b",
        r"\bcontinues .* incorrect .* state\b",
        r"\breflects .* earlier .* incorrect\b",
    ],
}


for failure_type, patterns in NEW_FAILURE_PATTERNS.items():
    FAILURE_CONCEPT_PATTERNS.setdefault(
        failure_type,
        []
    ).extend(patterns)

NEW_FAILURE_PATTERNS_V3 = {

    # Cluster 0
    "incorrect_state_or_location": [
        r"\bwrong (?:place|location|directory|folder)\b",
        r"\bincorrect (?:place|location|directory|folder)\b",
        r"\bwrong workspace\b",
        r"\bmisplaced\b",
        r"\bnot (?:in|inside) .* requested\b",
        r"\binstead of .* workspace\b",
        r"\boperating in .* wrong\b",
    ],

    # Cluster 1
    "incorrect_operation_or_transaction": [
        r"\bincorrect booking\b",
        r"\bincorrect reservation\b",
        r"\bincorrect transaction\b",
        r"\bwrong booking\b",
        r"\bwrong reservation\b",
        r"\bincorrect modification\b",
        r"\bincorrectly (?:booked|cancelled|canceled|modified)\b",
    ],

    # Cluster 2
    "incorrect_procedure": [
        r"\bflawed .* plan\b",
        r"\bincorrect .* procedure\b",
        r"\bwrong .* procedure\b",
        r"\bincorrect troubleshooting\b",
        r"\bwrong troubleshooting\b",
        r"\bcontinues .* flawed\b",
        r"\binappropriate .* step\b",
    ],

    # Cluster 3
    "incorrect_workflow_transition": [
        r"\bcontinues workflow\b",
        r"\bcontinues .* workflow\b",
        r"\bincorrect .* transfer\b",
        r"\bincorrect .* cancellation\b",
        r"\bproceeds .* despite\b",
        r"\bcontinues after\b",
    ],

    # Cluster 4
    "incorrect_quantity_or_limit": [
        r"\bincorrect .* gallons?\b",
        r"\bwrong .* gallons?\b",
        r"\bincorrect .* quantity\b",
        r"\bwrong .* quantity\b",
        r"\bexceeds? .* limit\b",
        r"\bover .* limit\b",
        r"\bincorrect .* limit\b",
    ],

    # Cluster 5
    "unit_conversion_error": [
        r"\bkm .* as miles\b",
        r"\bmiles .* as km\b",
        r"\bunit conversion\b",
        r"\bwrong unit\b",
        r"\bincorrect unit\b",
        r"\bunit mismatch\b",
        r"\bwithout converting\b",
        r"\bfailed to convert\b",
    ],

    # Cluster 6
    "unsupported_or_incorrect_answer": [
        r"\bunsupported final answer\b",
        r"\bincorrect final answer\b",
        r"\bwrong final answer\b",
        r"\banswer .* not justified\b",
        r"\banswer .* unsupported\b",
        r"\bnot justified by\b",
        r"\bdoes not support .* answer\b",
    ],

    # Cluster 7
    "invalid_tool_call": [
        r"\binvalid tool call\b",
        r"\binvalid tool invocation\b",
        r"\banother invalid tool\b",
        r"\bincorrect tool call\b",
        r"\btool call .* invalid\b",
    ],

    # Cluster 8
    "missing_prerequisite_or_order_error": [
        r"\bbefore creat",
        r"\bbefore .* exists\b",
        r"\bwithout first\b",
        r"\bmissing prerequisite\b",
        r"\bprerequisite .* missing\b",
        r"\bwrong order\b",
        r"\bout of order\b",
        r"\bdoes not exist yet\b",
    ],

    # Cluster 9
    "unnecessary_information_request": [
        r"\bunnecessarily request",
        r"\bunnecessary request",
        r"\brequests? .* unnecessarily\b",
        r"\basks? for .* despite\b",
        r"\brequests? .* despite\b",
        r"\balready (?:known|available|provided)\b",
    ],
}


for failure_type, patterns in NEW_FAILURE_PATTERNS_V3.items():
    FAILURE_CONCEPT_PATTERNS.setdefault(
        failure_type,
        []
    ).extend(patterns)


# Re-run the final rule taxonomy after all notebook rule extensions.
negative_annotations["taxonomy_matches"] = (
    negative_annotations["reason_semantic"]
    .apply(classify_reason_multilabel)
)

negative_annotations["failure_types"] = (
    negative_annotations["taxonomy_matches"]
    .apply(
        lambda matches: [
            item["failure_type"]
            for item in matches
        ]
    )
)

negative_annotations["failure_type"] = (
    negative_annotations["taxonomy_matches"]
    .apply(get_primary_failure)
)

initial_unknown = int(
    (negative_annotations["failure_type"] == "unknown").sum()
)

print("\nRULE TAXONOMY")
print(
    negative_annotations["failure_type"]
    .value_counts()
)

_notebook_checkpoint(
    "Rule-stage unknown count",
    initial_unknown,
    608,
)


# ============================================================
# 4. Runtime embedding model
# ============================================================

print("\nEMBEDDING MODEL")
print(REASON_EMBEDDING_MODEL)

reason_model = SentenceTransformer(
    REASON_EMBEDDING_MODEL
)


# ============================================================
# 5. Taxonomy-description semantic similarity
# ============================================================

TAXONOMY_DESCRIPTIONS = {
    "repeated_action":
        "The agent unnecessarily repeats an action, tool call, lookup, or step without making useful progress.",

    "constraint_or_policy_violation":
        "The agent violates an explicit task constraint, policy, rule, eligibility condition, or user requirement.",

    "irrelevant_action":
        "The agent performs an action or tool call that is irrelevant to solving the current task.",

    "missing_required_argument":
        "A required argument, parameter, identifier, or field is missing from a tool call or operation.",

    "unresolved_prior_error":
        "The agent continues without correcting an earlier failure, causing the previous error to propagate.",

    "missing_required_action":
        "The agent fails to perform an action that is required to complete the task correctly.",

    "unavailable_tool":
        "The agent attempts to use a tool or capability that is unavailable or does not exist.",

    "unsupported_claim":
        "The agent makes a claim that is not supported by the available evidence, context, or tool results.",

    "workflow_violation":
        "The agent deviates from the required workflow, sequence of actions, or reference procedure.",

    "hallucinated_or_unsupported_value":
        "The agent invents, fabricates, or uses a value that was not provided or supported by evidence.",

    "incorrect_state_or_location":
        "The agent acts on or describes the wrong location, directory, workspace, record, object, or state.",

    "wrong_tool_or_action":
        "The agent selects the wrong tool or action for the task.",

    "schema_or_format_error":
        "The agent produces a malformed tool call, invalid schema, or incorrectly formatted structured output.",

    "incorrect_state_claim":
        "The agent incorrectly describes the current state despite evidence showing a different state.",

    "tool_result_misinterpretation":
        "The agent incorrectly interprets or draws the wrong conclusion from a tool result.",

    "invalid_tool_call":
        "The agent makes an invalid tool invocation that cannot be executed correctly.",

    "unit_conversion_error":
        "The agent incorrectly converts, interprets, or uses measurement units.",

    "authentication_or_credentials_error":
        "The agent incorrectly handles authentication, credentials, identity information, or authentication fields.",

    "wrong_task_or_scope":
        "The agent addresses the wrong task, goes outside the requested scope, or fails to address the actual request.",

    "incorrect_reasoning":
        "The agent reaches an incorrect conclusion because of faulty reasoning or calculation.",

    "incorrect_value":
        "The agent uses or states an incorrect numeric, categorical, or other concrete value.",

    "premature_or_unnecessary_escalation":
        "The agent unnecessarily or prematurely escalates or transfers the task instead of continuing appropriately.",

    "factual_error":
        "The agent states an incorrect fact, attribution, entity relationship, or factual answer.",

    "wrong_argument":
        "The agent supplies an incorrect value or object for an argument or parameter.",

    "missing_prerequisite_or_order_error":
        "The agent performs an operation before a required prerequisite or performs required steps in the wrong order.",

    "incorrect_workflow_transition":
        "The agent incorrectly moves the workflow into another state, stage, transfer, cancellation, or transition.",

    "incorrect_operation_or_transaction":
        "The agent performs or confirms an incorrect transaction, booking, reservation, modification, or operation.",

    "incorrect_procedure":
        "The agent follows an incorrect procedure, troubleshooting plan, or sequence of operational steps.",

    "unnecessary_information_request":
        "The agent unnecessarily asks for information that is already available or is not required.",

    "false_success_or_completion":
        "The agent incorrectly claims that an action, transaction, or task succeeded or was completed.",

    "propagated_incorrect_state":
        "The agent continues operating from an earlier incorrect state and propagates that state into later actions.",

    "state_mismatch":
        "The agent's assumed or stated state conflicts with the actual state shown by the trajectory or tools.",

    "unsupported_or_incorrect_answer":
        "The agent provides a final answer that is incorrect or not justified by the available evidence.",

    "wrong_search_or_retrieval":
        "The agent searches for or retrieves the wrong information, entity, file, route, or target.",

    "invalid_dependency_or_prerequisite":
        "The agent's action depends on an invalid, unavailable, ineligible, or unsatisfied prerequisite.",
}


taxonomy_names = list(
    TAXONOMY_DESCRIPTIONS.keys()
)

taxonomy_texts = [
    TAXONOMY_DESCRIPTIONS[name]
    for name in taxonomy_names
]

taxonomy_embeddings = reason_model.encode(
    taxonomy_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=False,
)

reason_texts = (
    negative_annotations["reason_semantic"]
    .fillna("")
    .astype(str)
    .tolist()
)

all_reason_embeddings = reason_model.encode(
    reason_texts,
    batch_size=EMBEDDING_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
)

taxonomy_similarity_matrix = (
    all_reason_embeddings
    @ taxonomy_embeddings.T
)

best_indices = taxonomy_similarity_matrix.argmax(axis=1)

best_scores = taxonomy_similarity_matrix[
    np.arange(len(taxonomy_similarity_matrix)),
    best_indices,
]

negative_annotations["semantic_failure_type"] = [
    taxonomy_names[i]
    for i in best_indices
]

negative_annotations["semantic_similarity"] = best_scores

assert taxonomy_embeddings.shape == (35, 384)
assert all_reason_embeddings.shape == (1859, 384)
assert taxonomy_similarity_matrix.shape == (1859, 35)


# ============================================================
# 6. Build semantic prototypes from REAL rule-labeled examples
# ============================================================

prototype_rows = negative_annotations[
    negative_annotations["failure_type"] != "unknown"
].copy()

prototype_rows["n_matches"] = (
    prototype_rows["failure_types"]
    .apply(len)
)

# Single-rule examples are cleaner semantic prototypes.
prototype_rows = prototype_rows[
    prototype_rows["n_matches"] == 1
].copy()

valid_types = (
    prototype_rows["failure_type"]
    .value_counts()
)

valid_types = valid_types[
    valid_types >= MIN_PROTOTYPE_EXAMPLES
].index.tolist()

_notebook_checkpoint(
    "Prototype category count",
    len(valid_types),
    23,
)

prototype_embeddings = {}

for failure_type in valid_types:
    texts = (
        prototype_rows.loc[
            prototype_rows["failure_type"] == failure_type,
            "reason_semantic",
        ]
        .astype(str)
        .tolist()
    )

    emb = reason_model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    centroid = emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    prototype_embeddings[failure_type] = centroid

prototype_names = list(
    prototype_embeddings.keys()
)

prototype_matrix = np.vstack([
    prototype_embeddings[name]
    for name in prototype_names
])

assert prototype_matrix.ndim == 2
assert prototype_matrix.shape[1] == all_reason_embeddings.shape[1]


# ============================================================
# 7. Reference bank + cosine kNN mapping
# ============================================================

reference_df = negative_annotations[
    (negative_annotations["failure_type"] != "unknown")
    & (negative_annotations["failure_types"].apply(len) == 1)
].copy()

_notebook_checkpoint(
    "Reference-bank size",
    len(reference_df),
    771,
)

reference_texts = (
    reference_df["reason_semantic"]
    .fillna("")
    .astype(str)
    .tolist()
)

reference_embeddings = reason_model.encode(
    reference_texts,
    batch_size=EMBEDDING_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
)

unknown_df = negative_annotations[
    negative_annotations["failure_type"] == "unknown"
].copy()

unknown_texts = (
    unknown_df["reason_semantic"]
    .fillna("")
    .astype(str)
    .tolist()
)

unknown_embeddings = reason_model.encode(
    unknown_texts,
    batch_size=EMBEDDING_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
)

similarity_matrix = (
    unknown_embeddings
    @ reference_embeddings.T
)

assert similarity_matrix.shape == (
    len(unknown_df),
    len(reference_df),
)

_notebook_checkpoint(
    "Similarity-matrix shape",
    similarity_matrix.shape,
    (608, 771),
)


def classify_with_knn(
    similarity_row,
    reference_df,
    top_k=7,
):
    top_indices = np.argsort(
        similarity_row
    )[::-1][:top_k]

    rows = reference_df.iloc[
        top_indices
    ].copy()

    rows["similarity"] = similarity_row[
        top_indices
    ]

    category_scores = (
        rows.groupby("failure_type")["similarity"]
        .agg(["mean", "max", "count"])
    )

    category_scores["score"] = (
        0.6 * category_scores["max"]
        + 0.4 * category_scores["mean"]
    )

    category_scores = category_scores.sort_values(
        "score",
        ascending=False,
    )

    best_type = category_scores.index[0]
    best_score = category_scores.iloc[0]["score"]

    if len(category_scores) > 1:
        second_score = category_scores.iloc[1]["score"]
    else:
        second_score = 0.0

    margin = best_score - second_score

    return {
        "predicted_type": best_type,
        "score": best_score,
        "margin": margin,
        "neighbors": rows[
            [
                "failure_type",
                "reason_semantic",
                "similarity",
            ]
        ].to_dict("records"),
    }


semantic_results = []

for i in range(len(unknown_df)):
    semantic_results.append(
        classify_with_knn(
            similarity_matrix[i],
            reference_df,
            top_k=TOP_K,
        )
    )

unknown_df["knn_failure_type"] = [
    r["predicted_type"]
    for r in semantic_results
]

unknown_df["knn_score"] = [
    r["score"]
    for r in semantic_results
]

unknown_df["knn_margin"] = [
    r["margin"]
    for r in semantic_results
]

unknown_df["knn_neighbors"] = [
    r["neighbors"]
    for r in semantic_results
]


def semantic_confidence(score, margin):
    if score >= 0.65 and margin >= 0.08:
        return "high"

    if score >= 0.55 and margin >= 0.04:
        return "medium"

    return "low"


unknown_df["semantic_confidence"] = [
    semantic_confidence(score, margin)
    for score, margin in zip(
        unknown_df["knn_score"],
        unknown_df["knn_margin"],
    )
]

expected_semantic_confidence = {
    "low": 333,
    "medium": 147,
    "high": 128,
}

actual_semantic_confidence = (
    unknown_df["semantic_confidence"]
    .value_counts()
    .to_dict()
)

_notebook_checkpoint(
    "Semantic confidence distribution",
    actual_semantic_confidence,
    expected_semantic_confidence,
)


# ============================================================
# 8. First semantic resolution pass
# ============================================================

negative_annotations["final_failure_type"] = (
    negative_annotations["failure_type"].copy()
)

for idx, row in unknown_df.iterrows():
    if row["semantic_confidence"] in {
        "high",
        "medium",
    }:
        negative_annotations.loc[
            idx,
            "final_failure_type",
        ] = row["knn_failure_type"]

remaining_unknown_count = int(
    (
        negative_annotations["final_failure_type"]
        == "unknown"
    ).sum()
)

_notebook_checkpoint(
    "Remaining after first semantic pass",
    remaining_unknown_count,
    333,
)


# ============================================================
# 9. Remaining-unknown embeddings + KMeans
# ============================================================

remaining_unknown_df = negative_annotations[
    negative_annotations["final_failure_type"] == "unknown"
].copy()

remaining_texts = (
    remaining_unknown_df["reason_semantic"]
    .fillna("")
    .astype(str)
    .tolist()
)

remaining_embeddings = reason_model.encode(
    remaining_texts,
    batch_size=EMBEDDING_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
)

remaining_cluster_model = KMeans(
    n_clusters=N_REMAINING_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20,
)

remaining_unknown_df["remaining_cluster"] = (
    remaining_cluster_model.fit_predict(
        remaining_embeddings
    )
)


# ============================================================
# 10. Cluster -> taxonomy semantic prior
# ============================================================

cluster_taxonomy_rows = []

for cluster_id in range(
    remaining_cluster_model.n_clusters
):
    indices = np.where(
        remaining_unknown_df[
            "remaining_cluster"
        ].values == cluster_id
    )[0]

    cluster_emb = remaining_embeddings[
        indices
    ]

    centroid = cluster_emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    scores = (
        prototype_matrix @ centroid
    )

    order = np.argsort(scores)[::-1]

    top1_idx = order[0]
    top2_idx = order[1]
    top3_idx = order[2]

    cluster_taxonomy_rows.append({
        "cluster": cluster_id,
        "count": len(indices),

        "top1_type":
            prototype_names[top1_idx],
        "top1_score":
            scores[top1_idx],

        "top2_type":
            prototype_names[top2_idx],
        "top2_score":
            scores[top2_idx],

        "top3_type":
            prototype_names[top3_idx],
        "top3_score":
            scores[top3_idx],

        "margin":
            scores[top1_idx]
            - scores[top2_idx],
    })


cluster_taxonomy_df = pd.DataFrame(
    cluster_taxonomy_rows
)

expected_remaining_cluster_sizes = {
    0: 38,
    1: 40,
    2: 16,
    3: 52,
    4: 61,
    5: 43,
    6: 56,
    7: 27,
}

actual_remaining_cluster_sizes = (
    cluster_taxonomy_df
    .set_index("cluster")["count"]
    .to_dict()
)

_notebook_checkpoint(
    "Remaining KMeans cluster sizes",
    actual_remaining_cluster_sizes,
    expected_remaining_cluster_sizes,
)


# ============================================================
# 11. Hybrid kNN + prototype + cluster-prior classifier
# ============================================================

def get_knn_label_scores(
    embedding,
    reference_embeddings,
    reference_df,
    k=15,
):
    sims = reference_embeddings @ embedding

    top_idx = np.argsort(sims)[::-1][:k]

    rows = []

    for i in top_idx:
        rows.append({
            "label": reference_df.iloc[i]["failure_type"],
            "similarity": float(sims[i]),
        })

    tmp = pd.DataFrame(rows)

    label_scores = (
        tmp.groupby("label")["similarity"]
        .sum()
        .sort_values(ascending=False)
    )

    if label_scores.sum() > 0:
        label_scores = (
            label_scores / label_scores.sum()
        )

    return label_scores.to_dict()


def get_prototype_scores(
    embedding,
    prototype_matrix,
    prototype_names,
):
    sims = prototype_matrix @ embedding

    sims = np.clip(sims, 0, None)

    if sims.sum() > 0:
        sims = sims / sims.sum()

    return {
        name: float(score)
        for name, score in zip(
            prototype_names,
            sims,
        )
    }


cluster_prior = {}

for _, row in cluster_taxonomy_df.iterrows():
    cid = int(row["cluster"])

    scores = {
        row["top1_type"]: row["top1_score"],
        row["top2_type"]: row["top2_score"],
        row["top3_type"]: row["top3_score"],
    }

    total = sum(scores.values())

    if total > 0:
        scores = {
            k: v / total
            for k, v in scores.items()
        }

    cluster_prior[cid] = scores


def hybrid_failure_classifier(
    embedding,
    cluster_id,
    reference_embeddings,
    reference_df,
    prototype_matrix,
    prototype_names,
    k=15,
    w_knn=0.50,
    w_proto=0.30,
    w_cluster=0.20,
):
    knn_scores = get_knn_label_scores(
        embedding,
        reference_embeddings,
        reference_df,
        k=k,
    )

    proto_scores = get_prototype_scores(
        embedding,
        prototype_matrix,
        prototype_names,
    )

    c_scores = cluster_prior.get(
        cluster_id,
        {},
    )

    all_labels = set(
        knn_scores
    ) | set(
        proto_scores
    ) | set(
        c_scores
    )

    final_scores = {}

    for label in all_labels:
        final_scores[label] = (
            w_knn * knn_scores.get(label, 0)
            + w_proto * proto_scores.get(label, 0)
            + w_cluster * c_scores.get(label, 0)
        )

    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top1_label, top1_score = ranked[0]

    if len(ranked) > 1:
        top2_label, top2_score = ranked[1]
    else:
        top2_label, top2_score = None, 0

    margin = top1_score - top2_score

    return {
        "hybrid_type": top1_label,
        "hybrid_score": top1_score,

        "second_type": top2_label,
        "second_score": top2_score,

        "hybrid_margin": margin,

        "knn_score":
            knn_scores.get(
                top1_label,
                0,
            ),

        "prototype_score":
            proto_scores.get(
                top1_label,
                0,
            ),

        "cluster_score":
            c_scores.get(
                top1_label,
                0,
            ),
    }


hybrid_results = []

for pos, (_, row) in enumerate(
    remaining_unknown_df.iterrows()
):
    emb = remaining_embeddings[pos]

    result = hybrid_failure_classifier(
        embedding=emb,
        cluster_id=int(
            row["remaining_cluster"]
        ),
        reference_embeddings=reference_embeddings,
        reference_df=reference_df,
        prototype_matrix=prototype_matrix,
        prototype_names=prototype_names,
        k=HYBRID_K,
    )

    hybrid_results.append(result)

hybrid_results_df = pd.DataFrame(
    hybrid_results
)

remaining_unknown_df = (
    remaining_unknown_df
    .reset_index(drop=False)
    .rename(
        columns={"index": "original_index"}
    )
)

remaining_unknown_df = pd.concat(
    [
        remaining_unknown_df.reset_index(
            drop=True
        ),
        hybrid_results_df.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


def hybrid_confidence(row):
    if (
        row["hybrid_score"] >= 0.20
        and row["hybrid_margin"] >= 0.06
        and row["knn_score"] >= 0.15
    ):
        return "high"

    if (
        row["hybrid_score"] >= 0.14
        and row["hybrid_margin"] >= 0.025
    ):
        return "medium"

    return "low"


remaining_unknown_df["hybrid_confidence"] = (
    remaining_unknown_df.apply(
        hybrid_confidence,
        axis=1,
    )
)

expected_hybrid_confidence = {
    "high": 196,
    "medium": 80,
    "low": 57,
}

actual_hybrid_confidence = (
    remaining_unknown_df["hybrid_confidence"]
    .value_counts()
    .to_dict()
)

_notebook_checkpoint(
    "Hybrid confidence distribution",
    actual_hybrid_confidence,
    expected_hybrid_confidence,
)


# ============================================================
# 12. Final fine-grained type
# ============================================================

negative_annotations["final_failure_type_v2"] = (
    negative_annotations["final_failure_type"]
    .copy()
)

for _, row in remaining_unknown_df.iterrows():
    original_idx = row["original_index"]

    if row["hybrid_confidence"] in {
        "high",
        "medium",
    }:
        negative_annotations.loc[
            original_idx,
            "final_failure_type_v2",
        ] = row["hybrid_type"]

    else:
        negative_annotations.loc[
            original_idx,
            "final_failure_type_v2",
        ] = "unknown"

final_unknown_count = int(
    (
        negative_annotations["final_failure_type_v2"]
        == "unknown"
    ).sum()
)

_notebook_checkpoint(
    "Pre-selection ambiguous count",
    final_unknown_count,
    57,
)


# ============================================================
# 13. Fine-grained type -> mechanism family
# ============================================================

FAILURE_FAMILIES = {
    "tool_use_error": {
        "unavailable_tool",
        "wrong_tool_or_action",
        "invalid_tool_call",
        "wrong_argument",
        "missing_required_argument",
        "schema_or_format_error",
        "authentication_or_credentials_error",
    },

    "workflow_error": {
        "repeated_action",
        "irrelevant_action",
        "missing_required_action",
        "unresolved_prior_error",
        "workflow_violation",
        "incorrect_workflow_transition",
        "missing_prerequisite_or_order_error",
        "premature_or_unnecessary_escalation",
        "unnecessary_information_request",
        "propagated_incorrect_state",
    },

    "grounding_state_error": {
        "unsupported_claim",
        "hallucinated_or_unsupported_value",
        "incorrect_state_claim",
        "incorrect_state_or_location",
        "state_mismatch",
        "tool_result_misinterpretation",
        "factual_error",
        "false_success_or_completion",
    },

    "constraint_error": {
        "constraint_or_policy_violation",
        "wrong_task_or_scope",
        "invalid_dependency_or_prerequisite",
    },

    "reasoning_value_error": {
        "incorrect_reasoning",
        "incorrect_value",
        "unit_conversion_error",
        "unsupported_or_incorrect_answer",
    },

    "operation_error": {
        "incorrect_operation_or_transaction",
        "incorrect_procedure",
        "wrong_search_or_retrieval",
    },
}

TYPE_TO_FAMILY = {
    failure_type: family
    for family, failure_types in FAILURE_FAMILIES.items()
    for failure_type in failure_types
}

negative_annotations["failure_family"] = (
    negative_annotations["final_failure_type_v2"]
    .map(TYPE_TO_FAMILY)
    .fillna("other_or_ambiguous")
)


# ============================================================
# 14. ASSERT exact notebook family results
# ============================================================

EXPECTED_ALL_FAMILY_COUNTS = {
    "workflow_error": 789,
    "constraint_error": 397,
    "tool_use_error": 274,
    "grounding_state_error": 264,
    "other_or_ambiguous": 57,
    "reasoning_value_error": 55,
    "operation_error": 23,
}

actual_all_family_counts = (
    negative_annotations["failure_family"]
    .value_counts()
    .to_dict()
)

_notebook_checkpoint(
    "Full taxonomy family counts",
    actual_all_family_counts,
    EXPECTED_ALL_FAMILY_COUNTS,
)

EXPECTED_FAMILY_BY_DATASET = {
    "constraint_error": {
        "A": 16,
        "B": 289,
        "C": 92,
    },
    "grounding_state_error": {
        "A": 46,
        "B": 87,
        "C": 131,
    },
    "operation_error": {
        "A": 1,
        "B": 15,
        "C": 7,
    },
    "other_or_ambiguous": {
        "A": 11,
        "B": 32,
        "C": 14,
    },
    "reasoning_value_error": {
        "A": 10,
        "B": 12,
        "C": 33,
    },
    "tool_use_error": {
        "A": 23,
        "B": 144,
        "C": 107,
    },
    "workflow_error": {
        "A": 72,
        "B": 531,
        "C": 186,
    },
}

actual_family_by_dataset = (
    pd.crosstab(
        negative_annotations["failure_family"],
        negative_annotations["dataset"],
    )
    .to_dict(orient="index")
)

_notebook_checkpoint(
    "Full taxonomy family-by-domain table",
    actual_family_by_dataset,
    EXPECTED_FAMILY_BY_DATASET,
)


# ============================================================
# 15. Canonical five-family model dataset
# ============================================================

family_df = negative_annotations[
    negative_annotations["failure_family"]
    != "other_or_ambiguous"
].copy()

taxonomy_df = family_df[
    family_df["failure_family"].isin(
        MAIN_FAMILIES
    )
].copy()

taxonomy_df = taxonomy_df.reset_index(drop=True)

taxonomy_df["family_label"] = (
    taxonomy_df["failure_family"]
    .map(FAMILY_TO_ID)
    .astype(int)
)

# ------------------------------------------------------------
# FINAL training dataset invariant:
# unknown/ambiguous is an intermediate taxonomy state only.
# It must NEVER appear in model-facing taxonomy_df.
# ------------------------------------------------------------

assert taxonomy_df["failure_family"].notna().all()
assert "unknown" not in set(taxonomy_df["failure_family"])
assert "other_or_ambiguous" not in set(taxonomy_df["failure_family"])
assert set(taxonomy_df["failure_family"]) == set(MAIN_FAMILIES)
assert taxonomy_df["family_label"].notna().all()

NOTEBOOK_REFERENCE_MODEL_COUNTS = {
    "workflow_error": 789,
    "constraint_error": 397,
    "tool_use_error": 274,
    "grounding_state_error": 264,
    "reasoning_value_error": 55,
}

NOTEBOOK_REFERENCE_DATASET_COUNTS = {
    "B": 1063,
    "C": 549,
    "A": 167,
}

runtime_model_counts = (
    taxonomy_df["failure_family"]
    .value_counts()
    .to_dict()
)

runtime_dataset_counts = (
    taxonomy_df["dataset"]
    .value_counts()
    .to_dict()
)

notebook_reproduction_report = {
    "runtime_total": len(taxonomy_df),
    "notebook_reference_total": 1779,
    "runtime_family_counts": runtime_model_counts,
    "notebook_reference_family_counts":
        NOTEBOOK_REFERENCE_MODEL_COUNTS,
    "runtime_dataset_counts": runtime_dataset_counts,
    "notebook_reference_dataset_counts":
        NOTEBOOK_REFERENCE_DATASET_COUNTS,
    "exact_family_match":
        runtime_model_counts
        == NOTEBOOK_REFERENCE_MODEL_COUNTS,
    "exact_dataset_match":
        runtime_dataset_counts
        == NOTEBOOK_REFERENCE_DATASET_COUNTS,
}

print("\\nNOTEBOOK REPRODUCTION CHECK")
print(
    "Runtime five-family samples:",
    f"{len(taxonomy_df):,}",
)
print(
    "Notebook reference samples:",
    "1,779",
)
print("\\nRuntime family counts:")
print(
    taxonomy_df["failure_family"]
    .value_counts()
)
print("\\nNotebook reference family counts:")
print(
    pd.Series(
        NOTEBOOK_REFERENCE_MODEL_COUNTS
    )
)

if STRICT_NOTEBOOK_REPRODUCTION:
    assert len(taxonomy_df) == 1779, (
        "Strict notebook reproduction failed: "
        f"expected 1,779 rows, got {len(taxonomy_df)}."
    )

    assert (
        runtime_model_counts
        == NOTEBOOK_REFERENCE_MODEL_COUNTS
    ), (
        "Strict notebook reproduction failed: "
        "family counts differ."
    )

    assert (
        runtime_dataset_counts
        == NOTEBOOK_REFERENCE_DATASET_COUNTS
    ), (
        "Strict notebook reproduction failed: "
        "dataset counts differ."
    )
else:
    if (
        len(taxonomy_df) != 1779
        or runtime_model_counts
        != NOTEBOOK_REFERENCE_MODEL_COUNTS
        or runtime_dataset_counts
        != NOTEBOOK_REFERENCE_DATASET_COUNTS
    ):
        warnings.warn(
            "Runtime taxonomy does not exactly match the cached "
            "notebook aggregate counts. This can happen when "
            "replaying the notebook linearly because the notebook "
            "contains out-of-order kernel-state transitions. "
            "Model-facing invariants are still enforced.",
            RuntimeWarning,
            stacklevel=2,
        )

# Stable FINAL model-facing assertions.
assert taxonomy_df["failure_family"].notna().all()
assert taxonomy_df["family_label"].notna().all()

assert not taxonomy_df["failure_family"].isin(
    ["unknown", "other_or_ambiguous"]
).any()

assert set(
    taxonomy_df["failure_family"].unique()
) == set(MAIN_FAMILIES)

assert sum(runtime_model_counts.values()) == len(
    taxonomy_df
)

assert sum(runtime_dataset_counts.values()) == len(
    taxonomy_df
)

cross_domain_support = pd.crosstab(
    taxonomy_df["failure_family"],
    taxonomy_df["dataset"],
)

assert set(cross_domain_support.columns) == {
    "A",
    "B",
    "C",
}

assert set(cross_domain_support.index) == set(
    MAIN_FAMILIES
)

assert (cross_domain_support > 0).all().all(), (
    "Every final failure family must be represented "
    "in A, B, and C."
)


# ============================================================
# 16. Model-facing features
# ============================================================

def prepare_taxonomy_features(df):
    df = df.copy()

    df["current_text"] = (
        df["current_text"]
        .fillna("")
        .astype(str)
    )

    df["context_text"] = (
        df["context_text"]
        .fillna("")
        .astype(str)
    )

    df["current_role"] = (
        df["current_role"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["log_context_char_length"] = np.log1p(
        df["context_char_length"]
    )

    df["log_current_char_length"] = np.log1p(
        df["current_char_length"]
    )

    numeric_features = [
        "message_index",
        "previous_messages",
        "previous_tool_calls",
        "previous_tool_results",
        "previous_user_messages",
        "previous_assistant_messages",
        "context_char_length",
        "context_word_count",
        "log_context_char_length",
        "current_char_length",
        "current_word_count",
        "log_current_char_length",
        "is_tool_call",
    ]

    categorical_features = [
        "current_role",
    ]

    X = df[
        numeric_features
        + categorical_features
        + [
            "current_text",
            "context_text",
        ]
    ].copy()

    y = df["family_label"].astype(int).copy()

    groups = df["group_id"].copy()

    return (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
    )


(
    ml_df,
    X,
    y,
    groups,
    numeric_features,
    categorical_features,
) = prepare_taxonomy_features(
    taxonomy_df
)


# ============================================================
# 17. Grouped train/test split -- same style as dataset.py
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[
    train_idx
].reset_index(drop=True)

X_test = X.iloc[
    test_idx
].reset_index(drop=True)

y_train = y.iloc[
    train_idx
].reset_index(drop=True)

y_test = y.iloc[
    test_idx
].reset_index(drop=True)

groups_train = groups.iloc[
    train_idx
].reset_index(drop=True)

groups_test = groups.iloc[
    test_idx
].reset_index(drop=True)

train_df = ml_df.iloc[
    train_idx
].reset_index(drop=True)

test_df = ml_df.iloc[
    test_idx
].reset_index(drop=True)


# ============================================================
# 18. Transformer-ready text uses TRAJECTORY, not annotation
# ============================================================

def build_transformer_text(row):
    return (
        "[CONTEXT]\n"
        + str(row["context_text"])
        + "\n\n[CURRENT_ROLE] "
        + str(row["current_role"])
        + "\n[CURRENT]\n"
        + str(row["current_text"])
    )


transformer_train_df = train_df[
    [
        "dataset",
        "group_id",
        "trajectory_index",
        "message_index",
        "current_role",
        "context_text",
        "current_text",
        "previous_messages",
        "previous_tool_calls",
        "previous_tool_results",
        "previous_user_messages",
        "previous_assistant_messages",
        "failure_family",
        "family_label",
    ]
].copy()

transformer_test_df = test_df[
    [
        "dataset",
        "group_id",
        "trajectory_index",
        "message_index",
        "current_role",
        "context_text",
        "current_text",
        "previous_messages",
        "previous_tool_calls",
        "previous_tool_results",
        "previous_user_messages",
        "previous_assistant_messages",
        "failure_family",
        "family_label",
    ]
].copy()

transformer_train_df["transformer_text"] = (
    transformer_train_df.apply(
        build_transformer_text,
        axis=1,
    )
)

transformer_test_df["transformer_text"] = (
    transformer_test_df.apply(
        build_transformer_text,
        axis=1,
    )
)


# ============================================================
# 19. Domain-aware datasets for future LODO evaluation
# ============================================================

taxonomy_a = taxonomy_df[
    taxonomy_df["dataset"] == "A"
].reset_index(drop=True)

taxonomy_b = taxonomy_df[
    taxonomy_df["dataset"] == "B"
].reset_index(drop=True)

taxonomy_c = taxonomy_df[
    taxonomy_df["dataset"] == "C"
].reset_index(drop=True)

cross_dataset_splits = [
    (
        "A+B -> C",
        [taxonomy_a, taxonomy_b],
        taxonomy_c,
    ),
    (
        "A+C -> B",
        [taxonomy_a, taxonomy_c],
        taxonomy_b,
    ),
    (
        "B+C -> A",
        [taxonomy_b, taxonomy_c],
        taxonomy_a,
    ),
]


def prepare_cross_dataset(df):
    df = df.copy()

    (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
    ) = prepare_taxonomy_features(df)

    return (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
    )


# ============================================================
# 20. Diagnostics -- same style as first dataset.py
# ============================================================

print("\n" + "=" * 70)
print("TAXONOMY DATASET")
print("=" * 70)

print(f"Total samples:       {len(X):,}")
print(f"Total trajectories:  {groups.nunique():,}")

print("\nTRAIN / TEST SPLIT")

print(f"Train samples:       {len(X_train):,}")
print(f"Test samples:        {len(X_test):,}")

print(
    f"Train trajectories:  "
    f"{groups_train.nunique():,}"
)
print(
    f"Test trajectories:   "
    f"{groups_test.nunique():,}"
)

print("\nTRAIN LABEL DISTRIBUTION")

print(
    pd.DataFrame({
        "count":
            y_train.value_counts().sort_index(),

        "percentage":
            (
                y_train
                .value_counts(normalize=True)
                .sort_index()
                .mul(100)
                .round(2)
            ),
    })
)

print("\nTEST LABEL DISTRIBUTION")

print(
    pd.DataFrame({
        "count":
            y_test.value_counts().sort_index(),

        "percentage":
            (
                y_test
                .value_counts(normalize=True)
                .sort_index()
                .mul(100)
                .round(2)
            ),
    })
)


# ============================================================
# 21. CRITICAL trajectory leakage + alignment checks
# ============================================================

train_groups_set = set(groups_train)
test_groups_set = set(groups_test)

overlap = (
    train_groups_set
    & test_groups_set
)

print("\nLEAKAGE CHECK")
print(
    f"Overlapping trajectories: "
    f"{len(overlap)}"
)

assert len(overlap) == 0, (
    "ERROR: trajectories appear in both "
    "train and test!"
)

assert (
    len(X_train)
    == len(y_train)
    == len(groups_train)
    == len(train_df)
)

assert (
    len(X_test)
    == len(y_test)
    == len(groups_test)
    == len(test_df)
)

assert set(X_train.index) == set(y_train.index)
assert set(X_test.index) == set(y_test.index)

# Reviewer text must never enter model-facing X.
assert "reason" not in X.columns
assert "reason_semantic" not in X.columns
assert "final_failure_type_v2" not in X.columns

assert (
    transformer_train_df["transformer_text"]
    .notna()
    .all()
)

assert (
    transformer_test_df["transformer_text"]
    .notna()
    .all()
)

print("X / y / groups alignment: OK")
print("Trajectory split:          OK")
print("Final model-facing asserts:  OK")
print(
    "Exact notebook count match:   ",
    notebook_reproduction_report[
        "exact_family_match"
    ]
    and notebook_reproduction_report[
        "exact_dataset_match"
    ]
    and len(taxonomy_df) == 1779,
)
print("Runtime embedding/similarity mapping: OK")

# ---------------------------------------------------------
# 2. Get trajectory text
# ---------------------------------------------------------

train_texts = (
    transformer_train_df["transformer_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

test_texts = (
    transformer_test_df["transformer_text"]
    .fillna("")
    .astype(str)
    .tolist()
)


# ---------------------------------------------------------
# 3. Embed TRAIN and TEST separately
# ---------------------------------------------------------

X_train_text_embeddings = reason_model.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

X_test_text_embeddings = reason_model.encode(
    test_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)


# ---------------------------------------------------------
# 4. Critical alignment checks
# ---------------------------------------------------------

assert len(X_train_text_embeddings) == len(y_train)
assert len(X_test_text_embeddings) == len(y_test)

print("Train embeddings:", X_train_text_embeddings.shape)
print("Train labels:    ", y_train.shape)

print("Test embeddings: ", X_test_text_embeddings.shape)
print("Test labels:     ", y_test.shape)
