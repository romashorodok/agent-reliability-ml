"""
Export the canonical failure-taxonomy dataset.

Expected source:
    scripts/taxonomy_dataset.py

Expected objects exported by taxonomy_dataset.py:
    taxonomy_df
    transformer_train_df
    transformer_test_df
    y_train
    y_test
    groups_train
    groups_test
    cross_dataset_splits

This script:

1. Loads the canonical taxonomy dataset.
2. Validates labels and train/test alignment.
3. Validates group-safe splitting.
4. Reconstructs structured / relational trajectory features.
5. Aligns those features with the canonical train/test split.
6. Exports:
       taxonomy_full.csv
       taxonomy_train.csv
       taxonomy_test.csv
       taxonomy_metadata.json

Run from project root:

    python -m scripts.export_taxonomy_dataset
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FULL_OUTPUT = OUTPUT_DIR / "taxonomy_full.csv"
TRAIN_OUTPUT = OUTPUT_DIR / "taxonomy_train.csv"
TEST_OUTPUT = OUTPUT_DIR / "taxonomy_test.csv"
METADATA_OUTPUT = OUTPUT_DIR / "taxonomy_metadata.json"


LABEL_NAMES = [
    "workflow_error",
    "constraint_error",
    "tool_use_error",
    "grounding_state_error",
    "reasoning_value_error",
]

ID2LABEL = {
    i: name
    for i, name in enumerate(LABEL_NAMES)
}

LABEL2ID = {
    name: i
    for i, name in ID2LABEL.items()
}


# Columns uniquely identifying a row.
#
# message_index alone is NOT unique globally because it resets
# inside trajectories.
KEY_COLS = [
    "dataset",
    "group_id",
    "message_index",
]


# ============================================================
# 2. Load canonical dataset
# ============================================================

def load_canonical_dataset():

    from scripts.taxonomy_dataset import (
        taxonomy_df,
        transformer_train_df,
        transformer_test_df,
        y_train,
        y_test,
        groups_train,
        groups_test,
        cross_dataset_splits,
    )

    return {
        "full": taxonomy_df.copy(),
        "train": transformer_train_df.copy(),
        "test": transformer_test_df.copy(),

        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),

        "groups_train": np.asarray(groups_train),
        "groups_test": np.asarray(groups_test),

        "cross_dataset_splits": cross_dataset_splits,
    }


# ============================================================
# 3. Basic parsing helpers
# ============================================================

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL(?:\s+name=([^\]]+))?\]"
    r"\s*(?:\n\s*)*"
    r"([A-Za-z0-9_.\-]+)?",
    flags=re.IGNORECASE,
)


def safe_text(value) -> str:
    """Convert nullable dataframe value to safe string."""

    if pd.isna(value):
        return ""

    return str(value)


def count_words(text: str) -> int:
    return len(text.split())


def contains_error_signal(text: str) -> int:
    """
    Coarse error-signal detector.

    This is intentionally lexical and should be treated as an
    engineered feature, not ground truth.
    """

    text = safe_text(text).lower()

    patterns = [
        r"\berror\b",
        r"\bfailed\b",
        r"\bfailure\b",
        r"\bexception\b",
        r"\binvalid\b",
        r"\bnot found\b",
        r"\bdenied\b",
        r"\bunauthorized\b",
        r"\bforbidden\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bunable\b",
    ]

    return int(
        any(
            re.search(pattern, text)
            for pattern in patterns
        )
    )


# ============================================================
# 4. Tool parsing
# ============================================================

def extract_tool_names(text: str) -> list[str]:
    """
    Extract tool names from [TOOL_CALL] blocks.

    Works with examples such as:

        [TOOL_CALL]

        search({"query": "..."})

    and:

        [TOOL_CALL name=search]
    """

    text = safe_text(text)

    names = []

    # --------------------------------------------------------
    # Pattern 1:
    #
    # [TOOL_CALL]
    #
    # search(...)
    # --------------------------------------------------------

    pattern = re.compile(
        r"\[TOOL_CALL(?:\s+name=([^\]]+))?\]"
        r"\s*"
        r"(?:\n\s*)*"
        r"([A-Za-z_][A-Za-z0-9_.\-]*)?",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(text):

        explicit_name = match.group(1)
        following_name = match.group(2)

        name = explicit_name or following_name

        if name:
            names.append(name.strip())

    return names


def current_tool_from_text(current_text: str) -> str:

    tools = extract_tool_names(current_text)

    if tools:
        return tools[0]

    return "NO_TOOL"


def previous_tool_from_context(context_text: str) -> str:

    tools = extract_tool_names(context_text)

    if tools:
        return tools[-1]

    return "NO_TOOL"


# ============================================================
# 5. Build base structured features
# ============================================================

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    out["context_text"] = (
        out["context_text"]
        .fillna("")
        .astype(str)
    )

    out["current_text"] = (
        out["current_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Role
    # --------------------------------------------------------

    def infer_role(text):

        text = safe_text(text).lstrip()

        if text.startswith("[TOOL_CALL]"):
            return "TOOL_CALL"

        return "ASSISTANT"

    if "current_role" not in out.columns:
        out["current_role"] = out["current_text"].map(
            infer_role
        )

    # --------------------------------------------------------
    # Current-message properties
    # --------------------------------------------------------

    out["is_tool_call"] = (
        out["current_text"]
        .str.lstrip()
        .str.startswith("[TOOL_CALL]")
        .astype(int)
    )

    out["current_char_length"] = (
        out["current_text"]
        .str.len()
    )

    out["current_word_count"] = (
        out["current_text"]
        .map(count_words)
    )

    # --------------------------------------------------------
    # Context properties
    # --------------------------------------------------------

    out["context_char_length"] = (
        out["context_text"]
        .str.len()
    )

    out["context_word_count"] = (
        out["context_text"]
        .map(count_words)
    )

    # --------------------------------------------------------
    # Tool identity
    # --------------------------------------------------------

    out["current_tool"] = (
        out["current_text"]
        .map(current_tool_from_text)
    )

    out["previous_tool"] = (
        out["context_text"]
        .map(previous_tool_from_context)
    )

    # --------------------------------------------------------
    # Context-presence indicators
    # --------------------------------------------------------

    out["has_previous_tool_call"] = (
        out["context_text"]
        .str.contains(
            r"\[TOOL_CALL",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    out["has_previous_tool_result"] = (
        out["context_text"]
        .str.contains(
            r"\[TOOL_RESULT",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Parsed tool activity
    # --------------------------------------------------------

    out["parsed_tool_calls_in_context"] = (
        out["context_text"]
        .map(
            lambda x: int(
                len(extract_tool_names(x)) > 0
            )
        )
    )

    # --------------------------------------------------------
    # Error signals
    # --------------------------------------------------------

    out["previous_results_have_error"] = (
        out["context_text"]
        .map(contains_error_signal)
    )

    out["context_has_error_signal"] = (
        out["context_text"]
        .map(contains_error_signal)
    )

    return out


# ============================================================
# 6. Build trajectory-history features
# ============================================================

def build_trajectory_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    required = [
        "dataset",
        "group_id",
        "message_index",
        "current_tool",
        "current_role",
    ]

    missing = [
        col
        for col in required
        if col not in out.columns
    ]

    if missing:
        raise ValueError(
            f"Missing trajectory columns: {missing}"
        )

    # Preserve original row ordering.
    out["_original_order"] = np.arange(len(out))

    # Sort chronologically inside each trajectory.
    out = out.sort_values(
        [
            "dataset",
            "group_id",
            "message_index",
            "_original_order",
        ]
    ).copy()

    # Initialize.
    out["same_tool_as_previous"] = 0
    out["current_tool_previous_count"] = 0
    out["current_action_seen_before"] = 0

    # --------------------------------------------------------
    # Process trajectory-by-trajectory
    # --------------------------------------------------------

    for (_, _), idx in out.groupby(
        ["dataset", "group_id"],
        sort=False,
    ).groups.items():

        idx = list(idx)

        tool_counter = Counter()

        previous_tool = None

        for row_idx in idx:

            current_tool = out.at[
                row_idx,
                "current_tool",
            ]

            # Do not treat NO_TOOL as a repeated tool action.
            real_tool = (
                current_tool != "NO_TOOL"
            )

            if (
                real_tool
                and previous_tool is not None
                and current_tool == previous_tool
            ):
                out.at[
                    row_idx,
                    "same_tool_as_previous",
                ] = 1

            previous_count = (
                tool_counter[current_tool]
                if real_tool
                else 0
            )

            out.at[
                row_idx,
                "current_tool_previous_count",
            ] = previous_count

            out.at[
                row_idx,
                "current_action_seen_before",
            ] = int(previous_count > 0)

            if real_tool:
                tool_counter[current_tool] += 1
                previous_tool = current_tool

            elif previous_tool is None:
                previous_tool = "NO_TOOL"

            else:
                previous_tool = "NO_TOOL"

    # Restore canonical order.
    out = (
        out
        .sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    return out


# ============================================================
# 7. Validate labels
# ============================================================

def validate_labels(
    full_df,
    train_df,
    test_df,
    y_train,
    y_test,
):

    valid_ids = set(ID2LABEL.keys())

    assert set(np.unique(y_train)).issubset(valid_ids)
    assert set(np.unique(y_test)).issubset(valid_ids)

    if "family_label" in train_df.columns:

        np.testing.assert_array_equal(
            train_df["family_label"].to_numpy(),
            y_train,
        )

    if "family_label" in test_df.columns:

        np.testing.assert_array_equal(
            test_df["family_label"].to_numpy(),
            y_test,
        )

    if "failure_family" in full_df.columns:

        observed = set(
            full_df["failure_family"]
            .dropna()
            .unique()
        )

        unexpected = observed - set(LABEL_NAMES)

        if unexpected:
            raise ValueError(
                "Unexpected exported failure families: "
                f"{unexpected}"
            )


# ============================================================
# 8. Validate canonical split
# ============================================================

def validate_split(
    train_df,
    test_df,
    y_train,
    y_test,
    groups_train,
    groups_test,
):

    assert len(train_df) == len(y_train)
    assert len(test_df) == len(y_test)

    assert len(train_df) == len(groups_train)
    assert len(test_df) == len(groups_test)

    overlap = (
        set(groups_train)
        & set(groups_test)
    )

    if overlap:
        raise ValueError(
            f"Group leakage detected: {len(overlap)} groups"
        )

    print(
        "✓ Train/test group overlap:",
        len(overlap),
    )


# ============================================================
# 9. Align engineered features to canonical split
# ============================================================

def align_split(
    canonical_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:

    # Never duplicate key columns.
    feature_cols = [
        c
        for c in feature_df.columns
        if c not in KEY_COLS
    ]

    lookup = feature_df[
        KEY_COLS + feature_cols
    ].copy()

    # Keys should identify one exported taxonomy row.
    duplicated = lookup.duplicated(
        KEY_COLS,
        keep=False,
    )

    if duplicated.any():

        examples = (
            lookup.loc[
                duplicated,
                KEY_COLS,
            ]
            .head(10)
        )

        raise ValueError(
            "Feature lookup contains duplicate keys.\n"
            f"{examples}"
        )

    keys = canonical_df[KEY_COLS].copy()

    aligned = keys.merge(
        lookup,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = (
        aligned["_merge"] != "both"
    )

    if missing.any():

        print(
            aligned.loc[
                missing,
                KEY_COLS + ["_merge"],
            ].head(20)
        )

        raise ValueError(
            f"{missing.sum()} canonical rows could not "
            "be aligned to engineered features."
        )

    aligned = aligned.drop(
        columns="_merge"
    )

    # Verify exact key order.
    pd.testing.assert_frame_equal(
        aligned[KEY_COLS].reset_index(drop=True),
        canonical_df[KEY_COLS].reset_index(drop=True),
        check_dtype=False,
    )

    return aligned


# ============================================================
# 10. Create export dataframe
# ============================================================

def create_export_dataframe(
    canonical_df,
    aligned_features,
    labels,
    groups,
    split_name,
):

    result = aligned_features.copy()

    # Canonical text is always taken from the canonical split,
    # not reconstructed.
    result["current_text"] = (
        canonical_df["current_text"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    result["context_text"] = (
        canonical_df["context_text"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    result["family_label"] = np.asarray(labels)

    result["failure_family"] = [
        ID2LABEL[int(label)]
        for label in labels
    ]

    result["split"] = split_name

    # Store group explicitly for downstream scripts.
    result["canonical_group"] = np.asarray(groups)

    # Put important columns first.
    first_cols = [
        "dataset",
        "group_id",
        "message_index",
        "split",
        "family_label",
        "failure_family",
        "current_text",
        "context_text",
    ]

    remaining = [
        col
        for col in result.columns
        if col not in first_cols
    ]

    result = result[
        first_cols + remaining
    ]

    return result


# ============================================================
# 11. Final export validation
# ============================================================

def validate_export(
    train_export,
    test_export,
):

    print("\n" + "=" * 80)
    print("EXPORT VALIDATION")
    print("=" * 80)

    print(
        "Train:",
        train_export.shape,
    )

    print(
        "Test:",
        test_export.shape,
    )

    assert len(train_export) == 1489, (
        f"Expected 1489 train rows, "
        f"got {len(train_export)}"
    )

    assert len(test_export) == 287, (
        f"Expected 287 test rows, "
        f"got {len(test_export)}"
    )

    assert len(train_export) + len(test_export) == 1776

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print("\nTrain labels:")

    print(
        train_export[
            "failure_family"
        ].value_counts()
    )

    print("\nTest labels:")

    print(
        test_export[
            "failure_family"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Group leakage
    # --------------------------------------------------------

    train_groups = set(
        train_export["canonical_group"]
    )

    test_groups = set(
        test_export["canonical_group"]
    )

    overlap = train_groups & test_groups

    print(
        "\nTrain groups:",
        len(train_groups),
    )

    print(
        "Test groups:",
        len(test_groups),
    )

    print(
        "Group overlap:",
        len(overlap),
    )

    assert len(overlap) == 0

    # --------------------------------------------------------
    # Required structured features
    # --------------------------------------------------------

    required_features = [
        "current_role",
        "current_tool",
        "previous_tool",
        "message_index",
        "is_tool_call",
        "current_char_length",
        "current_word_count",
        "context_char_length",
        "context_word_count",
        "previous_messages",
        "previous_tool_calls",
        "previous_assistant_messages",
        "parsed_tool_calls_in_context",
        "same_tool_as_previous",
        "current_tool_previous_count",
        "current_action_seen_before",
        "previous_results_have_error",
        "context_has_error_signal",
        "has_previous_tool_result",
        "has_previous_tool_call",
    ]

    missing = [
        feature
        for feature in required_features
        if feature not in train_export.columns
    ]

    assert not missing, (
        f"Missing structured features: {missing}"
    )

    # --------------------------------------------------------
    # No missing labels
    # --------------------------------------------------------

    assert not train_export[
        "family_label"
    ].isna().any()

    assert not test_export[
        "family_label"
    ].isna().any()

    print("\n✓ Export validation passed")


# ============================================================
# 12. Metadata
# ============================================================

def build_metadata(
    train_export,
    test_export,
):

    metadata = {
        "dataset_name":
            "agent_failure_taxonomy",

        "version":
            1,

        "total_rows":
            int(
                len(train_export)
                + len(test_export)
            ),

        "train_rows":
            int(len(train_export)),

        "test_rows":
            int(len(test_export)),

        "train_groups":
            int(
                train_export[
                    "canonical_group"
                ].nunique()
            ),

        "test_groups":
            int(
                test_export[
                    "canonical_group"
                ].nunique()
            ),

        "label_mapping": {
            str(k): v
            for k, v in ID2LABEL.items()
        },

        "train_distribution": {
            str(k): int(v)
            for k, v in (
                train_export[
                    "family_label"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },

        "test_distribution": {
            str(k): int(v)
            for k, v in (
                test_export[
                    "family_label"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },

        "key_columns":
            KEY_COLS,

        "text_columns": [
            "current_text",
            "context_text",
        ],

        "structured_features": [
            "current_role",
            "current_tool",
            "previous_tool",
            "message_index",
            "is_tool_call",
            "current_char_length",
            "current_word_count",
            "context_char_length",
            "context_word_count",
            "previous_messages",
            "previous_tool_calls",
            "previous_assistant_messages",
            "parsed_tool_calls_in_context",
            "same_tool_as_previous",
            "current_tool_previous_count",
            "current_action_seen_before",
            "previous_results_have_error",
            "context_has_error_signal",
            "has_previous_tool_result",
            "has_previous_tool_call",
        ],
    }

    return metadata


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 80)
    print("CANONICAL TAXONOMY DATASET EXPORT")
    print("=" * 80)

    data = load_canonical_dataset()

    full_df = data["full"]
    canonical_train = data["train"]
    canonical_test = data["test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    groups_train = data["groups_train"]
    groups_test = data["groups_test"]

    print("\nLoaded:")
    print("Full:", full_df.shape)
    print("Train:", canonical_train.shape)
    print("Test:", canonical_test.shape)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_labels(
        full_df,
        canonical_train,
        canonical_test,
        y_train,
        y_test,
    )

    validate_split(
        canonical_train,
        canonical_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    )

    # --------------------------------------------------------
    # Engineer features ONCE from full canonical data.
    # --------------------------------------------------------

    print("\nBuilding relational trajectory features...")

    structured_df = build_base_features(
        full_df
    )

    structured_df = build_trajectory_features(
        structured_df
    )

    print(
        "Structured full:",
        structured_df.shape,
    )

    # --------------------------------------------------------
    # Align with canonical split
    # --------------------------------------------------------

    print("\nAligning train...")

    aligned_train = align_split(
        canonical_train,
        structured_df,
    )

    print("Aligning test...")

    aligned_test = align_split(
        canonical_test,
        structured_df,
    )

    # --------------------------------------------------------
    # Build final exports
    # --------------------------------------------------------

    train_export = create_export_dataframe(
        canonical_train,
        aligned_train,
        y_train,
        groups_train,
        "train",
    )

    test_export = create_export_dataframe(
        canonical_test,
        aligned_test,
        y_test,
        groups_test,
        "test",
    )

    full_export = pd.concat(
        [
            train_export,
            test_export,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_export(
        train_export,
        test_export,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = build_metadata(
        train_export,
        test_export,
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_export.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    test_export.to_csv(
        TEST_OUTPUT,
        index=False,
    )

    full_export.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

    print("Full:    ", FULL_OUTPUT)
    print("Train:   ", TRAIN_OUTPUT)
    print("Test:    ", TEST_OUTPUT)
    print("Metadata:", METADATA_OUTPUT)

    print("\nShapes:")
    print("Full:", full_export.shape)
    print("Train:", train_export.shape)
    print("Test:", test_export.shape)

    print("\n✓ Canonical reusable dataset exported successfully")


if __name__ == "__main__":
    main()