"""
Export the canonical FULL trajectory dataset for sequential learning.

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
    trajectory_sources

trajectory_sources should map dataset names to raw trajectory collections:

    trajectory_sources = {
        "A": trajectories_a,
        "B": trajectories_b,
        "C": trajectories_c,
    }

No paths are passed to this exporter.

This script:

1. Loads the canonical taxonomy split.
2. Loads every raw message/event from the same source trajectories.
3. Preserves full chronological history.
4. Preserves SYSTEM / USER / ASSISTANT / TOOL_CALL / TOOL_RESULT events.
5. Attaches canonical train/test trajectory membership.
6. Attaches taxonomy targets to the exact labeled current messages.
7. Adds safe historical sequence features.
8. Exports:
       trajectory_events_full.csv
       trajectory_events_train.csv
       trajectory_events_test.csv
       trajectory_targets.csv
       trajectory_metadata.json

Run from project root:

    python -m scripts.export_trajectory_dataset
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FULL_OUTPUT = (
    OUTPUT_DIR
    / "trajectory_events_full.csv"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "trajectory_events_train.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "trajectory_events_test.csv"
)

TARGET_OUTPUT = (
    OUTPUT_DIR
    / "trajectory_targets.csv"
)

METADATA_OUTPUT = (
    OUTPUT_DIR
    / "trajectory_metadata.json"
)


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


TARGET_KEY_COLS = [
    "dataset",
    "group_id",
    "message_index",
]


# ============================================================
# 2. Load canonical data + trajectory sources
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
    )

    from scripts.dataset import (
        context_a,
        context_b,
        context_c,
    )

    return {
        "full":
            taxonomy_df.copy(),

        "train":
            transformer_train_df.copy(),

        "test":
            transformer_test_df.copy(),

        "y_train":
            np.asarray(y_train),

        "y_test":
            np.asarray(y_test),

        "groups_train":
            np.asarray(groups_train),

        "groups_test":
            np.asarray(groups_test),

        # Full message-level trajectory sources
        "trajectory_sources": {
            "A": context_a.copy(),
            "B": context_b.copy(),
            "C": context_c.copy(),
        },
    }

# ============================================================
# 3. Helpers
# ============================================================

def safe_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def count_words(
    text: str,
) -> int:

    return len(
        safe_text(text).split()
    )


def contains_error_signal(
    text: str,
) -> int:

    text = (
        safe_text(text)
        .lower()
    )

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
        r"\btimeout\b",
    ]

    return int(
        any(
            re.search(pattern, text)
            for pattern in patterns
        )
    )


# ============================================================
# 4. Generic object access
# ============================================================

def get_value(
    obj,
    name,
    default=None,
):

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


# ============================================================
# 5. Normalize tool calls
# ============================================================

def normalize_tool_calls(
    step,
) -> list[dict]:

    tool_calls = (
        get_value(
            step,
            "tool_calls",
            [],
        )
        or []
    )

    result = []

    for call_index, call in enumerate(
        tool_calls
    ):

        schema = (
            get_value(
                call,
                "schema",
                {},
            )
            or {}
        )

        tool_name = (
            schema.get("name")
            if isinstance(schema, dict)
            else None
        )

        arguments = (
            schema.get("arguments")
            if isinstance(schema, dict)
            else None
        )

        result.append({
            "call_index":
                get_value(
                    call,
                    "call_index",
                    call_index,
                ),

            "tool_call_id":
                get_value(
                    call,
                    "id",
                ),

            "tool_name":
                tool_name,

            "arguments":
                safe_text(arguments),

            "inferred":
                int(
                    bool(
                        get_value(
                            call,
                            "inferred",
                            False,
                        )
                    )
                ),
        })

    return result


# ============================================================
# 6. Infer event type
# ============================================================

def infer_event_role(
    step,
) -> str:

    class_name = (
        step.__class__.__name__
        .upper()
    )

    if class_name == "STEPTOOLCALL":
        return "TOOL_CALL"

    if class_name == "TOOLRESULTDESCRIPTION":
        return "TOOL_RESULT"

    if class_name == "STEPAGENT":
        return "ASSISTANT"

    if class_name == "STEPUSER":
        return "USER"

    if class_name == "STEPSYSTEMPROMPT":
        return "SYSTEM"

    # Generic fallback.
    role = safe_text(
        get_value(
            step,
            "role",
        )
    ).upper()

    if role == "TOOL":
        return "TOOL_RESULT"

    if role:
        return role

    return "UNKNOWN"


# ============================================================
# 7. Normalize one trajectory event
# ============================================================

def normalize_step(
    dataset: str,
    trajectory_index: int,
    fallback_message_index: int,
    step,
) -> dict:

    event_role = infer_event_role(
        step
    )

    message_index = get_value(
        step,
        "message_index",
        fallback_message_index,
    )

    step_trajectory_index = get_value(
        step,
        "trajectory_index",
        trajectory_index,
    )

    content = safe_text(
        get_value(
            step,
            "content",
            "",
        )
    )

    label = get_value(
        step,
        "label",
    )

    reason = get_value(
        step,
        "reason",
    )

    tool_calls = normalize_tool_calls(
        step
    )

    # --------------------------------------------------------
    # Tool identity
    # --------------------------------------------------------

    tool_names = [
        call["tool_name"]
        for call in tool_calls
        if call["tool_name"]
    ]

    primary_tool = "NO_TOOL"

    if tool_names:
        primary_tool = tool_names[0]

    elif event_role == "TOOL_RESULT":

        result_name = get_value(
            step,
            "name",
        )

        if result_name:
            primary_tool = str(
                result_name
            )

    return {
        "dataset":
            dataset,

        "trajectory_index":
            int(step_trajectory_index),

        "message_index":
            int(message_index),

        "event_role":
            event_role,

        "content":
            content,

        # ----------------------------------------------------
        # Observable tool information
        # ----------------------------------------------------

        "primary_tool":
            primary_tool,

        "tool_call_count":
            len(tool_calls),

        "tool_names_json":
            json.dumps(
                tool_names,
                ensure_ascii=False,
            ),

        "tool_calls_json":
            json.dumps(
                tool_calls,
                ensure_ascii=False,
            ),

        "tool_call_id":
            get_value(
                step,
                "tool_call_id",
            ),

        "tool_result_name":
            get_value(
                step,
                "name",
            ),

        "inferred":
            int(
                bool(
                    get_value(
                        step,
                        "inferred",
                        False,
                    )
                )
                or any(
                    call["inferred"]
                    for call in tool_calls
                )
            ),

        # ----------------------------------------------------
        # Observable event properties
        # ----------------------------------------------------

        "char_length":
            len(content),

        "word_count":
            count_words(content),

        "has_content":
            int(bool(content)),

        "has_error_signal":
            contains_error_signal(
                content
            ),

        "is_system":
            int(
                event_role == "SYSTEM"
            ),

        "is_user":
            int(
                event_role == "USER"
            ),

        "is_assistant":
            int(
                event_role == "ASSISTANT"
            ),

        "is_tool_call":
            int(
                event_role == "TOOL_CALL"
            ),

        "is_tool_result":
            int(
                event_role == "TOOL_RESULT"
            ),

        # ----------------------------------------------------
        # Annotation.
        #
        # Preserve for analysis, but DO NOT feed these into
        # the first sequential model.
        # ----------------------------------------------------

        "raw_step_label":
            label,

        "raw_step_reason":
            reason,
    }


# ============================================================
# 8. Export all source trajectory events
# ============================================================

def current_tool_from_text(
    text: str,
) -> str:

    text = safe_text(text)

    match = re.search(
        r"\[TOOL_CALL(?:\s+name=([^\]]+))?\]"
        r"\s*(?:\n\s*)*"
        r"([A-Za-z_][A-Za-z0-9_.\-]*)?",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return "NO_TOOL"

    explicit = match.group(1)
    following = match.group(2)

    tool = (
        explicit
        or following
    )

    if not tool:
        return "NO_TOOL"

    return tool.strip()

def build_full_event_dataframe(
    trajectory_sources,
) -> pd.DataFrame:

    frames = []

    for dataset, source_df in (
        trajectory_sources.items()
    ):

        print(
            f"Reading dataset {dataset}..."
        )

        df = source_df.copy()

        print(
            "  source shape:",
            df.shape,
        )

        # ----------------------------------------------------
        # Required chronological identifiers
        # ----------------------------------------------------

        required = [
            "group_id",
            "trajectory_index",
            "message_index",
            "current_role",
            "current_text",
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset {dataset} missing "
                f"required columns: {missing}"
            )

        # ----------------------------------------------------
        # Dataset identity
        # ----------------------------------------------------

        df["dataset"] = dataset

        # ----------------------------------------------------
        # Normalize text
        # ----------------------------------------------------

        df["content"] = (
            df["current_text"]
            .fillna("")
            .astype(str)
        )

        df["current_role"] = (
            df["current_role"]
            .fillna("UNKNOWN")
            .astype(str)
        )

        # ----------------------------------------------------
        # Normalize event role
        # ----------------------------------------------------

        def normalize_role(row):

            role = str(
                row["current_role"]
            ).upper()

            text = str(
                row["content"]
            ).lstrip()

            if role in {
                "TOOL_CALL",
                "TOOL_RESULT",
                "ASSISTANT",
                "USER",
                "SYSTEM",
            }:
                return role

            if text.startswith(
                "[TOOL_CALL"
            ):
                return "TOOL_CALL"

            if text.startswith(
                "[TOOL_RESULT"
            ):
                return "TOOL_RESULT"

            if text.startswith(
                "[USER]"
            ):
                return "USER"

            if text.startswith(
                "[SYSTEM]"
            ):
                return "SYSTEM"

            return role

        df["event_role"] = (
            df.apply(
                normalize_role,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # Tool name
        # ----------------------------------------------------

        if "current_tool" in df.columns:

            df["primary_tool"] = (
                df["current_tool"]
                .fillna("NO_TOOL")
                .astype(str)
            )

        else:

            df["primary_tool"] = (
                df["content"]
                .map(
                    current_tool_from_text
                )
            )

        # ----------------------------------------------------
        # Basic event properties
        # ----------------------------------------------------

        df["char_length"] = (
            df["content"]
            .str.len()
        )

        df["word_count"] = (
            df["content"]
            .map(count_words)
        )

        df["has_content"] = (
            df["content"]
            .str.len()
            .gt(0)
            .astype(int)
        )

        df["has_error_signal"] = (
            df["content"]
            .map(
                contains_error_signal
            )
        )

        # ----------------------------------------------------
        # Role flags
        # ----------------------------------------------------

        df["is_system"] = (
            df["event_role"]
            .eq("SYSTEM")
            .astype(int)
        )

        df["is_user"] = (
            df["event_role"]
            .eq("USER")
            .astype(int)
        )

        df["is_assistant"] = (
            df["event_role"]
            .eq("ASSISTANT")
            .astype(int)
        )

        df["is_tool_call"] = (
            df["event_role"]
            .eq("TOOL_CALL")
            .astype(int)
        )

        df["is_tool_result"] = (
            df["event_role"]
            .eq("TOOL_RESULT")
            .astype(int)
        )

        # ----------------------------------------------------
        # Preserve original annotation if available.
        #
        # These are NOT model inputs later.
        # ----------------------------------------------------

        if "label" in df.columns:
            df["raw_step_label"] = (
                df["label"]
            )
        else:
            df["raw_step_label"] = np.nan

        if "reason" in df.columns:
            df["raw_step_reason"] = (
                df["reason"]
            )
        else:
            df["raw_step_reason"] = None

        # ----------------------------------------------------
        # Keep useful source columns
        # ----------------------------------------------------

        keep = [
            "dataset",
            "group_id",
            "trajectory_index",
            "message_index",

            "event_role",
            "current_role",

            "content",
            "primary_tool",

            "char_length",
            "word_count",
            "has_content",
            "has_error_signal",

            "is_system",
            "is_user",
            "is_assistant",
            "is_tool_call",
            "is_tool_result",

            "raw_step_label",
            "raw_step_reason",
        ]

        # Preserve already-computed historical columns
        # where available.
        optional = [
            "context_text",
            "previous_messages",
            "previous_tool_calls",
            "previous_tool_results",
            "previous_user_messages",
            "previous_assistant_messages",
            "context_char_length",
            "context_word_count",
        ]

        for col in optional:
            if col in df.columns:
                keep.append(col)

        frames.append(
            df[keep].copy()
        )

    events = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    events = (
        events
        .sort_values(
            [
                "dataset",
                "group_id",
                "message_index",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\nFull source events:",
        events.shape,
    )

    return events

# ============================================================
# 9. Build trajectory lookup from canonical taxonomy
# ============================================================

def build_trajectory_lookup(
    taxonomy_df: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "dataset",
        "group_id",
    ]

    missing = [
        col
        for col in required
        if col not in taxonomy_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing taxonomy columns: {missing}"
        )

    lookup = taxonomy_df.copy()

    # --------------------------------------------------------
    # Prefer trajectory_index if taxonomy dataset contains it.
    # --------------------------------------------------------

    if "trajectory_index" not in lookup.columns:

        def derive_trajectory_index(
            row,
        ):

            group_id = str(
                row["group_id"]
            )

            match = re.search(
                r"(\d+)$",
                group_id,
            )

            if not match:
                raise ValueError(
                    "Could not derive trajectory index "
                    f"from group_id={group_id}"
                )

            return int(
                match.group(1)
            )

        lookup[
            "trajectory_index"
        ] = lookup.apply(
            derive_trajectory_index,
            axis=1,
        )

    return (
        lookup[
            [
                "dataset",
                "trajectory_index",
                "group_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ============================================================
# 10. Build canonical split lookup
# ============================================================

def build_split_lookup(
    canonical_train,
    canonical_test,
    groups_train,
    groups_test,
):

    train_lookup = (
        canonical_train[
            [
                "dataset",
                "group_id",
            ]
        ]
        .copy()
    )

    train_lookup[
        "canonical_group"
    ] = np.asarray(
        groups_train
    )

    train_lookup[
        "split"
    ] = "train"


    test_lookup = (
        canonical_test[
            [
                "dataset",
                "group_id",
            ]
        ]
        .copy()
    )

    test_lookup[
        "canonical_group"
    ] = np.asarray(
        groups_test
    )

    test_lookup[
        "split"
    ] = "test"


    lookup = pd.concat(
        [
            train_lookup,
            test_lookup,
        ],
        ignore_index=True,
    )

    lookup = (
        lookup[
            [
                "dataset",
                "group_id",
                "canonical_group",
                "split",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicated = lookup.duplicated(
        [
            "dataset",
            "group_id",
        ],
        keep=False,
    )

    if duplicated.any():

        raise ValueError(
            "A trajectory appears in multiple "
            "canonical split assignments."
        )

    return lookup


# ============================================================
# 11. Attach canonical trajectory information
# ============================================================

def attach_canonical_groups(
    events,
    trajectory_lookup,
    split_lookup,
):

    events = events.merge(
        trajectory_lookup,
        on=[
            "dataset",
            "trajectory_index",
        ],
        how="left",
        validate="many_to_one",
        indicator="_trajectory_merge",
    )

    unmatched = (
        events[
            "_trajectory_merge"
        ]
        != "both"
    )

    if unmatched.any():

        print(
            "\nIgnoring source trajectories that "
            "contain no canonical taxonomy targets:"
        )

        print(
            events.loc[
                unmatched,
                [
                    "dataset",
                    "trajectory_index",
                ],
            ]
            .drop_duplicates()
            .head(20)
        )

    events = (
        events.loc[
            ~unmatched
        ]
        .drop(
            columns="_trajectory_merge"
        )
        .reset_index(drop=True)
    )

    events = events.merge(
        split_lookup,
        on=[
            "dataset",
            "group_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if events["split"].isna().any():
        raise ValueError(
            "Some matched trajectories have no "
            "canonical train/test split."
        )

    return events


# ============================================================
# 12. Attach taxonomy targets
# ============================================================

def attach_taxonomy_targets(
    events,
    taxonomy_df,
):

    target_cols = [
        "dataset",
        "group_id",
        "message_index",
        "failure_family",
        "family_label",
    ]

    available = [
        col
        for col in target_cols
        if col in taxonomy_df.columns
    ]

    targets = (
        taxonomy_df[
            available
        ]
        .drop_duplicates(
            [
                "dataset",
                "group_id",
                "message_index",
            ]
        )
        .copy()
    )

    events = events.merge(
        targets,
        on=[
            "dataset",
            "group_id",
            "message_index",
        ],
        how="left",
        validate="one_to_one",
        indicator="_target_merge",
    )

    events[
        "is_taxonomy_target"
    ] = (
        events[
            "_target_merge"
        ]
        == "both"
    ).astype(int)

    events = events.drop(
        columns="_target_merge"
    )

    return events


# ============================================================
# 13. Add chronological sequence features
# ============================================================

def add_sequence_features(
    events,
):

    out = (
        events
        .sort_values(
            [
                "dataset",
                "group_id",
                "message_index",
            ]
        )
        .reset_index(drop=True)
    )

    trajectory_cols = [
        "dataset",
        "group_id",
    ]

    out[
        "event_position"
    ] = (
        out.groupby(
            trajectory_cols
        )
        .cumcount()
    )

    out[
        "trajectory_event_count"
    ] = (
        out.groupby(
            trajectory_cols
        )["message_index"]
        .transform("size")
    )

    denominator = (
        out[
            "trajectory_event_count"
        ]
        - 1
    ).clip(lower=1)

    out[
        "relative_event_position"
    ] = (
        out[
            "event_position"
        ]
        / denominator
    )

    # --------------------------------------------------------
    # Prior-state counts.
    #
    # IMPORTANT:
    # Current event is subtracted so these describe HISTORY
    # only.
    # --------------------------------------------------------

    cumulative_features = {
        "is_tool_call":
            "previous_tool_calls",

        "is_tool_result":
            "previous_tool_results",

        "is_assistant":
            "previous_assistant_messages",

        "is_user":
            "previous_user_messages",

        "has_error_signal":
            "previous_error_signals",
    }

    for source, target in (
        cumulative_features.items()
    ):

        out[target] = (
            out.groupby(
                trajectory_cols
            )[source]
            .cumsum()
            - out[source]
        )

    # --------------------------------------------------------
    # Previous chronological event
    # --------------------------------------------------------

    out[
        "previous_event_role"
    ] = (
        out.groupby(
            trajectory_cols
        )["event_role"]
        .shift(1)
        .fillna("START")
    )

    out[
        "previous_event_tool"
    ] = (
        out.groupby(
            trajectory_cols
        )["primary_tool"]
        .shift(1)
        .fillna("NO_TOOL")
    )

    out[
        "role_transition"
    ] = (
        out[
            "previous_event_role"
        ]
        + "->"
        + out[
            "event_role"
        ]
    )

    out[
        "tool_transition"
    ] = (
        out[
            "previous_event_tool"
        ]
        + "->"
        + out[
            "primary_tool"
        ]
    )

    return out


# ============================================================
# 14. Build explicit target table
# ============================================================

def build_target_table(
    events,
):

    targets = (
        events.loc[
            events[
                "is_taxonomy_target"
            ]
            == 1
        ]
        .copy()
    )

    targets[
        "history_event_count"
    ] = targets[
        "event_position"
    ]

    targets[
        "has_history"
    ] = (
        targets[
            "history_event_count"
        ]
        > 0
    ).astype(int)

    target_cols = [
        "dataset",
        "group_id",
        "canonical_group",
        "split",
        "trajectory_index",
        "message_index",
        "event_position",
        "history_event_count",
        "has_history",
        "event_role",
        "primary_tool",
        "content",
        "family_label",
        "failure_family",
    ]

    return (
        targets[
            target_cols
        ]
        .reset_index(drop=True)
    )


# ============================================================
# 15. Validate export
# ============================================================

def validate_export(
    events,
    targets,
    canonical_train,
    canonical_test,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAJECTORY EXPORT VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        "Events:",
        events.shape,
    )

    print(
        "Targets:",
        targets.shape,
    )

    print(
        "Trajectories:",
        events[
            "canonical_group"
        ].nunique(),
    )

    print(
        "\nEvent roles:"
    )

    print(
        events[
            "event_role"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Exact target count
    # --------------------------------------------------------

    expected_targets = (
        len(canonical_train)
        + len(canonical_test)
    )

    print(
        "\nExpected targets:",
        expected_targets,
    )

    print(
        "Matched targets:",
        len(targets),
    )

    assert (
        len(targets)
        == expected_targets
    ), (
        f"Expected {expected_targets} taxonomy targets, "
        f"found {len(targets)}."
    )

    # --------------------------------------------------------
    # Expected canonical target split sizes
    # --------------------------------------------------------

    assert (
        targets[
            "split"
        ].eq("train").sum()
        == len(canonical_train)
    )

    assert (
        targets[
            "split"
        ].eq("test").sum()
        == len(canonical_test)
    )

    # --------------------------------------------------------
    # Group-safe split
    # --------------------------------------------------------

    train_groups = set(
        events.loc[
            events["split"]
            == "train",
            "canonical_group",
        ]
    )

    test_groups = set(
        events.loc[
            events["split"]
            == "test",
            "canonical_group",
        ]
    )

    overlap = (
        train_groups
        & test_groups
    )

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
    # Raw chronological key uniqueness
    # --------------------------------------------------------

    event_keys = [
        "dataset",
        "group_id",
        "message_index",
    ]

    duplicate_events = (
        events.duplicated(
            event_keys
        ).sum()
    )

    print(
        "\nDuplicate event keys:",
        duplicate_events,
    )

    assert duplicate_events == 0

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    assert (
        targets[
            "family_label"
        ]
        .notna()
        .all()
    )

    print(
        "\nTrain target labels:"
    )

    print(
        targets.loc[
            targets["split"]
            == "train",
            "failure_family",
        ].value_counts()
    )

    print(
        "\nTest target labels:"
    )

    print(
        targets.loc[
            targets["split"]
            == "test",
            "failure_family",
        ].value_counts()
    )

    # --------------------------------------------------------
    # History sanity
    # --------------------------------------------------------

    print(
        "\nHistory length:"
    )

    print(
        targets[
            "history_event_count"
        ].describe(
            percentiles=[
                .5,
                .9,
                .95,
                .99,
            ]
        )
    )

    print(
        "\nTargets without history:",
        (
            targets[
                "history_event_count"
            ]
            == 0
        ).sum(),
    )

    print(
        "\n✓ Full trajectory export validated"
    )


# ============================================================
# 16. Metadata
# ============================================================

def build_metadata(
    events,
    targets,
):

    return {
        "dataset_name":
            "agent_failure_trajectory_sequences",

        "version":
            1,

        "total_events":
            int(len(events)),

        "total_targets":
            int(len(targets)),

        "train_events":
            int(
                events[
                    "split"
                ].eq("train").sum()
            ),

        "test_events":
            int(
                events[
                    "split"
                ].eq("test").sum()
            ),

        "train_targets":
            int(
                targets[
                    "split"
                ].eq("train").sum()
            ),

        "test_targets":
            int(
                targets[
                    "split"
                ].eq("test").sum()
            ),

        "train_groups":
            int(
                events.loc[
                    events["split"]
                    == "train",
                    "canonical_group",
                ].nunique()
            ),

        "test_groups":
            int(
                events.loc[
                    events["split"]
                    == "test",
                    "canonical_group",
                ].nunique()
            ),

        "label_mapping": {
            str(k): v
            for k, v
            in ID2LABEL.items()
        },

        "event_roles": {
            str(k): int(v)
            for k, v
            in (
                events[
                    "event_role"
                ]
                .value_counts()
                .items()
            )
        },

        "history_definition": (
            "For target event t, model history is all events "
            "with the same canonical_group and "
            "message_index < target message_index."
        ),

        "recommended_first_sequence_features": [
            "event_role",
            "primary_tool",
            "has_error_signal",
            "char_length",
            "word_count",
            "relative_event_position",
        ],

        "annotation_columns_not_for_model_input": [
            "raw_step_label",
            "raw_step_reason",
            "family_label",
            "failure_family",
            "is_taxonomy_target",
        ],
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FULL TRAJECTORY SEQUENCE EXPORT")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load canonical taxonomy + full context datasets
    # --------------------------------------------------------

    data = load_canonical_dataset()

    taxonomy_df = data["full"]

    canonical_train = data["train"]
    canonical_test = data["test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    groups_train = data["groups_train"]
    groups_test = data["groups_test"]

    trajectory_sources = data[
        "trajectory_sources"
    ]

    print("\nCanonical taxonomy:")
    print("Full:", taxonomy_df.shape)
    print("Train:", canonical_train.shape)
    print("Test:", canonical_test.shape)

    print("\nTrajectory sources:")

    for name, source_df in (
        trajectory_sources.items()
    ):

        print(
            f"{name}:",
            source_df.shape,
        )

    # --------------------------------------------------------
    # 2. Build FULL message-level event dataframe
    #
    # This should contain all rows from context_a/b/c,
    # not just failure rows.
    # --------------------------------------------------------

    print("\nBuilding full trajectory events...")

    events = build_full_event_dataframe(
        trajectory_sources
    )

    print(
        "Raw full events:",
        events.shape,
    )

    # --------------------------------------------------------
    # 3. Sanity check event identifiers
    # --------------------------------------------------------

    required_event_cols = [
        "dataset",
        "group_id",
        "trajectory_index",
        "message_index",
        "event_role",
        "content",
    ]

    missing_event_cols = [
        col
        for col in required_event_cols
        if col not in events.columns
    ]

    print(
        "Missing event columns:",
        missing_event_cols,
    )

    if missing_event_cols:
        raise ValueError(
            "Full event dataframe is missing required "
            f"columns: {missing_event_cols}"
        )

    print(
        "\nEvent columns:"
    )

    print(
        events.columns.tolist()
    )

    print(
        "\nFirst event rows:"
    )

    print(
        events[
            [
                "dataset",
                "group_id",
                "trajectory_index",
                "message_index",
                "event_role",
            ]
        ].head(20)
    )

    # --------------------------------------------------------
    # 4. Build canonical train/test group lookup
    #
    # IMPORTANT:
    # map by dataset + group_id.
    #
    # Do not attempt to reconstruct group_id from
    # trajectory_index.
    # --------------------------------------------------------

    split_lookup = build_split_lookup(
        canonical_train,
        canonical_test,
        groups_train,
        groups_test,
    )

    print(
        "\nSplit lookup:",
        split_lookup.shape,
    )

    print(
        split_lookup.head()
    )

    # --------------------------------------------------------
    # 5. Verify lookup uniqueness
    # --------------------------------------------------------

    split_key = [
        "dataset",
        "group_id",
    ]

    duplicate_split_keys = (
        split_lookup
        .duplicated(
            split_key
        )
        .sum()
    )

    print(
        "Duplicate split keys:",
        duplicate_split_keys,
    )

    if duplicate_split_keys != 0:
        raise ValueError(
            "Split lookup contains duplicate "
            "dataset/group_id combinations."
        )

    # --------------------------------------------------------
    # 6. Keep only trajectories belonging to the canonical
    # taxonomy train/test experiment
    #
    # This replaces attach_canonical_groups().
    # --------------------------------------------------------

    print(
        "\nAttaching canonical train/test membership..."
    )

    events = events.merge(
        split_lookup,
        on=[
            "dataset",
            "group_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    print(
        "Events after canonical filtering:",
        events.shape,
    )

    if events.empty:
        raise ValueError(
            "No full trajectory events matched "
            "the canonical taxonomy groups."
        )

    # --------------------------------------------------------
    # 7. Check train/test group safety
    # --------------------------------------------------------

    train_event_groups = set(
        events.loc[
            events["split"] == "train",
            "canonical_group",
        ]
    )

    test_event_groups = set(
        events.loc[
            events["split"] == "test",
            "canonical_group",
        ]
    )

    overlap = (
        train_event_groups
        & test_event_groups
    )

    print(
        "\nTrain trajectory groups:",
        len(train_event_groups),
    )

    print(
        "Test trajectory groups:",
        len(test_event_groups),
    )

    print(
        "Group overlap:",
        len(overlap),
    )

    if overlap:
        raise ValueError(
            f"Trajectory group leakage detected: "
            f"{len(overlap)} groups"
        )

    # --------------------------------------------------------
    # 8. Attach taxonomy target labels
    #
    # Only the 1776 canonical failure rows should receive:
    #
    # family_label
    # failure_family
    # is_taxonomy_target = 1
    # --------------------------------------------------------

    print(
        "\nAttaching taxonomy targets..."
    )

    events = attach_taxonomy_targets(
        events,
        taxonomy_df,
    )

    print(
        "Taxonomy targets matched:",
        int(
            events[
                "is_taxonomy_target"
            ].sum()
        ),
    )

    # --------------------------------------------------------
    # 9. Add chronological sequence features
    # --------------------------------------------------------

    print(
        "\nBuilding chronological sequence features..."
    )

    events = add_sequence_features(
        events
    )

    # --------------------------------------------------------
    # 10. Build explicit target table
    # --------------------------------------------------------

    targets = build_target_table(
        events
    )

    print(
        "\nTarget table:",
        targets.shape,
    )

    # --------------------------------------------------------
    # 11. Validate exact expected target counts
    # --------------------------------------------------------

    expected_train_targets = len(
        canonical_train
    )

    expected_test_targets = len(
        canonical_test
    )

    actual_train_targets = int(
        targets[
            "split"
        ].eq("train").sum()
    )

    actual_test_targets = int(
        targets[
            "split"
        ].eq("test").sum()
    )

    print(
        "\nExpected train targets:",
        expected_train_targets,
    )

    print(
        "Actual train targets:",
        actual_train_targets,
    )

    print(
        "Expected test targets:",
        expected_test_targets,
    )

    print(
        "Actual test targets:",
        actual_test_targets,
    )

    if (
        actual_train_targets
        != expected_train_targets
    ):
        raise ValueError(
            "Train target count mismatch: "
            f"expected {expected_train_targets}, "
            f"got {actual_train_targets}"
        )

    if (
        actual_test_targets
        != expected_test_targets
    ):
        raise ValueError(
            "Test target count mismatch: "
            f"expected {expected_test_targets}, "
            f"got {actual_test_targets}"
        )

    # --------------------------------------------------------
    # 12. Full validation
    # --------------------------------------------------------

    validate_export(
        events,
        targets,
        canonical_train,
        canonical_test,
    )

    # --------------------------------------------------------
    # 13. Split complete event histories
    # --------------------------------------------------------

    train_events = (
        events.loc[
            events["split"]
            == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_events = (
        events.loc[
            events["split"]
            == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\nEvent split sizes:"
    )

    print(
        "Train events:",
        train_events.shape,
    )

    print(
        "Test events:",
        test_events.shape,
    )

    # --------------------------------------------------------
    # 14. Additional history sanity checks
    # --------------------------------------------------------

    print(
        "\nHistory-event-count distribution:"
    )

    print(
        targets[
            "history_event_count"
        ].describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\nTargets with zero history:",
        int(
            targets[
                "history_event_count"
            ].eq(0).sum()
        ),
    )

    print(
        "\nEvent-role distribution:"
    )

    print(
        events[
            "event_role"
        ].value_counts()
    )

    # --------------------------------------------------------
    # 15. Metadata
    # --------------------------------------------------------

    metadata = build_metadata(
        events,
        targets,
    )

    # --------------------------------------------------------
    # 16. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 17. Export
    # --------------------------------------------------------

    print(
        "\nWriting files..."
    )

    events.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    train_events.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    test_events.to_csv(
        TEST_OUTPUT,
        index=False,
    )

    targets.to_csv(
        TARGET_OUTPUT,
        index=False,
    )

    with METADATA_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 18. Final report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAJECTORY EXPORT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nFiles:"
    )

    print(
        "Full events:",
        FULL_OUTPUT,
    )

    print(
        "Train events:",
        TRAIN_OUTPUT,
    )

    print(
        "Test events:",
        TEST_OUTPUT,
    )

    print(
        "Targets:",
        TARGET_OUTPUT,
    )

    print(
        "Metadata:",
        METADATA_OUTPUT,
    )

    print(
        "\nShapes:"
    )

    print(
        "Full events:",
        events.shape,
    )

    print(
        "Train events:",
        train_events.shape,
    )

    print(
        "Test events:",
        test_events.shape,
    )

    print(
        "Targets:",
        targets.shape,
    )

    print(
        "\nTarget split:"
    )

    print(
        targets[
            "split"
        ].value_counts()
    )

    print(
        "\nTarget labels:"
    )

    print(
        targets[
            "failure_family"
        ].value_counts()
    )

    print(
        "\n✓ Full ordered trajectory dataset "
        "ready for sequence learning."
    )


if __name__ == "__main__":
    main()