import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from .trajectory import Trajectory, StepToolCall, ToolResultDescription, StepSystemPrompt, StepUser, StepAgent


CONTEXT_WINDOW = 2

dataset_bfcl = "../data/AgentProcessBench/bfcl.jsonl"
dataset_tau2 = "../data/AgentProcessBench/tau2.jsonl"
dataset = "../data/AgentProcessBench/hotpotqa.jsonl"


def step_to_text(step):
    role = step_role(step)

    if isinstance(step, StepToolCall):
        calls = []

        for call in step.tool_calls:
            name = call.schema.get("name", "UNKNOWN_TOOL")
            arguments = call.schema.get("arguments", "")

            calls.append(
                f"{name}({arguments})"
            )

        call_text = "\n".join(calls)

        content = step.content or ""

        return (
            f"[{role}]\n"
            f"{content}\n"
            f"{call_text}"
        ).strip()

    if isinstance(step, ToolResultDescription):
        return (
            f"[{role} name={step.name}]\n"
            f"{step.content or ''}"
        ).strip()

    return (
        f"[{role}]\n"
        f"{getattr(step, 'content', '') or ''}"
    ).strip()

def step_role(step):
    if isinstance(step, StepSystemPrompt):
        return "SYSTEM"

    if isinstance(step, StepUser):
        return "USER"

    if isinstance(step, StepAgent):
        return "ASSISTANT"

    if isinstance(step, StepToolCall):
        return "TOOL_CALL"

    if isinstance(step, ToolResultDescription):
        return "TOOL_RESULT"

    return "UNKNOWN"

def build_context_dataset(
    dataset_path,
    context_window=8,
):
    """
    Create one row per LABELED model decision.

    Each row contains:
      - current step
      - previous context
      - history-only structural features
      - target label

    context_window:
        Number of previous messages to include.
        None = entire previous trajectory.
    """

    trajectory_data = Trajectory(dataset_path)

    trajectories = {}

    # -----------------------------------------
    # Collect steps by trajectory
    # -----------------------------------------

    for step in trajectory_data:
        trajectory_index = getattr(
            step,
            "trajectory_index",
            None
        )

        message_index = getattr(
            step,
            "message_index",
            None
        )

        if trajectory_index is None:
            continue

        if message_index is None:
            continue

        trajectories.setdefault(
            trajectory_index,
            []
        ).append(step)

    rows = []

    # -----------------------------------------
    # Process each trajectory independently
    # -----------------------------------------

    for trajectory_index, steps in trajectories.items():

        steps = sorted(
            steps,
            key=lambda s: s.message_index
        )

        history = []

        previous_tool_calls = 0
        previous_tool_results = 0
        previous_user_messages = 0
        previous_assistant_messages = 0

        for step in steps:

            label = getattr(step, "label", None)

            # ---------------------------------
            # Context BEFORE current step
            # ---------------------------------

            if context_window is None:
                selected_history = history
            else:
                selected_history = history[-context_window:]

            context_text = "\n\n".join(
                item["text"]
                for item in selected_history
            )

            current_text = step_to_text(step)

            # ---------------------------------
            # Only labeled steps become targets
            # ---------------------------------

            if label is not None:

                rows.append({
                    "trajectory_index":
                        trajectory_index,

                    "message_index":
                        step.message_index,

                    "label":
                        int(label),

                    "reason":
                        getattr(step, "reason", None),

                    "current_text":
                        current_text,

                    "context_text":
                        context_text,

                    # Current step type
                    "current_role":
                        step_role(step),

                    "is_tool_call":
                        int(
                            isinstance(
                                step,
                                StepToolCall
                            )
                        ),

                    # History-only features
                    "previous_messages":
                        len(history),

                    "previous_tool_calls":
                        previous_tool_calls,

                    "previous_tool_results":
                        previous_tool_results,

                    "previous_user_messages":
                        previous_user_messages,

                    "previous_assistant_messages":
                        previous_assistant_messages,

                    # Context size
                    "context_char_length":
                        len(context_text),

                    "context_word_count":
                        len(context_text.split()),

                    # Current step size
                    "current_char_length":
                        len(current_text),

                    "current_word_count":
                        len(current_text.split()),
                })

            # ---------------------------------
            # AFTER prediction:
            # current step becomes history
            # ---------------------------------

            history.append({
                "message_index":
                    step.message_index,

                "role":
                    step_role(step),

                "text":
                    current_text,
            })

            if isinstance(step, StepToolCall):
                previous_tool_calls += len(
                    step.tool_calls
                )

            elif isinstance(
                step,
                ToolResultDescription
            ):
                previous_tool_results += 1

            elif isinstance(step, StepUser):
                previous_user_messages += 1

            elif isinstance(step, StepAgent):
                previous_assistant_messages += 1

    return pd.DataFrame(rows)


def prepare_three_context_datasets(window):
    context_a = build_context_dataset(
        dataset,
        context_window=window,
    )

    context_b = build_context_dataset(
        dataset_tau2,
        context_window=window,
    )

    context_c = build_context_dataset(
        dataset_bfcl,
        context_window=window,
    )

    # Unique trajectory groups across datasets
    context_a["group_id"] = (
        "a_" + context_a["trajectory_index"].astype(str)
    )

    context_b["group_id"] = (
        "b_" + context_b["trajectory_index"].astype(str)
    )

    context_c["group_id"] = (
        "c_" + context_c["trajectory_index"].astype(str)
    )

    context_df = pd.concat(
        [context_a, context_b, context_c],
        ignore_index=True,
    )

    return context_df


def prepare_context_features(context_df):
    df = context_df.copy()

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

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = (
        df[df["label"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    df["label"] = df["label"].astype(int)

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
        + ["current_text", "context_text"]
    ].copy()

    y = df["label"].copy()

    groups = df["group_id"].copy()

    return (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
    )

def prepare_ablation_data(df):
    df = df.copy()

    # -----------------------------------------
    # Basic cleanup
    # -----------------------------------------

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

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = (
        df[df["label"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    df["label"] = df["label"].astype(int)

    # -----------------------------------------
    # Extra structural features
    # -----------------------------------------

    df["log_context_char_length"] = np.log1p(
        df["context_char_length"]
    )

    df["log_current_char_length"] = np.log1p(
        df["current_char_length"]
    )

    # -----------------------------------------
    # Structural feature groups
    # -----------------------------------------

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

    y = df["label"].copy()

    groups = df["group_id"].copy()

    return (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
    )

context_df = prepare_three_context_datasets(
   CONTEXT_WINDOW 
)

(
    ml_df,
    X,
    y,
    groups,
    numeric_features,
    categorical_features,
) = prepare_ablation_data(context_df)

# splitter = StratifiedGroupKFold(
#     n_splits=2,
#     shuffle=True,
#     random_state=42,
# )

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]


X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)

y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

groups_train = groups.iloc[train_idx].reset_index(drop=True)
groups_test = groups.iloc[test_idx].reset_index(drop=True)


print("DATASET")

print(f"Total samples:       {len(X):,}")
print(f"Total trajectories:  {groups.nunique():,}")


print("TRAIN / TEST SPLIT")

print(f"Train samples:       {len(X_train):,}")
print(f"Test samples:        {len(X_test):,}")

print(f"Train trajectories:  {groups_train.nunique():,}")
print(f"Test trajectories:   {groups_test.nunique():,}")

# -----------------------------------------
# Class distributions
# -----------------------------------------

print("TRAIN LABEL DISTRIBUTION")

print(
    pd.DataFrame({
        "count": y_train.value_counts().sort_index(),
        "percentage": (
            y_train.value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )
    })
)


print("TEST LABEL DISTRIBUTION")

print(
    pd.DataFrame({
        "count": y_test.value_counts().sort_index(),
        "percentage": (
            y_test.value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )
    })
)

# -----------------------------------------
# CRITICAL: trajectory leakage check
# -----------------------------------------

train_groups = set(groups_train)
test_groups = set(groups_test)

overlap = train_groups & test_groups

print("LEAKAGE CHECK")

print(f"Overlapping trajectories: {len(overlap)}")

assert len(overlap) == 0, (
    "ERROR: trajectories appear in both train and test!"
)

# -----------------------------------------
# Alignment checks
# -----------------------------------------

assert len(X_train) == len(y_train) == len(groups_train)
assert len(X_test) == len(y_test) == len(groups_test)

print("X / y / groups alignment: OK")
print("Trajectory split:          OK")

def prepare_cross_dataset(df):
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

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = (
        df[df["label"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    df["label"] = df["label"].astype(int)

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

    y = df["label"].copy()

    return (
        df,
        X,
        y,
        numeric_features,
        categorical_features,
    )

context_a = build_context_dataset(
    dataset,
    context_window=CONTEXT_WINDOW,
)

context_b = build_context_dataset(
    dataset_tau2,
    context_window=CONTEXT_WINDOW,
)

context_c = build_context_dataset(
    dataset_bfcl,
    context_window=CONTEXT_WINDOW,
)

# Unique trajectory groups across datasets
context_a["group_id"] = (
    "a_" + context_a["trajectory_index"].astype(str)
)

context_b["group_id"] = (
    "b_" + context_b["trajectory_index"].astype(str)
)

context_c["group_id"] = (
    "c_" + context_c["trajectory_index"].astype(str)
)