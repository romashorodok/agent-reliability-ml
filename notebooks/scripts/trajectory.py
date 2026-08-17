from dataclasses import dataclass
import json
import re

@dataclass
class ToolResultDescription:
    content: str
    name: str | None
    tool_call_id: str | None
    trajectory_index: int
    message_index: int
    inferred: bool = False
    label: int | None = None
    reason: str | None = None

@dataclass
class ToolCallDescription:
    id: str
    type: str
    schema: dict[str, str]
    call_index: int
    result: ToolResultDescription | None
    inferred: bool = False

@dataclass
class StepToolCall:
    content: str
    tool_calls: list[ToolCallDescription]
    trajectory_index: int
    message_index: int
    label: int | None = None
    reason: str | None = None

@dataclass
class StepAgent:
    content: str
    trajectory_index: int
    message_index: int
    label: int | None = None
    reason: str | None = None

@dataclass
class StepSystemPrompt:
    content: str
    label: int | None = None
    reason: str | None = None

@dataclass
class StepUser:
    content: str
    label: int | None = None
    reason: str | None = None

def _parse_legacy_tool_calls(content: str | None) -> list[dict]:
    if not content:
        return []

    calls = []
    for payload in re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        content,
        flags=re.DOTALL,
    ):
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            calls.append(call)
    return calls

_trajectory_mapper = {
    "system": lambda m: StepSystemPrompt(m.get("content")),
    "user": lambda m: StepUser(m.get("content")),
}

class Trajectory:

    def __init__(self, filename: str):
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as f:
            for trajectory_index, line in enumerate(f):
                _line = json.loads(line)
                if messages := _line.get("messages"):
                    # Label keys are stringified, zero-based message indices.
                    step_labels = {
                        int(message_index): label
                        for message_index, label in _line.get("step_labels", {}).items()
                    }
                    step_reasons = {
                        int(message_index): reason
                        for message_index, reason in (
                            _line.get("explanations", {}).get("steps", {}).items()
                        )
                    }
                    # Results are joined by ID, never by their position.
                    results_by_id = {
                        message["tool_call_id"]: ToolResultDescription(
                            content=message.get("content"),
                            name=message.get("name"),
                            tool_call_id=message["tool_call_id"],
                            trajectory_index=trajectory_index,
                            message_index=message_index,
                            inferred=False,
                        )
                        for message_index, message in enumerate(messages)
                        if message.get("role") == "tool"
                        and message.get("tool_call_id")
                    }

                    # Older rows encode calls inside assistant content and
                    # omit IDs/names from tool results. Reconstruct a stable
                    # batch using adjacent result order. This is an inference
                    # because those IDs are not present in the raw dataset.
                    legacy_batches = {}
                    legacy_results_by_message_index = {}
                    for assistant_index, assistant_message in enumerate(messages):
                        if assistant_message.get("role") != "assistant":
                            continue
                        if assistant_message.get("tool_calls"):
                            continue

                        legacy_calls = _parse_legacy_tool_calls(
                            assistant_message.get("content")
                        )
                        if not legacy_calls:
                            continue

                        result_indices = []
                        result_index = assistant_index + 1
                        while (
                            result_index < len(messages)
                            and messages[result_index].get("role") == "tool"
                        ):
                            result_indices.append(result_index)
                            result_index += 1

                        calls = []
                        for call_index, legacy_call in enumerate(legacy_calls):
                            tool_call_id = (
                                f"legacy_call_{trajectory_index}_"
                                f"{assistant_index}_{call_index}"
                            )
                            result = None
                            if call_index < len(result_indices):
                                tool_result_index = result_indices[call_index]
                                tool_result_message = messages[tool_result_index]
                                result = ToolResultDescription(
                                    content=tool_result_message.get("content"),
                                    name=legacy_call["name"],
                                    tool_call_id=tool_call_id,
                                    trajectory_index=trajectory_index,
                                    message_index=tool_result_index,
                                    inferred=True,
                                )
                                legacy_results_by_message_index[
                                    tool_result_index
                                ] = result

                            calls.append(
                                ToolCallDescription(
                                    id=tool_call_id,
                                    type="function",
                                    schema={
                                        "name": legacy_call["name"],
                                        "arguments": json.dumps(
                                            legacy_call.get("arguments", {})
                                        ),
                                    },
                                    call_index=call_index,
                                    result=result,
                                    inferred=True,
                                )
                            )

                        legacy_batches[assistant_index] = StepToolCall(
                            content=assistant_message.get("content"),
                            tool_calls=calls,
                            trajectory_index=trajectory_index,
                            message_index=assistant_index,
                            label=step_labels.get(assistant_index),
                            reason=step_reasons.get(assistant_index),
                        )

                    for message_index, message in enumerate(messages):
                        if message_index in legacy_batches:
                            yield legacy_batches[message_index]
                            continue

                        if tool_calls := message.get("tool_calls"):
                            # One StepToolCall is one possibly-parallel batch.
                            # List order is retained with call_index; results
                            # retain their actual arrival position separately.
                            yield StepToolCall(
                                content=message.get("content"),
                                trajectory_index=trajectory_index,
                                message_index=message_index,
                                label=step_labels.get(message_index),
                                reason=step_reasons.get(message_index),
                                tool_calls=[
                                    ToolCallDescription(
                                        id=tool_call["id"],
                                        type=tool_call["type"],
                                        schema=tool_call["function"],
                                        call_index=call_index,
                                        result=results_by_id.get(tool_call["id"]),
                                        inferred=False,
                                    )
                                    for call_index, tool_call in enumerate(tool_calls)
                                ],
                            )
                            continue

                        # Yield every result at its original conversation
                        # position. ID-bearing results are also attached to
                        # their StepToolCall above for convenient pairing.
                        if message.get("role") == "tool":
                            if tool_call_id := message.get("tool_call_id"):
                                yield results_by_id[tool_call_id]
                            elif result := legacy_results_by_message_index.get(
                                message_index
                            ):
                                yield result
                            else:
                                # Defensive fallback for malformed legacy rows.
                                yield ToolResultDescription(
                                    content=message.get("content"),
                                    name=None,
                                    tool_call_id=(
                                        f"legacy_unmatched_{trajectory_index}_"
                                        f"{message_index}"
                                    ),
                                    trajectory_index=trajectory_index,
                                    message_index=message_index,
                                    inferred=True,
                                )
                            continue

                        if message.get("role") == "assistant":
                            yield StepAgent(
                                content=message.get("content"),
                                trajectory_index=trajectory_index,
                                message_index=message_index,
                                label=step_labels.get(message_index),
                                reason=step_reasons.get(message_index),
                            )
                            continue

                        if mapper := _trajectory_mapper.get(message.get("role")):
                            yield mapper(message)
                