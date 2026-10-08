"""Convert authoritative pi JSON messages to ATIF without counting deltas twice."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


def read_events(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    if not path.exists():
        return [], []
    events, malformed = [], []
    # JSON framing is LF only: U+2028/U+2029 can be valid characters in strings.
    for number, line in enumerate(path.read_text().split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSON event is not an object")
            events.append(value)
        except (ValueError, TypeError):
            malformed.append(number)
    return events, malformed


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")


def timestamp(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, UTC).isoformat()
    return value if isinstance(value, str) else None


def usage_metrics(usage: Any, *, priced: bool) -> Metrics:
    usage = usage if isinstance(usage, dict) else {}
    fields = [usage.get(key) for key in ("input", "cacheRead", "cacheWrite")]
    prompt = sum(fields) if all(isinstance(v, int) for v in fields) else None
    return Metrics(
        prompt_tokens=prompt,
        completion_tokens=usage.get("output"),
        cached_tokens=usage.get("cacheRead"),
        cost_usd=(usage.get("cost") or {}).get("total") if priced else None,
        extra={"reasoning_tokens": usage.get("reasoning"), "cache_write_tokens": fields[2]},
    )


def convert_events(
    events: list[dict[str, Any]],
    *,
    version: str,
    model_name: str,
    priced: bool = False,
    malformed_lines: list[int] | None = None,
) -> Trajectory | None:
    steps: list[Step] = []
    tool_steps: dict[str, Step] = {}
    seen_tools: set[str] = set()
    compaction_metrics: list[Metrics] = []
    compactions = 0
    session_id = None
    runtime_events = []
    for event in events:
        kind = event.get("type", "")
        if kind == "session":
            session_id = event.get("id")
        elif kind == "compaction_end":
            result = event.get("result")
            if result and not event.get("aborted"):
                compactions += 1
                compaction_metrics.append(usage_metrics(result.get("usage"), priced=priced))
            runtime_events.append(event)
        elif kind.startswith(("auto_retry_", "compaction_", "summarization_retry_")):
            runtime_events.append(event)
        elif kind == "message_end":
            msg = event.get("message", {})
            role = msg.get("role")
            content = msg.get("content", [])
            if role in {"user", "system"}:
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        source=role,
                        timestamp=timestamp(msg.get("timestamp")),
                        message=text_content(content),
                    )
                )
            elif role == "assistant":
                blocks = content if isinstance(content, list) else []
                calls = [
                    ToolCall(
                        tool_call_id=b["id"],
                        function_name=b["name"],
                        arguments=b.get("arguments", {}),
                    )
                    for b in blocks
                    if b.get("type") == "toolCall"
                ]
                step = Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    timestamp=timestamp(msg.get("timestamp")),
                    message=text_content(content),
                    reasoning_content="\n".join(
                        b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
                    )
                    or None,
                    model_name=msg.get("responseModel") or msg.get("model") or model_name,
                    reasoning_effort=msg.get("thinkingLevel"),
                    tool_calls=calls or None,
                    metrics=usage_metrics(msg.get("usage"), priced=priced),
                    llm_call_count=1,
                    extra={
                        "stop_reason": msg.get("stopReason"),
                        "error_message": msg.get("errorMessage"),
                    },
                )
                steps.append(step)
                for call in calls:
                    tool_steps[call.tool_call_id] = step
            elif role == "toolResult":
                call_id = msg.get("toolCallId")
                step = tool_steps.get(call_id)
                if step is not None and call_id not in seen_tools:
                    seen_tools.add(call_id)
                    if step.observation is None:
                        step.observation = Observation(results=[])
                    step.observation.results.append(
                        ObservationResult(
                            source_call_id=call_id,
                            content=text_content(content),
                            extra={
                                "is_error": msg.get("isError", False),
                                "timestamp": timestamp(msg.get("timestamp")),
                                "raw_content": content,
                            },
                        )
                    )
    if not steps:
        return None
    metrics = [s.metrics for s in steps if s.source == "agent" and s.metrics is not None]
    metrics.extend(compaction_metrics)

    def total(field: str) -> int | float | None:
        values = [getattr(m, field) for m in metrics]
        return sum(values) if values and all(v is not None for v in values) else None

    contexts = [m.prompt_tokens for m in metrics if m.prompt_tokens is not None]
    extra = {
        "peak_context_tokens": max(contexts) if contexts else None,
        "summarization_count": compactions,
        "compaction_usage": [m.model_dump() for m in compaction_metrics],
        "llm_call_count": len(metrics),
        "usage_complete": all(
            m.prompt_tokens is not None and m.completion_tokens is not None for m in metrics
        )
        and bool(metrics),
    }
    return Trajectory(
        session_id=session_id,
        agent=Agent(name="pi", version=version, model_name=model_name),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total("prompt_tokens"),
            total_completion_tokens=total("completion_tokens"),
            total_cached_tokens=total("cached_tokens"),
            total_cost_usd=total("cost_usd"),
            total_steps=len(steps),
            extra=extra,
        ),
        extra={
            "runtime_events": runtime_events,
            "malformed_event_lines": malformed_lines or [],
            "cost_source": "configured_prices" if priced else "unknown",
        },
    )
