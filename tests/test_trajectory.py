import json

from pi_deepswe.trajectory import convert_events, read_events


def event(role, content, **kwargs):
    return {
        "type": "message_end",
        "message": {"role": role, "content": content, "timestamp": 1700000000000, **kwargs},
    }


def usage(input=10, output=3):
    return {
        "input": input,
        "output": output,
        "cacheRead": 2,
        "cacheWrite": 1,
        "reasoning": 1,
        "cost": {"total": 0.05},
    }


def convert(events, **kwargs):
    return convert_events(events, version="1.1.0", model_name="benchmark/test", **kwargs)


def test_authoritative_messages_tools_reasoning_and_no_delta_double_count():
    call = {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {"command": "pwd"}}
    events = [
        {"type": "session", "id": "session-1"},
        event("user", "Task\u2028with a Unicode separator"),
        {"type": "message_update", "usage": usage(), "assistantMessageEvent": {"delta": "Hi"}},
        event("assistant", [{"type": "thinking", "thinking": "Use bash"}, call], usage=usage()),
        {"type": "turn_end", "message": {"usage": usage()}},
        event("toolResult", [{"type": "text", "text": "/app"}], toolCallId="call-1"),
        event("toolResult", [{"type": "text", "text": "duplicate"}], toolCallId="call-1"),
        event("assistant", [{"type": "text", "text": "Done"}], usage=usage()),
    ]
    result = convert(events)
    assert result.session_id == "session-1"
    assert len(result.steps) == 3
    assert result.steps[1].message == ""
    assert result.steps[1].reasoning_content == "Use bash"
    assert result.steps[1].observation.results[0].source_call_id == "call-1"
    assert len(result.steps[1].observation.results) == 1
    assert result.final_metrics.total_prompt_tokens == 26
    # Reasoning tokens are already included in output, never add them twice.
    assert result.final_metrics.total_completion_tokens == 6
    assert result.final_metrics.total_cost_usd is None
    assert convert(events, priced=True).final_metrics.total_cost_usd == 0.1


def test_retries_and_compaction_usage_are_preserved():
    events = [
        event("assistant", [], stopReason="error", errorMessage="429", usage=usage(0, 0)),
        {"type": "auto_retry_start", "attempt": 1},
        {
            "type": "compaction_end",
            "result": {"summary": "older work", "usage": usage(7, 2)},
            "aborted": False,
        },
        event("assistant", [{"type": "text", "text": "Done"}], usage=usage(20, 4)),
    ]
    result = convert(events)
    assert len(result.steps) == 2
    assert result.final_metrics.extra["summarization_count"] == 1
    assert result.final_metrics.extra["llm_call_count"] == 3
    assert result.final_metrics.total_prompt_tokens == 36
    assert result.final_metrics.total_completion_tokens == 6
    assert result.final_metrics.extra["peak_context_tokens"] == 23
    assert result.extra["runtime_events"][0]["type"] == "auto_retry_start"


def test_missing_usage_remains_unknown():
    result = convert([event("assistant", [], usage=usage()), event("assistant", [])])
    assert result.final_metrics.total_prompt_tokens is None
    assert result.final_metrics.total_completion_tokens is None
    assert result.final_metrics.extra["usage_complete"] is False


def test_json_framing_and_partial_trailing_record(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(event("user", "hello\u2028world"), ensure_ascii=False) + '\n{"type":'
    )
    events, malformed = read_events(path)
    assert len(events) == 1 and malformed == [2]
    result = convert(events, malformed_lines=malformed)
    assert result.steps[0].message == "hello\u2028world"
    assert result.extra["malformed_event_lines"] == [2]
