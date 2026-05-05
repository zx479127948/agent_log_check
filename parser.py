"""JSONL log parser for Claude Code agent sessions."""

import json
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ContentBlock:
    type: str  # text / thinking / tool_use / tool_result
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    error_content: str | None = None


@dataclass
class Message:
    type: str  # user / assistant
    timestamp: str | None = None
    role: str | None = None
    content: list[ContentBlock] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass
class LogSession:
    name: str
    log_path: str
    messages: list[Message] = field(default_factory=list)
    first_timestamp: str | None = None
    last_timestamp: str | None = None


@dataclass
class LogGroup:
    main: LogSession
    subagents: list[dict]  # list of {"meta": dict, "session": LogSession}


def _parse_content_block(block: dict) -> ContentBlock:
    """Parse a single content block from JSONL."""
    btype = block.get("type", "")
    if btype == "text":
        return ContentBlock(type="text", text=block.get("text", ""))
    elif btype == "thinking":
        return ContentBlock(type="thinking", text=block.get("thinking", ""))
    elif btype == "tool_use":
        return ContentBlock(
            type="tool_use",
            tool_name=block.get("name"),
            tool_input=block.get("input"),
            tool_use_id=block.get("id"),
        )
    elif btype == "tool_result":
        is_err = block.get("is_error", False)
        err_content = None
        if is_err:
            rc = block.get("content", "")
            err_content = rc if isinstance(rc, str) else str(rc)
        return ContentBlock(
            type="tool_result",
            tool_use_id=block.get("tool_use_id"),
            is_error=bool(is_err),
            error_content=err_content,
        )
    return ContentBlock(type=btype)


def _parse_message(line: dict) -> Message | None:
    """Parse a single JSONL line into a Message."""
    msg_type = line.get("type", "")
    if msg_type not in ("user", "assistant"):
        return None

    timestamp = line.get("timestamp")
    msg = line.get("message", {})
    role = msg.get("role") if isinstance(msg, dict) else None

    # Parse content blocks
    content_raw = msg.get("content", []) if isinstance(msg, dict) else []
    blocks = []
    if isinstance(content_raw, str):
        if content_raw:
            blocks.append(ContentBlock(type="text", text=content_raw))
    elif isinstance(content_raw, list):
        for c in content_raw:
            if isinstance(c, dict):
                blocks.append(_parse_content_block(c))

    # Parse usage
    usage = None
    if isinstance(msg, dict) and "usage" in msg:
        u = msg["usage"]
        usage = TokenUsage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
        )

    return Message(
        type=msg_type,
        timestamp=timestamp,
        role=role,
        content=blocks,
        usage=usage,
    )


def parse_jsonl(filepath: str) -> LogSession:
    """Parse a single JSONL file into a LogSession."""
    messages = []
    timestamps = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = _parse_message(obj)
            if msg is not None:
                messages.append(msg)
                if msg.timestamp:
                    timestamps.append(msg.timestamp)

    name = os.path.splitext(os.path.basename(filepath))[0]
    return LogSession(
        name=name,
        log_path=filepath,
        messages=messages,
        first_timestamp=timestamps[0] if timestamps else None,
        last_timestamp=timestamps[-1] if timestamps else None,
    )


def parse_log_dir(log_dir: str) -> LogGroup:
    """Parse a log directory into a LogGroup (main + subagents)."""
    # Find main agent JSONL (root level .jsonl file)
    main_jsonl = None
    for f in os.listdir(log_dir):
        if f.endswith(".jsonl"):
            main_jsonl = os.path.join(log_dir, f)
            break

    if main_jsonl is None:
        raise FileNotFoundError(f"No .jsonl file found in {log_dir}")

    main_session = parse_jsonl(main_jsonl)

    # Find subagents
    subagents = []
    sub_dir = os.path.join(log_dir, main_session.name, "subagents")
    if os.path.isdir(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            if f.endswith(".jsonl"):
                sub_path = os.path.join(sub_dir, f)
                meta_path = sub_path.replace(".jsonl", ".meta.json")
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                sub_session = parse_jsonl(sub_path)
                subagents.append({"meta": meta, "session": sub_session})

    return LogGroup(main=main_session, subagents=subagents)