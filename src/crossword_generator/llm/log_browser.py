"""Browse and replay structured LLM call logs."""

from __future__ import annotations

import difflib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from crossword_generator.config import ClaudeConfig, OllamaConfig, find_project_root
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.claude_provider import ClaudeProvider
from crossword_generator.llm.ollama_provider import OllamaProvider


@dataclass(frozen=True)
class LLMLogProblem:
    """Non-fatal issue encountered while loading logs."""

    path: Path
    message: str
    line_number: int | None = None


@dataclass(frozen=True)
class LLMLogRecord:
    """Normalized view of one structured LLM call log record."""

    raw: dict[str, object]
    source_path: Path
    line_number: int
    request_id: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    provider: str
    model: str | None
    context: dict[str, object]
    request: dict[str, object]
    response: dict[str, object] | None
    usage: dict[str, int | float]
    cost: dict[str, object]
    error: dict[str, object] | None

    @property
    def step(self) -> str | None:
        return _string_or_none(self.context.get("step"))

    @property
    def difficulty(self) -> str | None:
        return _string_or_none(self.context.get("difficulty"))

    @property
    def size(self) -> int | None:
        return _int_or_none(
            self.context.get("grid_size", self.context.get("size"))
        )

    @property
    def seed(self) -> int | None:
        return _int_or_none(self.context.get("seed"))

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def prompt_text(self) -> str:
        prompt = self.request.get("prompt")
        if isinstance(prompt, str):
            return prompt
        messages = self.request.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "user":
                    return _content_to_text(message.get("content"))
        return ""

    @property
    def system_text(self) -> str | None:
        system_text = self.request.get("system_text")
        if isinstance(system_text, str):
            return system_text
        system = self.request.get("system")
        text = _content_to_text(system).strip()
        return text or None

    @property
    def response_text(self) -> str:
        if self.response is None:
            return ""
        text = self.response.get("text")
        return text if isinstance(text, str) else ""

    @property
    def temperature(self) -> float | None:
        value = self.request.get("temperature")
        if value is None:
            options = self.request.get("options")
            if isinstance(options, dict):
                value = options.get("temperature")
        return _float_or_none(value)

    @property
    def estimated_cost_usd(self) -> float | None:
        return _float_or_none(self.cost.get("estimated_cost_usd"))

    def label(self) -> str:
        parts = [
            self.step or "?",
            self.model or "?",
            self.difficulty or "?",
            f"{self.size}x{self.size}" if self.size else "?",
            f"seed {self.seed}" if self.seed is not None else "seed ?",
        ]
        return " | ".join(parts)


@dataclass(frozen=True)
class LLMLogFilters:
    """Filters applied to normalized LLM log records."""

    step: str | None = None
    model: str | None = None
    seed: int | None = None
    difficulty: str | None = None
    size: int | None = None
    errors_only: bool = False
    text: str | None = None


@dataclass(frozen=True)
class LLMReplayRequest:
    """Transient replay settings for one logged LLM call."""

    model: str | None = None
    temperature: float | None = None
    system: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class LLMReplayArtifact:
    """Files written for a replay experiment."""

    directory: Path
    baseline_path: Path
    variant_path: Path
    comparison_path: Path
    response: LLMCallResponse


ProviderFactory = Callable[[str], LLMProvider]


def load_llm_logs(
    target: Path | str | None,
    *,
    filters: LLMLogFilters | None = None,
    project_root: Path | None = None,
) -> tuple[list[LLMLogRecord], list[LLMLogProblem]]:
    """Load normalized LLM log records from a file, manifest, or directory."""
    root = project_root or find_project_root()
    target_path = _resolve_target(target, root)
    paths, problems = discover_llm_log_paths(target_path)
    records: list[LLMLogRecord] = []

    for path in paths:
        file_records, file_problems = _read_jsonl_log(path)
        records.extend(file_records)
        problems.extend(file_problems)

    records.sort(key=lambda r: (r.started_at or "", str(r.source_path), r.line_number))
    if filters is not None:
        records = filter_llm_log_records(records, filters)
    return records, problems


def discover_llm_log_paths(target: Path) -> tuple[list[Path], list[LLMLogProblem]]:
    """Resolve a target into structured LLM JSONL files."""
    problems: list[LLMLogProblem] = []
    if not target.exists():
        return [], [LLMLogProblem(target, "Target does not exist")]

    if target.is_file():
        if target.name == "manifest.json":
            return _paths_from_manifest(target)
        if target.name.endswith(".llm.jsonl") or target.suffix == ".jsonl":
            return [target], problems
        return [], [LLMLogProblem(target, "Expected manifest.json or .jsonl file")]

    manifest = target / "manifest.json"
    if manifest.exists():
        return _paths_from_manifest(manifest)

    paths = sorted(target.rglob("*.llm.jsonl"))
    if not paths:
        problems.append(LLMLogProblem(target, "No .llm.jsonl files found"))
    return paths, problems


def filter_llm_log_records(
    records: list[LLMLogRecord],
    filters: LLMLogFilters,
) -> list[LLMLogRecord]:
    """Apply exact field filters plus optional free-text search."""
    text = filters.text.lower().strip() if filters.text else ""
    result: list[LLMLogRecord] = []
    for record in records:
        if filters.step and record.step != filters.step:
            continue
        if filters.model and record.model != filters.model:
            continue
        if filters.seed is not None and record.seed != filters.seed:
            continue
        if filters.difficulty and record.difficulty != filters.difficulty:
            continue
        if filters.size is not None and record.size != filters.size:
            continue
        if filters.errors_only and not record.has_error:
            continue
        if text and text not in _record_search_text(record):
            continue
        result.append(record)
    return result


def replay_llm_record(
    record: LLMLogRecord,
    replay: LLMReplayRequest,
    *,
    experiment_root: Path | str | None = None,
    provider_factory: ProviderFactory | None = None,
    project_root: Path | None = None,
) -> LLMReplayArtifact:
    """Run a transient replay of one logged call and write artifacts."""
    if record.provider not in {"claude", "ollama"}:
        raise ValueError(f"Unsupported replay provider: {record.provider}")

    root = project_root or find_project_root()
    output_root = Path(experiment_root) if experiment_root else (
        root / "output" / "llm-experiments"
    )
    if not output_root.is_absolute():
        output_root = root / output_root

    provider = (
        provider_factory(record.provider)
        if provider_factory is not None
        else _provider_for_record(record)
    )
    model = replay.model or record.model
    prompt = replay.prompt if replay.prompt is not None else record.prompt_text
    system = replay.system if replay.system is not None else record.system_text
    temperature = (
        replay.temperature if replay.temperature is not None else record.temperature
    )

    kwargs: dict[str, object] = {}
    if model:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature

    started = time.monotonic()
    response = provider.generate_with_details(prompt, system=system, **kwargs)
    duration = round(time.monotonic() - started, 3)

    directory = output_root / f"{_timestamp_slug()}-{_safe_id(record.request_id)}"
    directory.mkdir(parents=True, exist_ok=False)
    baseline_path = directory / "baseline.json"
    variant_path = directory / "variant.json"
    comparison_path = directory / "comparison.md"

    baseline_path.write_text(
        json.dumps(_jsonable(record.raw), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    variant = {
        "schema_version": 1,
        "baseline_request_id": record.request_id,
        "replayed_at": datetime.now(tz=UTC).isoformat(),
        "duration_seconds": duration,
        "provider": response.provider,
        "model": response.model,
        "request": response.request or {
            "prompt": prompt,
            "system": system,
            "kwargs": kwargs,
            "prompt_chars": len(prompt),
            "system_chars": len(system) if system else 0,
        },
        "response": response.response or {"text": response.text},
        "usage": response.usage,
        "cost": response.cost,
    }
    variant_path.write_text(
        json.dumps(_jsonable(variant), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_path.write_text(
        _comparison_markdown(record, replay, response, duration),
        encoding="utf-8",
    )
    return LLMReplayArtifact(
        directory=directory,
        baseline_path=baseline_path,
        variant_path=variant_path,
        comparison_path=comparison_path,
        response=response,
    )


def _resolve_target(target: Path | str | None, project_root: Path) -> Path:
    path = Path(target) if target else project_root / "output"
    return path if path.is_absolute() else project_root / path


def _paths_from_manifest(manifest_path: Path) -> tuple[list[Path], list[LLMLogProblem]]:
    problems: list[LLMLogProblem] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [LLMLogProblem(manifest_path, f"Malformed manifest JSON: {exc}")]

    paths: list[Path] = []
    if not isinstance(manifest, dict):
        return [], [LLMLogProblem(manifest_path, "Manifest is not a JSON object")]

    results = manifest.get("results")
    if not isinstance(results, list):
        return [], [LLMLogProblem(manifest_path, "Manifest has no results array")]

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        raw_path = result.get("llm_log_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.exists():
            paths.append(path)
        else:
            problems.append(
                LLMLogProblem(
                    manifest_path,
                    f"Manifest result {index} points to missing log: {path}",
                )
            )
    return sorted(set(paths)), problems


def _read_jsonl_log(path: Path) -> tuple[list[LLMLogRecord], list[LLMLogProblem]]:
    records: list[LLMLogRecord] = []
    problems: list[LLMLogProblem] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [LLMLogProblem(path, f"Could not read log: {exc}")]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("Log line is not a JSON object")
            records.append(_normalize_record(raw, path, line_number))
        except (json.JSONDecodeError, ValueError) as exc:
            problems.append(LLMLogProblem(path, str(exc), line_number))
    return records, problems


def _normalize_record(
    raw: dict[str, object],
    path: Path,
    line_number: int,
) -> LLMLogRecord:
    request = raw.get("request")
    response = raw.get("response")
    usage = raw.get("usage")
    cost = raw.get("cost")
    context = raw.get("context")
    error = raw.get("error")

    return LLMLogRecord(
        raw=raw,
        source_path=path,
        line_number=line_number,
        request_id=str(raw.get("request_id") or f"{path.name}:{line_number}"),
        started_at=_string_or_none(raw.get("started_at")),
        finished_at=_string_or_none(raw.get("finished_at")),
        duration_seconds=_float_or_none(raw.get("duration_seconds")),
        provider=str(raw.get("provider") or "unknown"),
        model=_string_or_none(raw.get("model")),
        context=dict(context) if isinstance(context, dict) else {},
        request=dict(request) if isinstance(request, dict) else {},
        response=dict(response) if isinstance(response, dict) else None,
        usage=_numeric_dict(usage),
        cost=dict(cost) if isinstance(cost, dict) else {},
        error=dict(error) if isinstance(error, dict) else None,
    )


def _provider_for_record(record: LLMLogRecord) -> LLMProvider:
    if record.provider == "claude":
        config = ClaudeConfig()
        if record.model:
            config.model = record.model
        return ClaudeProvider(config)
    if record.provider == "ollama":
        config = OllamaConfig()
        if record.model:
            config.model = record.model
        return OllamaProvider(config)
    raise ValueError(f"Unsupported replay provider: {record.provider}")


def _comparison_markdown(
    record: LLMLogRecord,
    replay: LLMReplayRequest,
    response: LLMCallResponse,
    duration: float,
) -> str:
    baseline_cost = record.estimated_cost_usd
    variant_cost = _float_or_none(response.cost.get("estimated_cost_usd"))
    cost_delta = (
        round(variant_cost - baseline_cost, 8)
        if baseline_cost is not None and variant_cost is not None
        else None
    )
    duration_delta = (
        round(duration - record.duration_seconds, 3)
        if record.duration_seconds is not None
        else None
    )

    prompt = replay.prompt if replay.prompt is not None else record.prompt_text
    system = replay.system if replay.system is not None else record.system_text
    old_system = record.system_text or ""
    new_system = system or ""

    return "\n".join(
        [
            "# LLM Replay Comparison",
            "",
            "## Context",
            "",
            f"- Baseline request: `{record.request_id}`",
            f"- Source log: `{record.source_path}:{record.line_number}`",
            f"- Step: `{record.step or ''}`",
            f"- Provider: `{record.provider}`",
            f"- Baseline model: `{record.model or ''}`",
            (
                "- Variant model: "
                f"`{response.model or replay.model or record.model or ''}`"
            ),
            f"- Baseline duration seconds: `{record.duration_seconds}`",
            f"- Variant duration seconds: `{duration}`",
            f"- Duration delta seconds: `{duration_delta}`",
            f"- Baseline cost USD: `{baseline_cost}`",
            f"- Variant cost USD: `{variant_cost}`",
            f"- Cost delta USD: `{cost_delta}`",
            "",
            "## Prompt Diff",
            "",
            "```diff",
            *_diff_lines("baseline-user", "variant-user", record.prompt_text, prompt),
            "```",
            "",
            "## System Diff",
            "",
            "```diff",
            *_diff_lines("baseline-system", "variant-system", old_system, new_system),
            "```",
            "",
            "## Baseline Response",
            "",
            "```text",
            _excerpt(record.response_text),
            "```",
            "",
            "## Variant Response",
            "",
            "```text",
            _excerpt(response.text),
            "```",
            "",
        ]
    )


def _diff_lines(old_label: str, new_label: str, old: str, new: str) -> list[str]:
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=old_label,
            tofile=new_label,
            lineterm="",
        )
    )
    return lines or ["(no changes)"]


def _record_search_text(record: LLMLogRecord) -> str:
    pieces = [
        record.request_id,
        record.provider,
        record.model or "",
        record.step or "",
        record.difficulty or "",
        str(record.seed or ""),
        record.prompt_text,
        record.system_text or "",
        record.response_text,
        json.dumps(record.error, sort_keys=True) if record.error else "",
    ]
    return "\n".join(pieces).lower()


def _content_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else json.dumps(_jsonable(value))
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value)


def _numeric_dict(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int | float)
    }


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _timestamp_slug() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return safe[:80] or "request"


def _excerpt(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n... omitted {omitted} chars ..."


def _jsonable(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return str(value)
    return value
