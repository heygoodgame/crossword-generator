"""Textual terminal UI for browsing LLM call logs."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from crossword_generator.llm.log_browser import (
    LLMLogFilters,
    LLMLogProblem,
    LLMLogRecord,
    LLMReplayRequest,
    filter_llm_log_records,
    replay_llm_record,
)


class LLMLogBrowserApp(App[None]):
    """Interactive browser for structured LLM logs."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #left {
        width: 45%;
        min-width: 64;
        border-right: solid $accent;
    }

    #right {
        width: 55%;
    }

    #filters {
        height: auto;
        padding: 0 1;
    }

    .filter-row {
        height: auto;
    }

    Input {
        margin: 0 1 0 0;
    }

    #records {
        height: 1fr;
    }

    #status {
        height: 3;
        padding: 0 1;
        border-top: solid $accent;
    }

    TabbedContent {
        height: 1fr;
    }

    TextArea {
        height: 1fr;
    }

    #experiment {
        height: 1fr;
    }

    #replayControls {
        height: auto;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "replay", "Replay"),
        ("/", "focus_search", "Search"),
        ("f5", "apply_filters", "Apply filters"),
    ]

    def __init__(
        self,
        records: list[LLMLogRecord],
        problems: list[LLMLogProblem],
        *,
        initial_filters: LLMLogFilters | None = None,
        experiment_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._all_records = records
        self._records = filter_llm_log_records(
            records,
            initial_filters or LLMLogFilters(),
        )
        self._problems = problems
        self._experiment_root = experiment_root
        self._selected_record: LLMLogRecord | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                with Vertical(id="filters"):
                    yield Label("Filters")
                    with Horizontal(classes="filter-row"):
                        yield Input(placeholder="Search", id="filterText")
                        yield Input(placeholder="Step", id="filterStep")
                    with Horizontal(classes="filter-row"):
                        yield Input(placeholder="Model", id="filterModel")
                        yield Input(placeholder="Difficulty", id="filterDifficulty")
                    with Horizontal(classes="filter-row"):
                        yield Input(placeholder="Seed", id="filterSeed")
                        yield Input(placeholder="Size", id="filterSize")
                    with Horizontal(classes="filter-row"):
                        yield Checkbox("Errors only", id="filterErrors")
                        yield Button("Apply", id="applyFilters")
                        yield Button("Clear", id="clearFilters")
                yield DataTable(id="records", cursor_type="row")
                yield Static(id="status")
            with Vertical(id="right"):
                with TabbedContent(id="tabs"):
                    with TabPane("Metadata", id="tabMeta"):
                        yield Static(id="metadata")
                    with TabPane("Prompt", id="tabPrompt"):
                        yield TextArea(read_only=True, id="prompt")
                    with TabPane("System", id="tabSystem"):
                        yield TextArea(read_only=True, id="systemPrompt")
                    with TabPane("Response", id="tabResponse"):
                        yield TextArea(read_only=True, id="response")
                    with TabPane("Usage", id="tabUsage"):
                        yield Static(id="usage")
                    with TabPane("Error", id="tabError"):
                        yield Static(id="error")
                    with TabPane("Experiment", id="tabExperiment"):
                        with Vertical(id="experiment"):
                            with Horizontal(id="replayControls"):
                                yield Input(
                                    placeholder="Model override",
                                    id="replayModel",
                                )
                                yield Input(
                                    placeholder="Temperature override",
                                    id="replayTemperature",
                                )
                                yield Button("Replay", id="replay")
                            yield Label("Prompt")
                            yield TextArea(id="replayPrompt")
                            yield Label("System")
                            yield TextArea(id="replaySystem")
                            yield Static(id="replayStatus")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#records", DataTable)
        table.add_columns(
            "Step",
            "Model",
            "Puzzle",
            "Seed",
            "Cost",
            "Duration",
            "Status",
        )
        self._refresh_table()
        if self._records:
            table.move_cursor(row=0)
            self._select_record(self._records[0])
        else:
            self._set_status("No records match the current filters.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "applyFilters":
            self.action_apply_filters()
        elif event.button.id == "clearFilters":
            self._clear_filters()
        elif event.button.id == "replay":
            self.action_replay()

    def on_data_table_cursor_moved(self, event: DataTable.CursorMoved) -> None:
        if event.cursor_row < 0 or event.cursor_row >= len(self._records):
            return
        self._select_record(self._records[event.cursor_row])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_index = self.query_one("#records", DataTable).get_row_index(
            event.row_key
        )
        if row_index < len(self._records):
            self._select_record(self._records[row_index])

    def action_focus_search(self) -> None:
        self.query_one("#filterText", Input).focus()

    def action_apply_filters(self) -> None:
        self._records = filter_llm_log_records(
            self._all_records,
            LLMLogFilters(
                step=_input_value(self, "#filterStep"),
                model=_input_value(self, "#filterModel"),
                difficulty=_input_value(self, "#filterDifficulty"),
                seed=_optional_int(_input_value(self, "#filterSeed")),
                size=_optional_int(_input_value(self, "#filterSize")),
                errors_only=self.query_one("#filterErrors", Checkbox).value,
                text=_input_value(self, "#filterText"),
            ),
        )
        self._refresh_table()
        if self._records:
            self.query_one("#records", DataTable).move_cursor(row=0)
            self._select_record(self._records[0])
        else:
            self._selected_record = None
            self._clear_details()
            self._set_status("No records match the current filters.")

    def action_replay(self) -> None:
        record = self._selected_record
        if record is None:
            self._set_replay_status("Select a record first.")
            return
        if record.provider not in {"claude", "ollama"}:
            self._set_replay_status(
                f"Replay is not supported for provider {record.provider!r}."
            )
            return

        temp_text = _input_value(self, "#replayTemperature")
        temperature = _optional_float(temp_text)
        if temp_text and temperature is None:
            self._set_replay_status("Temperature must be a number.")
            return

        self._set_replay_status("Running replay...")
        try:
            artifact = replay_llm_record(
                record,
                LLMReplayRequest(
                    model=_input_value(self, "#replayModel") or None,
                    temperature=temperature,
                    prompt=self.query_one("#replayPrompt", TextArea).text,
                    system=self.query_one("#replaySystem", TextArea).text or None,
                ),
                experiment_root=self._experiment_root,
            )
        except Exception as exc:  # pragma: no cover - interactive safety net
            self._set_replay_status(f"Replay failed: {exc}")
            return

        self._set_replay_status(
            f"Replay written: {artifact.comparison_path}"
        )

    def _refresh_table(self) -> None:
        table = self.query_one("#records", DataTable)
        table.clear()
        for index, record in enumerate(self._records):
            table.add_row(
                record.step or "",
                record.model or "",
                _puzzle_label(record),
                str(record.seed or ""),
                _cost_label(record),
                _duration_label(record),
                "error" if record.has_error else "ok",
                key=str(index),
            )
        self._set_status(
            f"{len(self._records)} records loaded"
            f" ({len(self._problems)} loader warnings)"
        )

    def _select_record(self, record: LLMLogRecord) -> None:
        self._selected_record = record
        self.query_one("#metadata", Static).update(_metadata_text(record))
        self.query_one("#prompt", TextArea).text = record.prompt_text
        self.query_one("#systemPrompt", TextArea).text = record.system_text or ""
        self.query_one("#response", TextArea).text = record.response_text
        self.query_one("#usage", Static).update(_json_text({
            "usage": record.usage,
            "cost": record.cost,
        }))
        self.query_one("#error", Static).update(
            _json_text(record.error) if record.error else "No error."
        )
        self.query_one("#replayModel", Input).value = record.model or ""
        self.query_one("#replayTemperature", Input).value = (
            str(record.temperature) if record.temperature is not None else ""
        )
        self.query_one("#replayPrompt", TextArea).text = record.prompt_text
        self.query_one("#replaySystem", TextArea).text = record.system_text or ""
        self._set_replay_status("")

    def _clear_details(self) -> None:
        self.query_one("#metadata", Static).update("")
        for selector in (
            "#prompt",
            "#systemPrompt",
            "#response",
            "#replayPrompt",
            "#replaySystem",
        ):
            self.query_one(selector, TextArea).text = ""
        self.query_one("#usage", Static).update("")
        self.query_one("#error", Static).update("")

    def _clear_filters(self) -> None:
        for selector in (
            "#filterText",
            "#filterStep",
            "#filterModel",
            "#filterDifficulty",
            "#filterSeed",
            "#filterSize",
        ):
            self.query_one(selector, Input).value = ""
        self.query_one("#filterErrors", Checkbox).value = False
        self.action_apply_filters()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _set_replay_status(self, text: str) -> None:
        self.query_one("#replayStatus", Static).update(text)


def run_llm_log_browser(
    records: list[LLMLogRecord],
    problems: list[LLMLogProblem],
    *,
    initial_filters: LLMLogFilters | None = None,
    experiment_root: Path | None = None,
) -> None:
    """Launch the interactive LLM log browser."""
    LLMLogBrowserApp(
        records,
        problems,
        initial_filters=initial_filters,
        experiment_root=experiment_root,
    ).run()


def _input_value(app: LLMLogBrowserApp, selector: str) -> str:
    return app.query_one(selector, Input).value.strip()


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _puzzle_label(record: LLMLogRecord) -> str:
    size = f"{record.size}x{record.size}" if record.size else ""
    return " ".join(part for part in [record.difficulty or "", size] if part)


def _cost_label(record: LLMLogRecord) -> str:
    cost = record.estimated_cost_usd
    return "" if cost is None else f"${cost:.5f}"


def _duration_label(record: LLMLogRecord) -> str:
    return "" if record.duration_seconds is None else f"{record.duration_seconds:.2f}s"


def _metadata_text(record: LLMLogRecord) -> str:
    lines = [
        f"Request ID: {record.request_id}",
        f"Source: {record.source_path}:{record.line_number}",
        f"Started: {record.started_at or ''}",
        f"Finished: {record.finished_at or ''}",
        f"Provider: {record.provider}",
        f"Model: {record.model or ''}",
        f"Step: {record.step or ''}",
        f"Difficulty: {record.difficulty or ''}",
        f"Size: {record.size or ''}",
        f"Seed: {record.seed or ''}",
        f"Duration: {record.duration_seconds or ''}",
        f"Status: {'error' if record.has_error else 'ok'}",
    ]
    return "\n".join(lines)


def _json_text(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
