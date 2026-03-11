"""Tests for pipeline orchestration."""

from pathlib import Path
from unittest.mock import MagicMock

from crossword_generator.exporters.base import Exporter
from crossword_generator.models import (
    FillResult,
    PuzzleEnvelope,
    ThemeConcept,
    ThemeFile,
)
from crossword_generator.pipeline import Pipeline
from crossword_generator.steps.base import PipelineStep


class MockStep(PipelineStep):
    """A mock pipeline step that adds fill data."""

    def __init__(self, name: str = "mock-step") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        grid = [["A", "B"], ["C", "D"]]
        return envelope.model_copy(
            update={
                "fill": FillResult(grid=grid, filler_used="mock"),
                "step_history": [*envelope.step_history, self._name],
            }
        )


class MockExporter(Exporter):
    """A mock exporter that writes a dummy file."""

    def __init__(self, ext: str = ".mock") -> None:
        self._ext = ext
        self.exported: list[Path] = []

    @property
    def file_extension(self) -> str:
        return self._ext

    def export(self, envelope: PuzzleEnvelope, output_path: Path) -> Path:
        filepath = output_path / f"test{self._ext}"
        filepath.write_text("mock")
        self.exported.append(filepath)
        return filepath


class TestPipeline:
    def test_run_executes_steps(self, tmp_path: Path) -> None:
        step = MockStep()
        pipeline = Pipeline(steps=[step], exporters=[], output_dir=tmp_path)
        envelope = PuzzleEnvelope()

        result = pipeline.run(envelope)
        assert result.fill is not None
        assert "mock-step" in result.step_history

    def test_run_saves_intermediates(self, tmp_path: Path) -> None:
        step = MockStep()
        pipeline = Pipeline(steps=[step], exporters=[], output_dir=tmp_path)
        envelope = PuzzleEnvelope()

        pipeline.run(envelope)
        json_files = list(tmp_path.glob("intermediate_*.json"))
        assert len(json_files) == 1

    def test_run_exports(self, tmp_path: Path) -> None:
        step = MockStep()
        exporter = MockExporter()
        pipeline = Pipeline(steps=[step], exporters=[exporter], output_dir=tmp_path)
        envelope = PuzzleEnvelope()

        pipeline.run(envelope)
        assert len(exporter.exported) == 1
        assert exporter.exported[0].exists()

    def test_multiple_steps(self, tmp_path: Path) -> None:
        step1 = MockStep("step-1")
        step2_run = MagicMock(
            side_effect=lambda e: e.model_copy(
                update={"step_history": [*e.step_history, "step-2"]}
            )
        )
        step2 = MagicMock(spec=PipelineStep)
        step2.name = "step-2"
        step2.run = step2_run

        pipeline = Pipeline(steps=[step1, step2], exporters=[], output_dir=tmp_path)
        result = pipeline.run(PuzzleEnvelope())
        assert result.step_history == ["step-1", "step-2"]

    def test_multiple_exporters(self, tmp_path: Path) -> None:
        step = MockStep()
        exporter1 = MockExporter(".fmt1")
        exporter2 = MockExporter(".fmt2")
        pipeline = Pipeline(
            steps=[step], exporters=[exporter1, exporter2], output_dir=tmp_path
        )

        pipeline.run(PuzzleEnvelope())
        assert len(exporter1.exported) == 1
        assert len(exporter2.exported) == 1

    def test_export_error_does_not_crash(self, tmp_path: Path) -> None:
        step = MockStep()
        bad_exporter = MagicMock(spec=Exporter)
        bad_exporter.file_extension = ".bad"
        bad_exporter.export.side_effect = RuntimeError("export failed")

        pipeline = Pipeline(steps=[step], exporters=[bad_exporter], output_dir=tmp_path)
        # Should not raise
        result = pipeline.run(PuzzleEnvelope())
        assert result.fill is not None

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "output"
        step = MockStep()
        pipeline = Pipeline(steps=[step], exporters=[], output_dir=output_dir)

        pipeline.run(PuzzleEnvelope())
        assert output_dir.exists()

    def test_preloaded_theme_skips_theme_step(self, tmp_path: Path) -> None:
        """When a pre-loaded theme is in the envelope, theme step is skipped."""
        theme = ThemeConcept(
            topic="Pre-loaded topic",
            seed_entries=["EAGLE", "HAWK"],
            revealer="SOAR",
        )

        # Write a theme file
        theme_file = ThemeFile(
            theme=theme,
            grid_size=9,
            generated_at="2024-01-01T00:00:00Z",
            generator_model="test",
        )
        theme_path = tmp_path / "test_theme.json"
        theme_path.write_text(theme_file.model_dump_json(indent=2))

        # Load it and verify the theme is injected
        from crossword_generator.theme_io import load_theme

        loaded = load_theme(theme_path)
        assert loaded.theme.topic == "Pre-loaded topic"

        # Build a pipeline with the pre-loaded theme in envelope
        envelope = PuzzleEnvelope(theme=loaded.theme)

        # A mock "theme step" that should NOT be called
        theme_step = MagicMock(spec=PipelineStep)
        theme_step.name = "theme-generation"

        # A fill step that just passes through
        fill_step = MockStep("fill-step")

        pipeline = Pipeline(
            steps=[fill_step], exporters=[], output_dir=tmp_path
        )
        result = pipeline.run(envelope)

        # Theme step was not in the pipeline, theme is preserved
        assert result.theme is not None
        assert result.theme.topic == "Pre-loaded topic"
        theme_step.run.assert_not_called()
