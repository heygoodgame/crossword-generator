"""Tests for go-crossword Docker filler."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crossword_generator.config import GoCrosswordConfig
from crossword_generator.fillers.base import FillError, GridSpec
from crossword_generator.fillers.go_crossword import GoCrosswordFiller

COMPACT_OUTPUT = """\
Generating crossword...

A B C D E
F G H I J
K L M N O
P Q R S T
U V W X Y

Crossword generated successfully!
Seed: 42
"""

JSON_OUTPUT = json.dumps(
    {
        "rows": 5,
        "cols": 5,
        "seed": 42,
        "grid": [
            ["A", "B", "C", "D", "E"],
            ["F", "G", "H", "I", "J"],
            ["K", "L", "M", "N", "O"],
            ["P", "Q", "R", "S", "T"],
            ["U", "V", "W", "X", "Y"],
        ],
        "words_across": [
            {"word": "ABCDE", "row": 0, "col": 0},
            {"word": "FGHIJ", "row": 1, "col": 0},
            {"word": "KLMNO", "row": 2, "col": 0},
            {"word": "PQRST", "row": 3, "col": 0},
            {"word": "UVWXY", "row": 4, "col": 0},
        ],
        "words_down": [
            {"word": "AFKPU", "row": 0, "col": 0},
            {"word": "BGLQV", "row": 0, "col": 1},
            {"word": "CHMRW", "row": 0, "col": 2},
            {"word": "DINSX", "row": 0, "col": 3},
            {"word": "EJOTY", "row": 0, "col": 4},
        ],
    }
)


@pytest.fixture
def config() -> GoCrosswordConfig:
    return GoCrosswordConfig()


@pytest.fixture
def filler(config: GoCrosswordConfig) -> GoCrosswordFiller:
    return GoCrosswordFiller(config)


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec(rows=5, cols=5)


class TestBuildCommand:
    def test_basic_command(self, filler: GoCrosswordFiller, spec: GridSpec) -> None:
        cmd = filler._build_command(spec, seed=42)
        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "-rows" in cmd
        assert "5" in cmd
        assert "-cols" in cmd
        assert "-seed" in cmd
        assert "42" in cmd
        assert "-format" in cmd
        assert "json" in cmd
        assert "-compact" not in cmd

    def test_dictionary_flags(self, filler: GoCrosswordFiller, spec: GridSpec) -> None:
        cmd = filler._build_command(spec, seed=42)
        assert "-dictionary" in cmd
        assert "/data/dictionary.txt" in cmd
        assert "-min-score" in cmd
        assert "50" in cmd

    def test_dictionary_volume_mount(
        self, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        cmd = filler._build_command(spec, seed=42)
        # Find the -v flag and verify it mounts read-only
        idx = cmd.index("-v")
        mount = cmd[idx + 1]
        assert mount.endswith(":/data/dictionary.txt:ro")
        assert "HggCuratedCrosswordList.txt" in mount

    def test_volume_mount_before_image(
        self, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        cmd = filler._build_command(spec, seed=42)
        v_idx = cmd.index("-v")
        img_idx = cmd.index(filler._config.docker_image)
        assert v_idx < img_idx

    def test_no_dictionary_command(self, spec: GridSpec) -> None:
        config = GoCrosswordConfig(dictionary_path=None)
        filler = GoCrosswordFiller(config)
        cmd = filler._build_command(spec, seed=1)
        assert "-dictionary" not in cmd
        assert "-min-score" not in cmd
        assert "-v" not in cmd

    def test_compact_mode_command(self, spec: GridSpec) -> None:
        config = GoCrosswordConfig(output_format="compact")
        filler = GoCrosswordFiller(config)
        cmd = filler._build_command(spec, seed=1)
        assert "-compact" in cmd
        assert "-format" not in cmd

    def test_threads_flag(self, filler: GoCrosswordFiller, spec: GridSpec) -> None:
        cmd = filler._build_command(spec, seed=1)
        idx = cmd.index("-threads")
        assert cmd[idx + 1] == "100"

    def test_custom_image(self, spec: GridSpec) -> None:
        config = GoCrosswordConfig(docker_image="custom/image:v2")
        filler = GoCrosswordFiller(config)
        cmd = filler._build_command(spec, seed=1)
        assert "custom/image:v2" in cmd


class TestFill:
    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_successful_fill(
        self, mock_run: MagicMock, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        # Mock image inspect (already present)
        inspect_result = MagicMock(returncode=0)
        # Mock docker run
        run_result = MagicMock(returncode=0, stdout=JSON_OUTPUT, stderr="")
        mock_run.side_effect = [inspect_result, run_result]

        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 5
        assert result.grid[0][0] == "A"

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_successful_fill_compact(self, mock_run: MagicMock, spec: GridSpec) -> None:
        config = GoCrosswordConfig(output_format="compact")
        filler = GoCrosswordFiller(config)
        inspect_result = MagicMock(returncode=0)
        run_result = MagicMock(returncode=0, stdout=COMPACT_OUTPUT, stderr="")
        mock_run.side_effect = [inspect_result, run_result]

        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 5
        assert result.grid[0][0] == "A"

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_docker_failure(
        self, mock_run: MagicMock, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        inspect_result = MagicMock(returncode=0)
        run_result = MagicMock(returncode=1, stdout="", stderr="container error")
        mock_run.side_effect = [inspect_result, run_result]

        with pytest.raises(FillError, match="exited with code 1"):
            filler.fill(spec, seed=42)

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_timeout(
        self, mock_run: MagicMock, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        import subprocess

        inspect_result = MagicMock(returncode=0)
        mock_run.side_effect = [inspect_result, subprocess.TimeoutExpired("docker", 60)]

        with pytest.raises(FillError, match="timed out"):
            filler.fill(spec, seed=42)

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_docker_not_installed(
        self, mock_run: MagicMock, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        mock_run.side_effect = FileNotFoundError("docker not found")

        with pytest.raises(FillError):
            filler.fill(spec, seed=42)

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_seed_is_logged(
        self, mock_run: MagicMock, filler: GoCrosswordFiller, spec: GridSpec
    ) -> None:
        inspect_result = MagicMock(returncode=0)
        run_result = MagicMock(returncode=0, stdout=JSON_OUTPUT, stderr="")
        mock_run.side_effect = [inspect_result, run_result]

        filler.fill(spec, seed=12345)
        # Verify docker run was called with the seed
        docker_run_call = mock_run.call_args_list[1]
        cmd = docker_run_call[0][0]
        idx = cmd.index("-seed")
        assert cmd[idx + 1] == "12345"


class TestIsAvailable:
    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_available(self, mock_run: MagicMock, filler: GoCrosswordFiller) -> None:
        # Both docker info and docker image inspect succeed
        mock_run.return_value = MagicMock(returncode=0)
        assert filler.is_available() is True
        assert mock_run.call_count == 2

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_not_available_docker_not_running(
        self, mock_run: MagicMock, filler: GoCrosswordFiller
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert filler.is_available() is False
        # Should only call docker info, not image inspect
        assert mock_run.call_count == 1

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_not_available_image_missing(
        self, mock_run: MagicMock, filler: GoCrosswordFiller
    ) -> None:
        docker_info = MagicMock(returncode=0)
        image_inspect = MagicMock(returncode=1)
        mock_run.side_effect = [docker_info, image_inspect]
        assert filler.is_available() is False
        assert mock_run.call_count == 2

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_docker_missing(
        self, mock_run: MagicMock, filler: GoCrosswordFiller
    ) -> None:
        mock_run.side_effect = FileNotFoundError()
        assert filler.is_available() is False


class TestEnsureImage:
    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_image_already_exists(
        self, mock_run: MagicMock, filler: GoCrosswordFiller
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        filler._ensure_image()
        # Should only call docker image inspect, not pull
        assert mock_run.call_count == 1

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_image_pulled_when_missing(
        self, mock_run: MagicMock, filler: GoCrosswordFiller
    ) -> None:
        inspect_result = MagicMock(returncode=1)  # not found
        pull_result = MagicMock(returncode=0)
        mock_run.side_effect = [inspect_result, pull_result]

        filler._ensure_image()
        assert mock_run.call_count == 2

    @patch("crossword_generator.fillers.go_crossword.subprocess.run")
    def test_pull_failure(self, mock_run: MagicMock, filler: GoCrosswordFiller) -> None:
        inspect_result = MagicMock(returncode=1)
        pull_result = MagicMock(returncode=1, stderr="pull failed")
        mock_run.side_effect = [inspect_result, pull_result]

        with pytest.raises(FillError, match="Failed to pull"):
            filler._ensure_image()
