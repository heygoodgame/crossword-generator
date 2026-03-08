"""go-crossword Docker wrapper filler."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from crossword_generator.config import GoCrosswordConfig, find_project_root
from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec
from crossword_generator.fillers.parser import (
    ParseError,
    parse_compact_output,
    parse_json_output,
)

logger = logging.getLogger(__name__)


class GoCrosswordFiller(GridFiller):
    """Grid filler using the go-crossword Docker image."""

    def __init__(
        self, config: GoCrosswordConfig, *, name: str | None = None
    ) -> None:
        self._config = config
        self._name_override = name

    @property
    def name(self) -> str:
        return self._name_override or "go-crossword"

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        """Fill a grid by invoking go-crossword via Docker.

        Args:
            spec: Grid dimensions.
            seed: Random seed for reproducibility. Generated if not provided.

        Returns:
            FilledGrid with the filled grid.

        Raises:
            FillError: If Docker invocation fails or output cannot be parsed.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        logger.info(
            "Filling %dx%d grid with go-crossword (seed=%d)", spec.rows, spec.cols, seed
        )

        self._ensure_image()
        cmd = self._build_command(spec, seed)
        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise FillError(
                f"go-crossword timed out after {self._config.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise FillError("Docker is not installed or not in PATH") from e

        if result.returncode != 0:
            raise FillError(
                f"go-crossword exited with code {result.returncode}:\n{result.stderr}"
            )

        try:
            if self._config.output_format == "json":
                filled = parse_json_output(result.stdout)
            else:
                filled = parse_compact_output(result.stdout)
        except ParseError as e:
            raise FillError(f"Failed to parse go-crossword output: {e}") from e

        logger.info(
            "Grid filled: %d across words, %d down words",
            len(filled.words_across),
            len(filled.words_down),
        )
        return filled

    def is_available(self) -> bool:
        """Check if Docker is running and the required image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

        # Check that the Docker image exists locally
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self._config.docker_image],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _build_command(self, spec: GridSpec, seed: int) -> list[str]:
        """Build the Docker run command."""
        cmd = [
            "docker",
            "run",
            "--rm",
        ]

        # Volume mount for external dictionary (must come before image name)
        if self._config.dictionary_path is not None:
            dict_path = Path(self._config.dictionary_path)
            if not dict_path.is_absolute():
                dict_path = find_project_root() / dict_path
            cmd.extend(["-v", f"{dict_path}:/data/dictionary.txt:ro"])

        cmd.extend([
            self._config.docker_image,
            "-rows",
            str(spec.rows),
            "-cols",
            str(spec.cols),
            "-seed",
            str(seed),
            "-threads",
            str(self._config.threads),
        ])
        if self._config.output_format == "json":
            cmd.extend(["-format", "json"])
        else:
            cmd.append("-compact")

        # External dictionary flags
        if self._config.dictionary_path is not None:
            cmd.extend([
                "-dictionary",
                "/data/dictionary.txt",
                "-min-score",
                str(self._config.min_dictionary_score),
            ])
        # Future go-crossword fork flags:
        # -grid-template <path>: accept pre-built grid
        return cmd

    def _ensure_image(self) -> None:
        """Pull the Docker image if it's not already available locally."""
        image = self._config.docker_image
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise FillError("Docker is not installed or not responding")

        logger.info("Pulling Docker image %s...", image)
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise FillError(f"Failed to pull image {image}:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            raise FillError(f"Timed out pulling image {image}")
