"""The on-disk model repository — Triton's layout, unchanged.

::

    model_repository/
      ship_detector/
        config.yaml
        1/model.plan
        2/model.plan
      person_embedder/
        config.yaml
        1/model.plan
      ship_pipeline/
        config.yaml            # platform: ensemble, no version dir needed

Using the same layout is not nostalgia: it means an existing Triton deployment's
repository can be pointed at this server (and back) without moving a single artefact, and
it means the version-selection semantics are ones people already know.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from shipinfer.core.errors import (
    ConfigurationError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
)
from shipinfer.core.logging import get_logger
from shipinfer.repository.model_config import ModelConfig, load_model_config

__all__ = ["CONFIG_FILENAMES", "ModelArtifact", "ModelEntry", "ModelRepository"]

_LOG = get_logger("repository")

#: Accepted config filenames, most-preferred first. ``config.pbtxt`` is *listed* so the
#: error message can say "found a Triton config, convert it" instead of "no config".
CONFIG_FILENAMES = ("config.yaml", "config.yml")
_VERSION_DIR = re.compile(r"^\d+$")


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One loadable version of one model: its config plus where its files live."""

    name: str
    version: int
    path: Path
    config: ModelConfig

    def file(self, filename: str) -> Path:
        """Resolve a file inside the version directory, checking it exists.

        Raises:
            ConfigurationError: naming both the missing file and the directory searched.
        """
        candidate = self.path / filename
        if not candidate.is_file():
            present = sorted(p.name for p in self.path.iterdir()) if self.path.is_dir() else []
            raise ConfigurationError(
                f"{self.name} v{self.version}: {filename!r} not found in {self.path} "
                f"(present: {present})"
            )
        return candidate

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Everything the repository knows about one model name."""

    name: str
    root: Path
    config: ModelConfig
    versions: tuple[int, ...]

    @property
    def latest(self) -> int:
        return self.versions[-1]


class ModelRepository:
    """A read-only index over a repository directory.

    Scanning is eager and done once, at construction: a filesystem walk per request would
    be an obvious waste, and a repository that changes under a running server is a
    deployment mistake, not a feature to support silently.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser()
        self._entries: dict[str, ModelEntry] = {}
        self._scan()

    # -- construction ------------------------------------------------------------------

    @classmethod
    def load(cls, root: Path | str) -> ModelRepository:
        return cls(Path(root))

    def _scan(self) -> None:
        if not self._root.is_dir():
            raise ConfigurationError(f"model repository {self._root} is not a directory")

        for model_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            if model_dir.name.startswith((".", "_")):
                continue
            config_path = self._find_config(model_dir)
            if config_path is None:
                _LOG.warning("skipping %s: no config.yaml", model_dir.name)
                continue

            config = load_model_config(config_path, name_hint=model_dir.name)
            if config.name != model_dir.name:
                raise ConfigurationError(
                    f"{config_path}: name {config.name!r} does not match directory "
                    f"{model_dir.name!r} — the directory is authoritative"
                )

            versions = self._discover_versions(model_dir, config)
            self._entries[config.name] = ModelEntry(
                name=config.name, root=model_dir, config=config, versions=versions
            )
            _LOG.info(
                "indexed model %s (platform=%s, versions=%s)",
                config.name,
                config.platform,
                list(versions),
            )

        if not self._entries:
            _LOG.warning("model repository %s contains no models", self._root)

    @staticmethod
    def _find_config(model_dir: Path) -> Path | None:
        for filename in CONFIG_FILENAMES:
            candidate = model_dir / filename
            if candidate.is_file():
                return candidate
        if (model_dir / "config.pbtxt").is_file():
            raise ConfigurationError(
                f"{model_dir}: found a Triton config.pbtxt but this server reads config.yaml — "
                "convert it with `shipinfer repo convert`"
            )
        return None

    @staticmethod
    def _discover_versions(model_dir: Path, config: ModelConfig) -> tuple[int, ...]:
        found = sorted(
            int(p.name)
            for p in model_dir.iterdir()
            if p.is_dir() and _VERSION_DIR.match(p.name)
        )
        if not found:
            # An ensemble has no artefacts of its own; giving it a synthetic version 1
            # keeps every downstream code path version-addressed with no special case.
            if config.is_ensemble:
                return (1,)
            raise ConfigurationError(
                f"{model_dir}: no numbered version directory (expected e.g. {model_dir/'1'})"
            )
        return tuple(config.version_policy.select(found))

    # -- queries -----------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[ModelEntry]:
        return iter(self._entries.values())

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, name: str) -> ModelEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise ModelNotFoundError(name, self.names()) from None

    def resolve(self, name: str, version: int | None = None) -> ModelArtifact:
        """Resolve ``name``/``version`` to a concrete artefact.

        Args:
            version: ``None`` selects the highest loaded version — the same default a
                Triton client gets when it omits the version.
        """
        entry = self.entry(name)
        chosen = entry.latest if version is None else version
        if chosen not in entry.versions:
            raise ModelVersionNotFoundError(name, chosen, list(entry.versions))
        # An ensemble's "version directory" is the model directory itself.
        path = entry.root if entry.config.is_ensemble else entry.root / str(chosen)
        return ModelArtifact(name=name, version=chosen, path=path, config=entry.config)

    def artifacts(self, names: list[str] | None = None) -> list[ModelArtifact]:
        """Every version the version policy selected, for every requested model."""
        selected = self.names() if names is None else names
        return [
            self.resolve(name, version)
            for name in selected
            for version in self.entry(name).versions
        ]
