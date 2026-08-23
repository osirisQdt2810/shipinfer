"""A generic, type-safe plugin registry — the extension mechanism used everywhere.

Every pluggable family in this project (placement policies, model backends, video
sources, allocators) is a folder of one-class-per-file modules plus a registry object.
Adding an implementation is then a new file and a decorator, never an edit to a switch
statement that already knows about five other things.

Two registration styles, because the families differ:

* **eager** — ``@registry.register("power_of_two")`` on the class. Right when importing the
  module is cheap (everything pure-Python).
* **lazy** — ``registry.register_lazy("tensorrt", "shipinfer.backends.tensorrt:TensorRTBackend")``.
  Right when importing the module drags in a 2 GB runtime. The class is imported on first
  :meth:`Registry.create`, so ``shipinfer repo ls`` does not need TensorRT installed to
  list a TensorRT model.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from shipinfer.core.errors import ConfigurationError

__all__ = ["Registry", "RegistryEntry"]

T = TypeVar("T")


class RegistryEntry(Generic[T]):
    """One registered implementation, resolved eagerly or on first use."""

    __slots__ = ("_cls", "_target", "aliases", "description", "name")

    def __init__(
        self,
        name: str,
        *,
        cls: type[T] | None = None,
        target: str | None = None,
        aliases: tuple[str, ...] = (),
        description: str = "",
    ) -> None:
        if (cls is None) == (target is None):
            raise ValueError("a registry entry needs exactly one of `cls` or `target`")
        self.name = name
        self.aliases = aliases
        self._cls = cls
        self._target = target
        self.description = description

    def resolve(self) -> type[T]:
        """Import the implementation if needed and return the class.

        Raises:
            ConfigurationError: if the target module or attribute cannot be imported. The
                message names the dotted path, because "ModuleNotFoundError: tensorrt" on
                its own never says *which* registration wanted it.
        """
        if self._cls is not None:
            return self._cls
        assert self._target is not None
        module_name, _, attribute = self._target.partition(":")
        if not attribute:
            raise ConfigurationError(f"lazy target {self._target!r} must be 'module:ClassName'")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigurationError(
                f"cannot load {self.name!r}: importing {module_name} failed ({exc}). "
                "Is the matching optional extra installed?"
            ) from exc
        try:
            self._cls = getattr(module, attribute)
        except AttributeError as exc:
            raise ConfigurationError(
                f"cannot load {self.name!r}: {module_name} has no {attribute!r}"
            ) from exc
        return self._cls

    @property
    def is_loaded(self) -> bool:
        return self._cls is not None

    def __repr__(self) -> str:
        where = "loaded" if self.is_loaded else f"lazy:{self._target}"
        return f"<RegistryEntry {self.name} ({where})>"


class Registry(Generic[T]):
    """Name -> implementation, with aliases, lazy targets and helpful errors.

    Args:
        kind: what is being registered, used verbatim in error messages
            (``"placement policy"``, ``"backend"``, ...).
        base: optional base class every registration must subclass. Checking at
            registration time turns a subtle duck-typing failure at request time into an
            import-time error.
    """

    def __init__(self, kind: str, base: type[T] | None = None) -> None:
        self.kind = kind
        self._base = base
        self._entries: dict[str, RegistryEntry[T]] = {}
        self._by_alias: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- registration -------------------------------------------------------------------

    def register(
        self, name: str, *aliases: str, description: str = ""
    ) -> Callable[[type[T]], type[T]]:
        """Class decorator: ``@registry.register("round_robin")``."""

        def decorator(cls: type[T]) -> type[T]:
            if self._base is not None and not issubclass(cls, self._base):
                raise TypeError(
                    f"{cls.__name__} cannot be registered as a {self.kind}: "
                    f"it does not subclass {self._base.__name__}"
                )
            self._add(
                RegistryEntry(
                    name,
                    cls=cls,
                    aliases=aliases,
                    description=description or (cls.__doc__ or "").strip().split("\n")[0],
                )
            )
            return cls

        return decorator

    def register_lazy(
        self, name: str, target: str, *aliases: str, description: str = ""
    ) -> None:
        """Register ``"module:ClassName"`` to be imported on first use."""
        self._add(RegistryEntry(name, target=target, aliases=aliases, description=description))

    def _add(self, entry: RegistryEntry[T]) -> None:
        with self._lock:
            existing = self._entries.get(entry.name)
            if existing is not None and existing is not entry:
                raise ConfigurationError(
                    f"{self.kind} {entry.name!r} is already registered "
                    f"({existing!r}); pick another name"
                )
            self._entries[entry.name] = entry
            for alias in entry.aliases:
                if alias in self._by_alias and self._by_alias[alias] != entry.name:
                    raise ConfigurationError(
                        f"{self.kind} alias {alias!r} already points at "
                        f"{self._by_alias[alias]!r}"
                    )
                self._by_alias[alias] = entry.name

    # -- lookup ------------------------------------------------------------------------

    def canonical(self, name: str) -> str:
        """Resolve an alias to the registered name."""
        return self._by_alias.get(name, name)

    def entry(self, name: str) -> RegistryEntry[T]:
        key = self.canonical(name)
        try:
            return self._entries[key]
        except KeyError:
            raise ConfigurationError(
                f"unknown {self.kind} {name!r}; available: {self.names()}"
            ) from None

    def get(self, name: str) -> type[T]:
        """The implementation class, importing it if the entry is lazy."""
        return self.entry(name).resolve()

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate the named implementation."""
        return self.get(name)(*args, **kwargs)  # type: ignore[call-arg]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> list[tuple[str, str]]:
        """``(name, one-line description)`` pairs — what ``shipinfer ... --list`` prints."""
        return [(name, self._entries[name].description) for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.canonical(name) in self._entries

    def __iter__(self) -> Iterator[RegistryEntry[T]]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<Registry {self.kind}: {self.names()}>"
