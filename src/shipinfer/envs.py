"""Project-wide, environment-overridable knobs (vLLM/vio-ai-style lazy env access).

Read a knob as ``envs.NAME`` (e.g. ``envs.OMNIA_LLM_TEMPERATURE``). Each read pulls the
*current* environment value — so tests can ``monkeypatch.setenv`` at runtime — and falls back
to the default; a malformed value never raises, it just yields the default.

These are small runtime toggles/overrides, not the primary config: structured, user-facing
settings (providers, models, per-provider ``temperature``, secrets) live in the ``config/*.toml``
files behind :class:`~omnia.core.config.repository.ConfigRepository`. An env knob here is the
escape hatch — handy for tests (deterministic ``temperature=0``), CI, and power users — and,
where it overlaps a config field, the **env wins** so a one-off override needs no file edit.

Add a knob as a new entry in ``environment_variables`` (keyed by the env var name).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _bool(name: str, default: bool = False) -> bool:
    """Truthy env (``1/true/yes/on``) → True; unset → ``default`` (never raises)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    """``float(env)`` if parseable, else ``default`` (never raises)."""
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    """``int(env)`` if parseable, else ``default`` (never raises)."""
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


environment_variables: dict[str, Callable[[], Any]] = {
    # ── logging ── empty -> use the configured log_level (config/omnia.toml).
    "OMNIA_LOG_LEVEL": lambda: _str("OMNIA_LOG_LEVEL", ""),
    # ── per-LLM-call temperatures (OMNIA_{PLUGIN}_{FUNCTION}_TEMPERATURE) ──
    # Each distinct LLM call gets its own knob, defaulted to what that task wants and
    # env-overridable (e.g. set to 0 for deterministic runs). The general per-provider default
    # temperature (used by smart_notes FIELD GENERATION) lives in providers.toml, not here —
    # these override only the specific structured/authoring calls below.
    #   smart_notes · detect-language: classification → deterministic.
    "OMNIA_SMART_NOTES_DETECT_LANGUAGE_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_DETECT_LANGUAGE_TEMPERATURE", 0.0
    ),
    #   smart_notes · auto-prompt: infer each field's type+prompt from its name.
    "OMNIA_SMART_NOTES_AUTO_PROMPT_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_AUTO_PROMPT_TEMPERATURE", 0.4
    ),
    #   smart_notes · improve-prompt: polish one field's rough prompt.
    "OMNIA_SMART_NOTES_IMPROVE_PROMPT_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_IMPROVE_PROMPT_TEMPERATURE", 0.4
    ),
    #   smart_notes · improve-all: polish many fields' prompts at once.
    "OMNIA_SMART_NOTES_IMPROVE_ALL_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_IMPROVE_ALL_TEMPERATURE", 0.4
    ),
    #   smart_notes · classify-deps: label refs hard/soft → deterministic (B2: no flicker).
    "OMNIA_SMART_NOTES_CLASSIFY_DEPS_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_CLASSIFY_DEPS_TEMPERATURE", 0.0
    ),
    #   smart_notes · rewrite-edge: rewrite a prompt to reflect ONE graph edge change.
    "OMNIA_SMART_NOTES_REWRITE_EDGE_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_REWRITE_EDGE_TEMPERATURE", 0.3
    ),
    #   smart_notes · tool-author: compile a description into a user tool's Python → the
    #   user reads and runs the result, so favour the conventional answer over a creative one.
    "OMNIA_SMART_NOTES_TOOL_AUTHOR_TEMPERATURE": lambda: _float(
        "OMNIA_SMART_NOTES_TOOL_AUTHOR_TEMPERATURE", 0.1
    ),
    # ── smart_notes · K-note batching (LAYER 3) ── K, the notes one request may cover.
    #
    # ``-1`` (anything below 1) is OFF and means the feature does not exist:
    # ``SmartNotesSettings.notes_per_call()`` returns 1, ``batch_planner`` hands back
    # ``SOLO_PLANNER``, and every field takes the pre-LAYER-3 code path — no envelope, no item
    # ids, no batched parser. Any value >= 1 is K, and it is the CEILING the stored
    # ``batch_notes_per_call`` is clamped to, so this knob always has the last word.
    #
    # DEFAULT 10, from the OUTPUT BUDGET, which is the one argument for a K that does not depend
    # on a timing study: a chunk asks for K answers inside ONE completion, the measured deck's
    # binding field is "Synonyms (explained)" at ~677 output tokens at its longest, and
    # 8192/677 ~= 12. 10 stays under the cap even when every answer in the chunk is the longest
    # ever seen. FieldBudget shrinks K further per field from the lengths that field actually
    # produces (the K=20 runs sent 155 and 124 chunks where a flat 20 would have sent 50), but a
    # floor that holds without the adaptive layer is worth having.
    #
    # WHAT BATCHING BUYS IS REQUESTS. That is measured and it reproduces: at 8 workers over 100
    # real notes, K=10 sent 794.5 provider calls and K=20 sent 574.5, against 1300 ungrouped (-39%
    # and -56%); the second session's 20-note runs came out at -41% and -59%. The harness is
    # tests/benchmarks/smart_notes_live.py and the rows both sessions produced are committed
    # beside it in tests/benchmarks/data/ — see that directory's README for what each session
    # does and does not establish.
    #
    # THE LATENCY EFFECT IS UNPROVEN, AND TWO OPPOSITE CLAIMS HAVE BEEN MADE HERE ALREADY. The
    # first came from the fake rig, which charged a chunk per OUTPUT ITEM — so K answers in one
    # call cost exactly what K calls cost and grouping could only ever measure slower; that is an
    # artefact of the rig and "fewer requests, never fewer seconds" was never established. The
    # second came from one live session, which had K=20 at 1049.5 s and K=10 at 1162.5 s against
    # 1254.1 s ungrouped and was read as "faster everywhere". It does not reproduce: re-running
    # the same harness against the same collection gave 8x1 213.9 s (206.3-221.5), 8x20 215.2 s
    # (175.0-255.4, a tie) and 8x10 476.5 s (435.4-517.6, 2.2x SLOWER). Two sessions, opposite
    # answers, within-arm spread as wide as the between-arm gap. Treat K's effect on wall clock
    # as WITHIN NOISE: do not claim batching is faster, and do not claim it is slower.
    #
    # The worker count is the knob the same study DID settle, and it is a different knob — see
    # SmartNotesSettings.max_concurrent_generations.
    "OMNIA_SMART_NOTES_BATCHING": lambda: _int("OMNIA_SMART_NOTES_BATCHING", 10),
    # ── provider request bound ── how many provider HTTP requests may be in flight at once
    # while a generation fan-out is running (see core/network/limiter.py). 0 = size the bound
    # to the fan-out itself, which is the default and changes nothing. Set it BELOW the worker
    # count to express "run 8 fields at once but keep at most 3 requests in flight" — the case
    # a bound derived from the pool width cannot express, and the reason the limiter is a
    # separate mechanism from the pool rather than a restatement of it.
    #
    # STAYS 0 after the live benchmark, and that is a result rather than an omission. Across
    # twelve real runs at 4, 8 and 16 workers the provider returned ZERO 429s; the retry loop
    # fired FOUR times, once each in four different runs (4x1 rep1, 4x10 rep1, 8x20 rep2,
    # 16x10 rep1), every one of them a network error rather than a throttle. The limiter's own
    # peak was exactly the pool width in every arm and its total wait was 0.0 s, i.e. nothing
    # ever queued behind the bound. There is no measured load at which the limiter needs to bind
    # BELOW the pool, so shipping a number here would only be a slower default wearing a safety
    # label. It stays the escape hatch for a tighter account — which is exactly the account whose
    # 429s this run could not observe, because it ran against one Vertex project with a generous
    # quota.
    #
    # TWO LIMITS ON THAT ZERO, both stated because the number is otherwise read as a guarantee.
    # (a) It is one account's quota, not a property of the provider. (b) The 429 counter watches
    # the urllib client, and about a third of the run's provider calls do not go through it: the
    # edge_tts TTS path speaks a WebSocket (198-201 of every run's calls; 200 of the 8x20 arm's
    # 600). Throttling there arrives as a socket timeout, which no 429 classifier can recognise —
    # and the study's single provider error was on exactly that path. "Zero 429s" is established
    # for the HTTP providers and is simply not measured for edge_tts.
    "OMNIA_MAX_CONCURRENT_REQUESTS": lambda: _int("OMNIA_MAX_CONCURRENT_REQUESTS", 0),
    # ── HTTP ── default request timeout (seconds) for the stdlib HTTP client.
    "OMNIA_HTTP_TIMEOUT": lambda: _float("OMNIA_HTTP_TIMEOUT", 30.0),
    # ── storage dispatch (ADR-006) ── one knob per persistence concern, selecting its backend.
    # Default "database" = the Anki collection config (config, usage, and voices all in col
    # config, so they ride along with AnkiWeb sync); the file backends stay first-class and
    # selectable. These are read at startup by
    # the PersistenceDispatcher: changing a value triggers a ONE-TIME sync of that concern's
    # data from the previous backend to the newly-selected one on the next startup (the last-used
    # value is remembered in user_files/.storage.json), so switching never loses state.
    "OMNIA_CONFIG_STORAGE": lambda: _str(
        "OMNIA_CONFIG_STORAGE", "database"
    ),  # "database" | "toml"
    "OMNIA_USAGE_STORAGE": lambda: _str(
        "OMNIA_USAGE_STORAGE", "database"
    ),  # "database" | "json"
    "OMNIA_VOICE_CACHE_STORAGE": lambda: _str(
        "OMNIA_VOICE_CACHE_STORAGE", "database"
    ),  # "database" | "json"
}


def __getattr__(name: str) -> Any:
    # Lazy evaluation of environment variables (PEP 562) — read at access time, not import.
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)