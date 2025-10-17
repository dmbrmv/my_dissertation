"""Logging utility.

Centralized, idempotent logging setup for the logging providing:

Primary entry points:
1. get_logger(): Base singleton project logger (no function context).
2. setup_logger(function_name, ...): Returns a LoggerAdapter injecting a logical
   function / task name (func_ctx) into each record for clearer traceability.

Features:
- Emoji + (optional ANSI color) enhanced console output for rapid triage.
- Size‑based rotating file handler (UTF‑8, safe for long runs / batch jobs).
- Safe argument handling (avoids '%'-format crashes when users pass comma args).
- Environment overrides for level and file path.
- Idempotent initialization (no duplicate handlers).
- Designed for library + CLI consistency (no propagation to root).

Environment variables (legacy + new):
- camels_ru_LOG_LEVEL, camels_ru_LOG_FILE, NO_COLOR (as before)
- NO_EMOJI: If set (any value), suppresses emoji while retaining spacing.

Notes:
- Auto‑detects Jupyter to enable color even when stdout is not a TTY.
- Multi‑line messages are indented for readability in notebooks.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Any

# --- Constants ---
_DEFAULT_LOGGER_NAME = "PhDLogger"
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILENAME = "PhDLogger.log"
_ROTATE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_ROTATE_BACKUP_COUNT = 5

# --- Mappings for Formatting ---
_LEVEL_EMOJIS: dict[int, str] = {
    logging.DEBUG: "🐞  ",
    logging.INFO: "ℹ️  ",
    logging.WARNING: "⚠️  ",
    logging.ERROR: "❌  ",
    logging.CRITICAL: "🚨  ",
}

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\x1b[38;5;244m",  # Grey
    logging.INFO: "\x1b[38;5;39m",  # Blue
    logging.WARNING: "\x1b[38;5;214m",  # Orange
    logging.ERROR: "\x1b[38;5;196m",  # Red
    logging.CRITICAL: "\x1b[48;5;196;38;5;231m",  # White on Red
}
_RESET_COLOR = "\x1b[0m"


def _in_jupyter() -> bool:
    """Return True if running inside a Jupyter / IPython kernel."""
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return bool(ip) and "IPKernelApp" in ip.config
    except Exception:
        return False


class EmojiFormatter(logging.Formatter):
    """A custom log formatter that injects emojis, function context, and optional color."""

    def __init__(
        self,
        *,
        use_color: bool = True,
        use_emoji: bool = True,
        indent_multiline: bool = True,
    ):
        """Initializes the formatter.

        Args:
            use_color: If True, applies ANSI color codes to the output.
            use_emoji: If True, includes emojis in the output.
            indent_multiline: If True, indents continuation lines for multiline messages.
        """
        # Include %(emoji)s placeholder so we don't mutate record.msg directly
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(func_ctx)s | %(emoji)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color
        self.use_emoji = use_emoji
        self.indent_multiline = indent_multiline

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, adding an emoji and optional color, with safe arg handling."""
        # Capture original (possibly multi-line) message early
        original_text = record.getMessage()
        lines = original_text.splitlines()

        # Ensure func_ctx exists, even if the adapter is not used
        if not hasattr(record, "func_ctx"):
            record.func_ctx = "-"  # type: ignore[attr-defined]

        # Gracefully handle incorrect usage like logger.info("Text:", value)
        if record.args:
            try:
                # Test formatting; if it fails we'll join args instead
                _ = record.msg % record.args
            except Exception:
                record.msg = " ".join(
                    [str(record.msg), *(str(a) for a in record.args)]
                )
                record.args = ()

        # Emoji (optionally suppressed)
        record.emoji = (
            _LEVEL_EMOJIS.get(record.levelno, "➡️  ") if self.use_emoji else ""
        )

        # Let base class build the formatted message
        formatted_message = super().format(record)

        # Indent multiline continuation lines for readability
        if self.indent_multiline and len(lines) > 1:
            first = lines[0]
            try:
                start_idx = formatted_message.index(first)
                indent = " " * start_idx
                continuation = "\n".join(indent + l for l in lines[1:])
                formatted_message = formatted_message.replace(
                    first, first + "\n" + continuation, 1
                )
            except ValueError:
                # Fallback: simple indent without alignment
                continuation = "\n".join("    " + l for l in lines[1:])
                formatted_message = formatted_message + "\n" + continuation

        # Apply color if enabled and a color is defined for the level
        if self.use_color and (color := _LEVEL_COLORS.get(record.levelno)):
            return f"{color}{formatted_message}{_RESET_COLOR}"

        return formatted_message


def _determine_log_level(explicit_level: int | str | None) -> int:
    """Determines the log level from explicit, environment, or default settings.

    Args:
        explicit_level: An explicit log level (e.g., logging.INFO or "INFO").

    Returns:
        The determined log level as an integer.
    """
    level_str = str(
        explicit_level or os.getenv("PhDLogger_LOG_LEVEL", "INFO")
    ).upper()
    return getattr(logging, level_str, logging.INFO)


def get_logger(
    name: str = _DEFAULT_LOGGER_NAME, *, level: int | str | None = None
) -> logging.Logger:
    """Return the singleton project logger, configuring it on first use.

    Args:
        name: The logical name for the logger. All code should generally use the
              default unless specific isolation is required.
        level: An optional override for the log level (e.g., "DEBUG" or logging.DEBUG).
               If provided, it updates the level even on subsequent calls.

    Returns:
        A configured instance of `logging.Logger`.
    """
    logger = logging.getLogger(name)
    log_level = _determine_log_level(level)

    # If already configured, just update the level if a new one was passed
    if getattr(logger, "_camels_ru_configured", False):
        if level is not None:
            logger.setLevel(log_level)
        return logger

    # --- First-time configuration ---
    logger.setLevel(log_level)
    logger.propagate = False  # Avoid duplicate output from the root logger

    # Improved environment detection (Jupyter support)
    interactive = sys.stdout.isatty() or _in_jupyter()
    use_color = interactive and os.getenv("NO_COLOR") is None
    use_emoji = os.getenv("NO_EMOJI") is None
    formatter = EmojiFormatter(
        use_color=use_color, use_emoji=use_emoji, indent_multiline=True
    )

    # Add a console handler if one doesn't already exist
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Mark as configured to prevent re-initialization
    logger._camels_ru_configured = True  # type: ignore[attr-defined]
    logger.debug(
        "Logger '%s' initialized at level %s (color=%s, emoji=%s, jupyter=%s).",
        name,
        logging.getLevelName(log_level),
        use_color,
        use_emoji,
        _in_jupyter(),
    )
    return logger


class _FunctionContextAdapter(logging.LoggerAdapter):
    """A LoggerAdapter that injects function context via the `func_ctx` attribute."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Processes the log message and kwargs to add function context.

        Args:
            msg: The original log message.
            kwargs: The keyword arguments passed to the logging call.

        Returns:
            A tuple containing the processed message and kwargs.
        """
        # Ensure 'extra' dict exists and set 'func_ctx' from the adapter's context
        if self.extra is not None:
            kwargs.setdefault("extra", {})["func_ctx"] = self.extra.get(
                "func_ctx", "-"
            )
        return msg, kwargs


def setup_logger(
    function_name: str,
    *,
    level: int | str | None = None,
    logger_name: str = _DEFAULT_LOGGER_NAME,
    log_file: str | Path | None = None,
    rotate: bool = True,
    max_bytes: int = _ROTATE_MAX_BYTES,
    backup_count: int = _ROTATE_BACKUP_COUNT,
) -> logging.LoggerAdapter:
    """Return a logger adapter bound to a specific function or logical unit.

    The adapter injects the provided ``function_name`` via the ``func_ctx`` field for
    every emitted record. When a rotating file handler is requested, any filesystem
    failures are caught and logged with full stack traces so that caller code can
    continue operating while still surfacing diagnostics.

    Args:
        function_name: A descriptive name of the current function or task.
        level: An optional log level override.
        logger_name: The base logger name, shared project-wide.
        log_file: An explicit path for the log file. Overrides environment variables
                  and defaults.
        rotate: If True, enables size-based log rotation.
        max_bytes: The maximum size in bytes to trigger a log rotation.
        backup_count: The number of backup log files to keep.

    Returns:
        A `logging.LoggerAdapter` that injects `func_ctx` into log records.

    Raises:
        Exception: Any unexpected error bubbled up from the stdlib ``logging``
            module during handler configuration. Filesystem-related issues are
            logged with stack traces and suppressed to keep the application running.
    """
    base_logger = get_logger(logger_name, level=level)

    # Determine the log file path with precedence: function arg > env var > default
    effective_log_file = (
        log_file
        or os.getenv("camels_ru_LOG_FILE")
        or Path(_DEFAULT_LOG_DIR) / _DEFAULT_LOG_FILENAME
    )

    # Ensure the log directory exists
    try:
        Path(effective_log_file).parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        base_logger.exception(
            "Failed to create log directory for %s", effective_log_file
        )
        rotate = False  # Disable rotation if directory creation fails

    # Attach a rotating file handler if rotation is enabled and not already present
    if rotate:
        abs_log_path = str(Path(effective_log_file).resolve())
        handler_exists = any(
            isinstance(h, RotatingFileHandler)
            and h.baseFilename == abs_log_path
            for h in base_logger.handlers
        )

        if not handler_exists:
            try:
                rfh = RotatingFileHandler(
                    abs_log_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                # File logs: no color / emoji for portability
                rfh.setFormatter(
                    EmojiFormatter(
                        use_color=False,
                        use_emoji=False,
                        indent_multiline=False,
                    )
                )
                base_logger.addHandler(rfh)
                base_logger.debug(
                    "Added rotating file handler for %s (max=%d bytes, backups=%d)",
                    abs_log_path,
                    max_bytes,
                    backup_count,
                )
            except OSError:
                base_logger.exception(
                    "Could not add rotating file handler for %s", abs_log_path
                )

    return _FunctionContextAdapter(base_logger, {"func_ctx": function_name})


__all__ = ["get_logger", "setup_logger"]
