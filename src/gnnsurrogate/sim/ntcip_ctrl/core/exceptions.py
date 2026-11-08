"""Domain exceptions. Everything raised on purpose derives from :class:`ControllerError`."""

from __future__ import annotations


class ControllerError(Exception):
    """Base class with structured context for logging and API error bodies."""

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        return f"{self.message} ({', '.join(f'{k}={v!r}' for k, v in self.context.items())})"


class ConfigurationError(ControllerError):
    """Invalid configuration or timing (e.g. unequal barrier sums)."""


class UTDFParseError(ControllerError):
    """A UTDF file is malformed or lacks a required record."""


class SNMPError(ControllerError):
    """Malformed SNMP/BER message or protocol violation."""


class VissimAutomationError(ControllerError):
    """PTV Vissim COM automation failed or is unavailable."""
