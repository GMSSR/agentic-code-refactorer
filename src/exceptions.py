class StartupError(Exception):
    """Base exception for initialization errors."""

    pass


class ConfigError(StartupError):
    """Raised when configuration loading or validation fails."""

    pass


class CheckpointError(StartupError):
    """Raised when checkpoint loading or validation fails."""

    pass
