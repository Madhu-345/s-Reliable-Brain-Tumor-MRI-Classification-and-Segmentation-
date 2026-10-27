"""
Configuration utilities for the uncertainty-aware multi-task
brain tumor MRI classification and segmentation framework.

Provides:
    - YAML configuration loading
    - Nested configuration access
    - Command-line style overrides
    - Dictionary conversion
    - Configuration merging
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Lightweight wrapper around a nested configuration dictionary.

    Examples
    --------
    >>> cfg = Config.from_yaml("configs/multitask.yaml")
    >>> cfg["training"]["batch_size"]
    >>> cfg.get("training.lr", 1e-4)
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._data = deepcopy(data or {})

    # -----------------------------------------------------------------
    # Dictionary-style access
    # -----------------------------------------------------------------

    def __getitem__(
        self,
        key: str,
    ) -> Any:
        return self._data[key]

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self._data[key] = value

    def __contains__(
        self,
        key: str,
    ) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -----------------------------------------------------------------
    # Attribute-style access
    # -----------------------------------------------------------------

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        if name not in self._data:
            raise AttributeError(
                f"Configuration key '{name}' not found."
            )

        value = self._data[name]

        if isinstance(value, dict):
            return Config(value)

        return value

    # -----------------------------------------------------------------
    # Get nested values
    # -----------------------------------------------------------------

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a configuration value.

        Supports both:

            cfg.get("training")

        and nested paths:

            cfg.get("training.batch_size")
        """

        if "." not in key:
            return self._data.get(
                key,
                default,
            )

        current: Any = self._data

        for part in key.split("."):
            if not isinstance(current, dict):
                return default

            if part not in current:
                return default

            current = current[part]

        return current

    # -----------------------------------------------------------------
    # Set nested values
    # -----------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Set a configuration value using a dotted path.

        Example
        -------
        cfg.set("training.batch_size", 32)
        """

        parts = key.split(".")

        current = self._data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}

            if not isinstance(
                current[part],
                dict,
            ):
                raise TypeError(
                    f"Cannot create nested key '{key}': "
                    f"'{part}' is not a dictionary."
                )

            current = current[part]

        current[parts[-1]] = value

    # -----------------------------------------------------------------
    # Conversion
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deep copy of the configuration dictionary.
        """

        return deepcopy(self._data)

    # -----------------------------------------------------------------
    # YAML loading
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "Config":
        """
        Load configuration from a YAML file.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Configuration path is not a file: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ValueError(
                "The root element of a configuration "
                "file must be a YAML mapping."
            )

        return cls(data)

    # -----------------------------------------------------------------
    # YAML saving
    # -----------------------------------------------------------------

    def save_yaml(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save the current configuration to YAML.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                self._data,
                file,
                sort_keys=False,
                default_flow_style=False,
            )

        return path

    # -----------------------------------------------------------------
    # Merge configurations
    # -----------------------------------------------------------------

    def merge(
        self,
        other: dict[str, Any] | "Config",
    ) -> "Config":
        """
        Recursively merge another configuration.

        Values from `other` override existing values.
        """

        if isinstance(
            other,
            Config,
        ):
            other = other.to_dict()

        if not isinstance(
            other,
            dict,
        ):
            raise TypeError(
                "Configuration to merge must be "
                "a dictionary or Config object."
            )

        merged = self.to_dict()

        _deep_merge(
            merged,
            other,
        )

        return Config(merged)

    # -----------------------------------------------------------------
    # Command-line override
    # -----------------------------------------------------------------

    def override(
        self,
        key: str,
        value: str,
    ) -> None:
        """
        Override a configuration value using a string value.

        The value is automatically converted to an appropriate
        Python type where possible.

        Examples
        --------
        >>> cfg.override("training.batch_size", "32")
        >>> cfg.override("training.amp", "true")
        """

        self.set(
            key,
            _parse_value(value),
        )

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Config({self._data!r})"
        )


# ---------------------------------------------------------------------
# Recursive dictionary merge
# ---------------------------------------------------------------------


def _deep_merge(
    base: dict[str, Any],
    update: dict[str, Any],
) -> None:
    """
    Recursively merge `update` into `base`.
    """

    for key, value in update.items():

        if (
            key in base
            and isinstance(
                base[key],
                dict,
            )
            and isinstance(
                value,
                dict,
            )
        ):
            _deep_merge(
                base[key],
                value,
            )

        else:
            base[key] = deepcopy(value)


# ---------------------------------------------------------------------
# Automatic value parsing
# ---------------------------------------------------------------------


def _parse_value(
    value: str,
) -> Any:
    """
    Convert a command-line string into a Python value.
    """

    text = value.strip()

    # Boolean
    if text.lower() == "true":
        return True

    if text.lower() == "false":
        return False

    # Null
    if text.lower() in {
        "none",
        "null",
    }:
        return None

    # Integer
    try:
        return int(text)
    except ValueError:
        pass

    # Float
    try:
        return float(text)
    except ValueError:
        pass

    # YAML lists/dictionaries/strings
    try:
        parsed = yaml.safe_load(text)

        if parsed is not None:
            return parsed

    except yaml.YAMLError:
        pass

    return text


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------


def load_config(
    path: str | Path,
) -> Config:
    """
    Convenience wrapper for loading a YAML configuration.
    """

    return Config.from_yaml(path)


def save_config(
    config: Config | dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Convenience wrapper for saving a configuration.
    """

    if isinstance(
        config,
        Config,
    ):
        return config.save_yaml(path)

    return Config(config).save_yaml(path)


def merge_configs(
    base: Config | dict[str, Any],
    override: Config | dict[str, Any],
) -> Config:
    """
    Merge two configurations.

    The second configuration takes precedence.
    """

    if isinstance(
        base,
        Config,
    ):
        base_config = base
    else:
        base_config = Config(base)

    return base_config.merge(
        override
    )


__all__ = [
    "Config",
    "load_config",
    "save_config",
    "merge_configs",
]