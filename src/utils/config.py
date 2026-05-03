from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

import yaml


class Config(dict):
    """Dict 包了一层支持 cfg.foo.bar.baz 访问"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return _wrap(value)


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: Union[str, Path]) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
