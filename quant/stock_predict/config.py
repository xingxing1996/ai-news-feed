"""全局配置加载。

- 读取 ``config/settings.yaml``，并用 ``config/settings.local.yaml``（若存在）覆盖。
- 把所有 ``paths.*`` 解析为相对项目根目录的绝对路径。
- 提供 ``get_settings()`` 单例与一个支持属性链访问的 ``AttrDict``。

设计原则：配置驱动一切，代码里不写死路径/参数。
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# 项目根目录：本文件位于 stock_predict/config.py，根是其上两级。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "config" / "settings.local.yaml"

_ENV_VAR = "STOCK_PREDICT_CONFIG"


class AttrDict(dict):
    """支持 ``cfg.paths.raw_dir`` 这种属性链访问的 dict。"""

    def __getattr__(self, item: str) -> Any:  # noqa: D401
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - 调试用
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base（override 优先）。"""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_paths(cfg: AttrDict, root: Path) -> AttrDict:
    """把 paths.* 下的相对路径解析为绝对路径，并补几个常用派生路径。"""
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    qlib_dir = Path(paths.get("qlib_dir", "data/qlib_data"))
    warehouse_dir = Path(paths.get("warehouse_dir", "data/warehouse"))
    sqlite_path = Path(paths.get("sqlite_path", "data/meta.db"))
    output_dir = Path(paths.get("output_dir", "data/output"))

    def _abs(p: Path) -> str:
        return str(p if p.is_absolute() else (root / p))

    cfg["paths"] = AttrDict(
        project_root=str(root),
        raw_dir=_abs(raw_dir),
        qlib_dir=_abs(qlib_dir),
        warehouse_dir=_abs(warehouse_dir),
        sqlite_path=_abs(sqlite_path),
        output_dir=_abs(output_dir),
    )
    for d in ("raw_dir", "qlib_dir", "warehouse_dir", "output_dir"):
        try:
            Path(cfg.paths[d]).mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
    try:
        Path(cfg.paths.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return cfg


def load_settings(config_path: str | os.PathLike | None = None) -> AttrDict:
    """加载并合并配置。优先级：环境变量 > 显式参数 > settings.yaml。"""
    path = Path(config_path) if config_path else None
    if path is None and os.getenv(_ENV_VAR):
        path = Path(os.environ[_ENV_VAR])
    base_path = path if path and path.exists() else DEFAULT_CONFIG

    with open(base_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # 本地覆盖（仅当用户用的是默认配置时才叠加 local）
    if not path and LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, local)

    cfg = _to_attrdict(cfg)
    # 解析项目根：默认相对 PROJECT_ROOT；若 settings 里 project_root 显式给了别的，则用它。
    root = Path(cfg.paths.project_root) if cfg.get("paths", {}).get("project_root") else PROJECT_ROOT
    root = root.resolve() if root.is_absolute() else (PROJECT_ROOT / root).resolve()
    cfg = _resolve_paths(cfg, root)
    cfg.paths.project_root = str(root)
    return cfg


_SETTINGS: AttrDict | None = None


def get_settings() -> AttrDict:
    """返回配置单例（首次调用时加载）。"""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def reset_settings() -> None:
    """测试用：清空单例缓存。"""
    global _SETTINGS
    _SETTINGS = None
