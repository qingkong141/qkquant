"""全局配置：读取 config/settings.yaml，暴露 pydantic 模型。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class FetcherSettings(BaseModel):
    adjust: str = "hfq"
    retry_times: int = 3
    retry_wait_seconds: int = 2
    request_sleep_ms: int = 150


class DataSettings(BaseModel):
    root: str = "./data"
    duckdb_path: str = "./data/daily.duckdb"
    fetcher: FetcherSettings = Field(default_factory=FetcherSettings)

    @property
    def duckdb_abs_path(self) -> Path:
        p = Path(self.duckdb_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def root_abs_path(self) -> Path:
        p = Path(self.root)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class BacktestSettings(BaseModel):
    initial_capital: float = 100_000.0
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_tax: float = 0.001
    slippage_pct: float = 0.002
    limit_up_pct: float = 0.10
    report_root: str = "./reports"

    @property
    def report_root_abs(self) -> Path:
        p = Path(self.report_root)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "./logs/qkquant.log"
    rotation: str = "10 MB"
    retention: str = "14 days"

    @property
    def file_abs(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Settings(BaseModel):
    data: DataSettings = Field(default_factory=DataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(path: str | Path | None = None) -> Settings:
    """从 YAML 文件加载配置；找不到则使用默认值。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Settings()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例配置；首次调用加载，后续复用。"""
    return load_settings()
