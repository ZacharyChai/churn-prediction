"""Typed access to config.toml."""
from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"


@dataclass(frozen=True)
class Asset:
    power_mw: float
    energy_mwh: float
    round_trip_efficiency: float
    soc_min_frac: float
    soc_max_frac: float
    soc_start_frac: float
    cycles_per_day: float
    degradation_usd_per_mwh: float

    @property
    def eta(self) -> float:
        """One-way efficiency, applied on both charge and discharge."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_start(self) -> float:
        return self.soc_start_frac * self.energy_mwh

    @property
    def daily_discharge_cap_mwh(self) -> float:
        return self.cycles_per_day * self.energy_mwh


@dataclass(frozen=True)
class Scenario:
    name: str
    asset: Asset


@dataclass(frozen=True)
class Config:
    asset: Asset
    scenarios: list[Scenario]
    data: dict
    solver: dict
    study: dict
    root: Path

    def path(self, key: str) -> Path:
        return self.root / self.data[key]


def load_config(path: Path | str | None = None) -> Config:
    """Load config.toml, or the file named by $BESS_CONFIG. Relative data paths resolve beside it."""
    path = Path(path or os.environ.get("BESS_CONFIG") or DEFAULT_CONFIG)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    asset = Asset(**raw["asset"])
    scenarios = []
    for s in raw["scenario"]:
        overrides = {k: v for k, v in s.items() if k != "name"}
        scenarios.append(Scenario(s["name"], replace(asset, **overrides)))
    return Config(
        asset=asset,
        scenarios=scenarios,
        data=raw["data"],
        solver=raw["solver"],
        study=raw["study"],
        root=path.parent,
    )
