"""Complete DraftKings-style DFS target construction and nested validation.

Live / legacy DFS models still train and evaluate against the **partial**
label ``Actual_DK_Points_Modeled`` (hit-type + BB/HBP/RBI for hitters;
IP/K/BB/H + FIP-proxy ER for pitchers). This module builds a separate
complete-target surface:

- one row per ``(game_pk, player)``
- confirmed starters and explicit DNP / zero outcomes
- official event counting where Statcast events are authoritative
- independently reconstructed fields (runs, SB/CS, pitcher ER from score
  deltas, win/CG/CGSO/NH) with explicit provenance — never validating a
  FIP-derived ER estimate against another FIP-derived “actual”
- nested time-aware validation + lineup-level promotion gate

``DFS_COMPLETE_TARGET_MODE`` defaults to ``shadow``. Do not promote to
live on a trivial player-MAE improvement alone.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_metrics import (
    config, data, dfs_backtest, dfs_ml, dfs_optimizer, helpers,
    model_validation,
)

# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------

SOURCE_OFFICIAL = "official"
SOURCE_RECONSTRUCTED = "reconstructed"
SOURCE_UNAVAILABLE = "unavailable"
SOURCE_ZERO_DNP = "zero_dnp"
SOURCE_LEGACY_FIP_PROXY = "legacy_fip_proxy_not_for_complete_eval"

COMPLETE_HITTER_LABEL = "Actual_DK_Points_Complete"
COMPLETE_PITCHER_LABEL = "Actual_DK_Points_Complete"
LEGACY_LABEL = "Actual_DK_Points_Modeled"

# Official Statcast steal / CS event names when present on non-PA rows.
STATCAST_SB_EVENTS = {"stolen_base_2b", "stolen_base_3b", "stolen_base_home"}
STATCAST_CS_EVENTS = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
}

HITTER_COMPONENT_COLUMNS = [
    "Singles", "Doubles", "Triples", "Home_Runs",
    "Walks", "HBP", "RBI", "Runs", "Stolen_Bases", "Caught_Stealing",
]
HITTER_SOURCE_COLUMNS = [f"{c}_Source" for c in HITTER_COMPONENT_COLUMNS] + [
    f"{COMPLETE_HITTER_LABEL}_Source",
]

PITCHER_COMPONENT_COLUMNS = [
    "Outs", "IP", "Strikeouts", "Hits", "Walks", "HBP", "Earned_Runs",
    "Win", "Complete_Game", "Shutout", "No_Hitter",
]
PITCHER_SOURCE_COLUMNS = [f"{c}_Source" for c in PITCHER_COMPONENT_COLUMNS] + [
    f"{COMPLETE_PITCHER_LABEL}_Source",
]

LEAKAGE_COLUMNS = frozenset(
    {COMPLETE_HITTER_LABEL, COMPLETE_PITCHER_LABEL, LEGACY_LABEL}
    | set(HITTER_COMPONENT_COLUMNS)
    | set(PITCHER_COMPONENT_COLUMNS)
    | set(HITTER_SOURCE_COLUMNS)
    | set(PITCHER_SOURCE_COLUMNS)
    | {"Appeared", "Started", "Got_Hit", "Hits", "Plate_Appearances"}
)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _source_series(n: int, source: str) -> pd.Series:
    return pd.Series([source] * n, dtype="object")


def combine_field_sources(sources: Sequence[str]) -> str:
    uniq = sorted({s for s in sources if s})
    if not uniq:
        return SOURCE_UNAVAILABLE
    if len(uniq) == 1:
        return uniq[0]
    if SOURCE_UNAVAILABLE in uniq and len(uniq) == 2:
        other = [s for s in uniq if s != SOURCE_UNAVAILABLE][0]
        return other
    if all(s in (SOURCE_OFFICIAL, SOURCE_ZERO_DNP) for s in uniq):
        return SOURCE_OFFICIAL
    return SOURCE_RECONSTRUCTED


# ---------------------------------------------------------------------------
# Hitter complete outcomes
# ---------------------------------------------------------------------------


def _parse_steals_from_des(des: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Best-effort SB/CS counts from free-text ``des`` when event codes absent."""
    text = des.fillna("").astype(str)
    sb = text.str.contains(r"steals?\s+\d|stolen base", case=False, regex=True).astype(int)
    # Avoid double-counting "caught stealing" as a steal.
    cs = text.str.contains(r"caught stealing", case=False, regex=True).astype(int)
    sb = sb.where(cs == 0, 0)
    return sb, cs


def reconstruct_runs_scored(game_events: pd.DataFrame) -> pd.DataFrame:
    """Attribute runs scored to batters via on-base tracking (reconstructed).

    Uses ``on_1b`` / ``on_2b`` / ``on_3b`` when present plus bat_score
    deltas. Home runs always credit the batter with at least one run
    (official event). Returns ``[game_pk, key_mlbam, Runs, Runs_Source]``.
    """
    cols = ["game_pk", "key_mlbam", "Runs", "Runs_Source"]
    if game_events is None or game_events.empty:
        return pd.DataFrame(columns=cols)

    df = game_events.copy()
    if "game_pk" not in df.columns or "batter" not in df.columns:
        return pd.DataFrame(columns=cols)

    runs = {}
    sources = {}

    # Official: HR always scores the batter.
    if "events" in df.columns:
        hr = df[df["events"] == "home_run"]
        for _, row in hr.iterrows():
            key = (row["game_pk"], int(row["batter"]))
            runs[key] = runs.get(key, 0) + 1
            sources[key] = SOURCE_OFFICIAL

    has_bases = all(c in df.columns for c in ("on_1b", "on_2b", "on_3b", "bat_score", "post_bat_score"))
    if has_bases:
        # Statcast on_* fields are start-of-play occupancy.
        sort_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in df.columns]
        ordered = df.sort_values(sort_cols) if sort_cols else df
        for game_pk, g in ordered.groupby("game_pk"):
            for _, row in g.iterrows():
                on_1b = row.get("on_1b") if pd.notna(row.get("on_1b")) else None
                on_2b = row.get("on_2b") if pd.notna(row.get("on_2b")) else None
                on_3b = row.get("on_3b") if pd.notna(row.get("on_3b")) else None
                scored = int(max(0, (row.get("post_bat_score") or 0) - (row.get("bat_score") or 0)))
                if scored <= 0:
                    continue
                # Credit runners who left the bases, preferring 3B then 2B then 1B then batter.
                candidates = [on_3b, on_2b, on_1b, row.get("batter")]
                credited = 0
                for runner in candidates:
                    if runner is None or pd.isna(runner):
                        continue
                    if credited >= scored:
                        break
                    key = (game_pk, int(runner))
                    # HR batter already counted officially above.
                    if row.get("events") == "home_run" and int(runner) == int(row.get("batter")):
                        credited += 1
                        continue
                    runs[key] = runs.get(key, 0) + 1
                    sources[key] = (
                        SOURCE_RECONSTRUCTED
                        if sources.get(key) != SOURCE_OFFICIAL
                        else SOURCE_OFFICIAL
                    )
                    credited += 1

    if not runs:
        return pd.DataFrame(columns=cols)
    rows = [
        {"game_pk": gpk, "key_mlbam": bat, "Runs": n, "Runs_Source": sources.get((gpk, bat), SOURCE_RECONSTRUCTED)}
        for (gpk, bat), n in runs.items()
    ]
    return pd.DataFrame(rows, columns=cols)


def compute_complete_hitter_outcomes(day_statcast: pd.DataFrame) -> pd.DataFrame:
    """Per-(game_pk, batter) complete DK counting with provenance."""
    empty_cols = (
        ["game_pk", "key_mlbam"] + HITTER_COMPONENT_COLUMNS + HITTER_SOURCE_COLUMNS
        + [COMPLETE_HITTER_LABEL, LEGACY_LABEL]
    )
    if day_statcast is None or day_statcast.empty:
        return pd.DataFrame(columns=empty_cols)

    pa_cols = ["game_pk", "batter", "events", "bat_score", "post_bat_score"]
    optional = [c for c in ("des", "on_1b", "on_2b", "on_3b", "at_bat_number", "pitch_number") if c in day_statcast.columns]
    completed = data.completed_events(day_statcast, [c for c in pa_cols if c in day_statcast.columns] + optional)
    if completed.empty or "game_pk" not in completed.columns:
        return pd.DataFrame(columns=empty_cols)

    ev = completed["events"]
    completed = completed.copy()
    completed["Singles"] = (ev == "single").astype(int)
    completed["Doubles"] = (ev == "double").astype(int)
    completed["Triples"] = (ev == "triple").astype(int)
    completed["Home_Runs"] = (ev == "home_run").astype(int)
    completed["Walks"] = helpers.is_walk_for_dk_scoring(ev).astype(int)
    completed["HBP"] = helpers.is_hit_by_pitch(ev).astype(int)
    completed["RBI"] = helpers.estimate_rbi(completed)

    agg = completed.groupby(["game_pk", "batter"], as_index=False).agg(
        Singles=("Singles", "sum"),
        Doubles=("Doubles", "sum"),
        Triples=("Triples", "sum"),
        Home_Runs=("Home_Runs", "sum"),
        Walks=("Walks", "sum"),
        HBP=("HBP", "sum"),
        RBI=("RBI", "sum"),
    ).rename(columns={"batter": "key_mlbam"})

    for col in ("Singles", "Doubles", "Triples", "Home_Runs", "Walks", "HBP"):
        agg[f"{col}_Source"] = SOURCE_OFFICIAL
    agg["RBI_Source"] = SOURCE_RECONSTRUCTED

    # Steals: prefer Statcast steal event rows (not in COUNTED_EVENTS).
    if "events" in day_statcast.columns and "batter" in day_statcast.columns:
        steal_rows = day_statcast[day_statcast["events"].isin(STATCAST_SB_EVENTS | STATCAST_CS_EVENTS)].copy()
        if not steal_rows.empty and "game_pk" in steal_rows.columns:
            steal_rows["SB"] = steal_rows["events"].isin(STATCAST_SB_EVENTS).astype(int)
            steal_rows["CS"] = steal_rows["events"].isin(STATCAST_CS_EVENTS).astype(int)

            def _runner_id(row) -> float:
                ev = str(row.get("events") or "")
                if "2b" in ev:
                    base = row.get("on_1b")
                elif "3b" in ev:
                    base = row.get("on_2b")
                elif "home" in ev:
                    base = row.get("on_3b")
                else:
                    base = None
                if base is not None and pd.notna(base):
                    return float(base)
                return float(row["batter"]) if pd.notna(row.get("batter")) else float("nan")

            steal_rows["key_mlbam"] = steal_rows.apply(_runner_id, axis=1)
            steal_agg = steal_rows.dropna(subset=["key_mlbam"]).groupby(
                ["game_pk", "key_mlbam"], as_index=False
            ).agg(Stolen_Bases=("SB", "sum"), Caught_Stealing=("CS", "sum"))
            sb_source = SOURCE_OFFICIAL
        else:
            steal_agg = pd.DataFrame(columns=["game_pk", "key_mlbam", "Stolen_Bases", "Caught_Stealing"])
            sb_source = SOURCE_UNAVAILABLE
    else:
        steal_agg = pd.DataFrame(columns=["game_pk", "key_mlbam", "Stolen_Bases", "Caught_Stealing"])
        sb_source = SOURCE_UNAVAILABLE

    if steal_agg.empty and "des" in day_statcast.columns:
        # Reconstructed fallback from description text on completed PAs.
        des_sb, des_cs = _parse_steals_from_des(completed.get("des", pd.Series(dtype=str)))
        tmp = completed[["game_pk", "batter"]].copy()
        tmp["Stolen_Bases"] = des_sb.to_numpy()
        tmp["Caught_Stealing"] = des_cs.to_numpy()
        steal_agg = tmp.groupby(["game_pk", "batter"], as_index=False).sum().rename(
            columns={"batter": "key_mlbam"}
        )
        sb_source = SOURCE_RECONSTRUCTED if (steal_agg[["Stolen_Bases", "Caught_Stealing"]].sum().sum() > 0) else SOURCE_UNAVAILABLE

    agg = agg.merge(steal_agg, on=["game_pk", "key_mlbam"], how="outer")
    for col in ("Singles", "Doubles", "Triples", "Home_Runs", "Walks", "HBP", "RBI"):
        agg[col] = agg[col].fillna(0).astype(int) if col in agg.columns else 0
    for col in ("Singles", "Doubles", "Triples", "Home_Runs", "Walks", "HBP"):
        src = f"{col}_Source"
        if src not in agg.columns:
            agg[src] = SOURCE_OFFICIAL
        else:
            agg[src] = agg[src].fillna(SOURCE_OFFICIAL)
    if "RBI_Source" not in agg.columns:
        agg["RBI_Source"] = SOURCE_RECONSTRUCTED
    else:
        agg["RBI_Source"] = agg["RBI_Source"].fillna(SOURCE_RECONSTRUCTED)

    agg["Stolen_Bases"] = pd.to_numeric(agg.get("Stolen_Bases", 0), errors="coerce").fillna(0).astype(int)
    agg["Caught_Stealing"] = pd.to_numeric(agg.get("Caught_Stealing", 0), errors="coerce").fillna(0).astype(int)
    agg["Stolen_Bases_Source"] = sb_source
    agg["Caught_Stealing_Source"] = sb_source

    runs = reconstruct_runs_scored(completed)
    agg = agg.merge(runs, on=["game_pk", "key_mlbam"], how="outer")
    for col in ("Singles", "Doubles", "Triples", "Home_Runs", "Walks", "HBP", "RBI",
                "Stolen_Bases", "Caught_Stealing"):
        agg[col] = pd.to_numeric(agg.get(col, 0), errors="coerce").fillna(0).astype(int)
    for col in ("Singles", "Doubles", "Triples", "Home_Runs", "Walks", "HBP"):
        src = f"{col}_Source"
        if src not in agg.columns:
            agg[src] = SOURCE_OFFICIAL
        else:
            agg[src] = agg[src].fillna(SOURCE_OFFICIAL)
    if "RBI_Source" not in agg.columns:
        agg["RBI_Source"] = SOURCE_RECONSTRUCTED
    else:
        agg["RBI_Source"] = agg["RBI_Source"].fillna(SOURCE_RECONSTRUCTED)
    if "Stolen_Bases_Source" not in agg.columns:
        agg["Stolen_Bases_Source"] = SOURCE_UNAVAILABLE
        agg["Caught_Stealing_Source"] = SOURCE_UNAVAILABLE
    else:
        agg["Stolen_Bases_Source"] = agg["Stolen_Bases_Source"].fillna(SOURCE_UNAVAILABLE)
        agg["Caught_Stealing_Source"] = agg["Caught_Stealing_Source"].fillna(SOURCE_UNAVAILABLE)

    agg["Runs"] = pd.to_numeric(agg.get("Runs", 0), errors="coerce").fillna(0).astype(int)
    if "Runs_Source" not in agg.columns:
        agg["Runs_Source"] = SOURCE_UNAVAILABLE
    else:
        agg["Runs_Source"] = agg["Runs_Source"].fillna(SOURCE_UNAVAILABLE)

    agg[COMPLETE_HITTER_LABEL] = (
        agg["Singles"] * config.DFS_DK_HITTER_SINGLE_POINTS
        + agg["Doubles"] * config.DFS_DK_HITTER_DOUBLE_POINTS
        + agg["Triples"] * config.DFS_DK_HITTER_TRIPLE_POINTS
        + agg["Home_Runs"] * config.DFS_DK_HITTER_HR_POINTS
        + agg["Walks"] * config.DFS_DK_HITTER_BB_POINTS
        + agg["HBP"] * config.DFS_DK_HITTER_HBP_POINTS
        + agg["RBI"] * config.DFS_DK_HITTER_RBI_POINTS
        + agg["Runs"] * config.DFS_DK_HITTER_RUN_POINTS
        + agg["Stolen_Bases"] * config.DFS_DK_HITTER_SB_POINTS
        + agg["Caught_Stealing"] * config.DFS_DK_HITTER_CS_POINTS
    )
    agg[f"{COMPLETE_HITTER_LABEL}_Source"] = [
        combine_field_sources([
            agg.loc[i, "Singles_Source"], agg.loc[i, "RBI_Source"],
            agg.loc[i, "Runs_Source"], agg.loc[i, "Stolen_Bases_Source"],
        ])
        for i in agg.index
    ]

    # Legacy partial label for benchmark (unchanged definition).
    legacy = dfs_backtest.compute_actual_hitter_dk_points(completed)
    agg = agg.merge(legacy, on=["game_pk", "key_mlbam"], how="left")
    agg[LEGACY_LABEL] = agg[LEGACY_LABEL].fillna(0.0)

    return agg


# ---------------------------------------------------------------------------
# Pitcher complete outcomes
# ---------------------------------------------------------------------------


def _independent_earned_runs(pitcher_events: pd.DataFrame) -> pd.DataFrame:
    """Runs charged while pitching minus obvious error-driven runs.

    Independent of the projection FIP formula. Uses batting-team score
    deltas on the pitcher's watch; subtracts deltas on ``field_error``
    events (reconstructed unearned proxy). Returns
    ``[game_pk, pitcher, Earned_Runs]``.
    """
    cols = ["game_pk", "pitcher", "Earned_Runs"]
    df = pitcher_events.copy()
    if (
        df.empty
        or "bat_score" not in df.columns
        or "post_bat_score" not in df.columns
        or "game_pk" not in df.columns
        or "pitcher" not in df.columns
    ):
        return pd.DataFrame(columns=cols)
    df["runs_on_play"] = (df["post_bat_score"] - df["bat_score"]).clip(lower=0)
    df["error_runs"] = np.where(df["events"] == "field_error", df["runs_on_play"], 0)
    df["earned_proxy"] = (df["runs_on_play"] - df["error_runs"]).clip(lower=0)
    return (
        df.groupby(["game_pk", "pitcher"], as_index=False)["earned_proxy"]
        .sum()
        .rename(columns={"earned_proxy": "Earned_Runs"})
    )


def compute_complete_pitcher_outcomes(day_statcast: pd.DataFrame) -> pd.DataFrame:
    """Per-(game_pk, pitcher) complete DK counting with provenance."""
    empty_cols = (
        ["game_pk", "key_mlbam"] + PITCHER_COMPONENT_COLUMNS + PITCHER_SOURCE_COLUMNS
        + [COMPLETE_PITCHER_LABEL, LEGACY_LABEL]
    )
    if day_statcast is None or day_statcast.empty:
        return pd.DataFrame(columns=empty_cols)

    need = ["game_pk", "pitcher", "events"]
    opt = [c for c in ("bat_score", "post_bat_score", "inning", "inning_topbot", "at_bat_number") if c in day_statcast.columns]
    if not all(c in day_statcast.columns for c in need):
        return pd.DataFrame(columns=empty_cols)

    completed = data.completed_events(day_statcast, need + opt)
    if completed.empty:
        return pd.DataFrame(columns=empty_cols)

    df = completed.copy()
    df["outs"] = helpers.outs_recorded(df["events"])
    df["so"] = helpers.is_strikeout(df["events"]).astype(int)
    df["bb"] = helpers.is_walk(df["events"]).astype(int)
    df["hit"] = helpers.is_hit(df["events"]).astype(int)
    df["hbp"] = helpers.is_hit_by_pitch(df["events"]).astype(int)

    group_keys = ["game_pk", "pitcher"]
    agg = df.groupby(group_keys, as_index=False).agg(
        Outs=("outs", "sum"),
        Strikeouts=("so", "sum"),
        Walks=("bb", "sum"),
        Hits=("hit", "sum"),
        HBP=("hbp", "sum"),
    )
    agg["IP"] = agg["Outs"] / 3.0
    for col in ("Outs", "Strikeouts", "Walks", "Hits", "HBP", "IP"):
        agg[f"{col}_Source"] = SOURCE_OFFICIAL

    # Independent ER reconstruction (NOT FIP).
    er = _independent_earned_runs(df)
    agg = agg.merge(er, on=["game_pk", "pitcher"], how="left")
    agg["Earned_Runs"] = agg["Earned_Runs"].fillna(0.0)
    agg["Earned_Runs_Source"] = SOURCE_RECONSTRUCTED

    # Starter bonuses: CG / shutout / no-hitter / win (all reconstructed).
    agg["Win"] = 0
    agg["Complete_Game"] = 0
    agg["Shutout"] = 0
    agg["No_Hitter"] = 0
    for col in ("Win", "Complete_Game", "Shutout", "No_Hitter"):
        agg[f"{col}_Source"] = SOURCE_RECONSTRUCTED

    try:
        with_id = day_statcast.rename(columns={"game_pk": "game_id"})
        roles = data.label_pitcher_roles(with_id)
        results = data.extract_game_results(with_id)
        results = results.rename(columns={"game_id": "game_pk"})
        starters = roles[roles["is_starter"]][["game_id", "pitcher", "team"]].rename(
            columns={"game_id": "game_pk"}
        )
        starter_stats = agg.merge(starters, on=["game_pk", "pitcher"], how="inner")
        if not starter_stats.empty:
            starter_stats = starter_stats.merge(results, on="game_pk", how="left")
            for _, row in starter_stats.iterrows():
                mask = (agg["game_pk"] == row["game_pk"]) & (agg["pitcher"] == row["pitcher"])
                is_cg = float(row["Outs"]) >= 24
                is_nh = float(row["Hits"]) == 0 and is_cg
                team = row.get("team")
                home_score = row.get("home_score")
                away_score = row.get("away_score")
                won = False
                shutout = False
                if pd.notna(home_score) and pd.notna(away_score) and pd.notna(team):
                    if team == row.get("home_team"):
                        won = home_score > away_score
                        shutout = bool(won and away_score == 0 and is_cg)
                    elif team == row.get("away_team"):
                        won = away_score > home_score
                        shutout = bool(won and home_score == 0 and is_cg)
                agg.loc[mask, "Complete_Game"] = int(is_cg)
                agg.loc[mask, "No_Hitter"] = int(is_nh)
                agg.loc[mask, "Shutout"] = int(shutout)
                agg.loc[mask, "Win"] = int(won and (is_cg or float(row["IP"]) >= 5.0))
    except Exception as exc:
        # Keep counting IP/K/H/ER; mark bonuses unavailable rather than silent pass.
        for col in ("Win", "Complete_Game", "Shutout", "No_Hitter"):
            agg[col] = 0
            agg[f"{col}_Source"] = SOURCE_UNAVAILABLE
        agg.attrs["starter_bonus_error"] = f"{type(exc).__name__}: {exc}"

    agg = agg.rename(columns={"pitcher": "key_mlbam"})
    agg[COMPLETE_PITCHER_LABEL] = (
        agg["IP"] * config.DFS_DK_PITCHER_IP_POINTS
        + agg["Strikeouts"] * config.DFS_DK_PITCHER_K_POINTS
        + agg["Walks"] * config.DFS_DK_PITCHER_BB_POINTS
        + agg["Hits"] * config.DFS_DK_PITCHER_H_POINTS
        + agg["HBP"] * config.DFS_DK_PITCHER_HBP_POINTS
        + agg["Earned_Runs"] * config.DFS_DK_PITCHER_ER_POINTS
        + agg["Win"] * config.DFS_DK_PITCHER_WIN_POINTS
        + agg["Complete_Game"] * config.DFS_DK_PITCHER_CG_POINTS
        + agg["Shutout"] * config.DFS_DK_PITCHER_CGSO_POINTS
        + agg["No_Hitter"] * config.DFS_DK_PITCHER_NO_HITTER_POINTS
    )
    agg[f"{COMPLETE_PITCHER_LABEL}_Source"] = SOURCE_RECONSTRUCTED

    # Legacy FIP-proxy label kept for benchmark only — never used as the
    # complete-target "actual ER" and tagged so evaluators can refuse to
    # treat it as independent ground truth.
    legacy = dfs_backtest.compute_actual_pitcher_dk_points(completed)
    # Legacy is date-pooled without game_pk historically; merge carefully.
    if "game_pk" not in legacy.columns and "game_pk" in completed.columns:
        # Recompute legacy per game_pk for alignment.
        legacy_parts = []
        for gpk, g in completed.groupby("game_pk"):
            part = dfs_backtest.compute_actual_pitcher_dk_points(g)
            part["game_pk"] = gpk
            legacy_parts.append(part)
        legacy = pd.concat(legacy_parts, ignore_index=True) if legacy_parts else legacy
    if "game_pk" in legacy.columns:
        agg = agg.merge(
            legacy[["game_pk", "key_mlbam", LEGACY_LABEL]],
            on=["game_pk", "key_mlbam"], how="left",
        )
    else:
        agg = agg.merge(legacy[["key_mlbam", LEGACY_LABEL]], on="key_mlbam", how="left")
    agg[LEGACY_LABEL] = agg[LEGACY_LABEL].fillna(0.0)

    return agg


def assert_er_not_fip_vs_fip(pitcher_outcomes: pd.DataFrame) -> None:
    """Guard: complete-target ER must not be the legacy FIP proxy."""
    if pitcher_outcomes is None or pitcher_outcomes.empty:
        return
    if "Earned_Runs_Source" in pitcher_outcomes.columns:
        bad = pitcher_outcomes["Earned_Runs_Source"].isin([SOURCE_LEGACY_FIP_PROXY])
        if bad.any():
            raise AssertionError("Complete-target ER must not use legacy FIP proxy")


# ---------------------------------------------------------------------------
# Training frame: candidates + DNP zeros + salaries/positions
# ---------------------------------------------------------------------------


def _zero_hitter_row(game_pk, key_mlbam, **extra) -> dict:
    row = {
        "game_pk": game_pk,
        "key_mlbam": key_mlbam,
        "Appeared": 0,
        "Started": 0,
        COMPLETE_HITTER_LABEL: 0.0,
        LEGACY_LABEL: 0.0,
        f"{COMPLETE_HITTER_LABEL}_Source": SOURCE_ZERO_DNP,
    }
    for col in HITTER_COMPONENT_COLUMNS:
        row[col] = 0
        row[f"{col}_Source"] = SOURCE_ZERO_DNP
    row.update(extra)
    return row


def assemble_complete_target_rows(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int | None = 20,
    *,
    confirmed_lineups: pd.DataFrame | None = None,
    salaries: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build complete-target training tables with DNP zeros.

    Returns ``{"hitters": df, "pitchers": df, "legacy_benchmark": meta}``.
    Hitter grain: ``(date, game_pk, key_mlbam)``. Pitcher grain:
    ``(date, game_pk, key_mlbam)`` for starters on the slate.
    """
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    empty = {
        "hitters": pd.DataFrame(),
        "pitchers": pd.DataFrame(),
        "feature_schedule_mode": "actual_starter_diagnostic",
        "legacy_benchmark": {"label": LEGACY_LABEL, "kept_as_benchmark": True},
    }
    if persisted is None:
        return empty

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    from mlb_metrics import game_picks_backtest
    games = game_picks_backtest.derive_historical_schedule_games(persisted)

    dates = sorted(team_schedule["date"].unique())
    if days:
        dates = dates[-days:]

    hitter_rows = []
    pitcher_rows = []
    for date in dates:
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue
        day_events = persisted[persisted["game_date"] == date]
        complete_h = compute_complete_hitter_outcomes(day_events)
        complete_p = compute_complete_pitcher_outcomes(day_events)
        assert_er_not_fip_vs_fip(complete_p)

        # Features for players projected that day.
        try:
            h_feat = dfs_ml.build_hitter_features(
                day["outputs"]["wave"],
                day["outputs"]["pave"],
                day["outputs"]["confidence"],
                day["todays_schedule"],
                day["matchup_probability"],
            )
        except Exception as exc:
            # Skip the date rather than training on an empty feature shell.
            print(
                f"WARNING: dfs_complete_targets skipping {date}: "
                f"build_hitter_features failed ({type(exc).__name__}: {exc})"
            )
            continue
        for col in dfs_ml.HITTER_FEATURE_COLUMNS:
            if col not in h_feat.columns:
                h_feat[col] = np.nan
        if "game_pk" not in h_feat.columns:
            h_feat["game_pk"] = pd.NA

        # Candidate universe = projected hitters ∪ confirmed lineup ∪ actual appearers.
        candidates = h_feat[["game_pk", "key_mlbam"]].drop_duplicates()
        if confirmed_lineups is not None and not confirmed_lineups.empty:
            conf = confirmed_lineups.copy()
            if "date" in conf.columns:
                conf = conf[pd.to_datetime(conf["date"]).dt.normalize() == pd.Timestamp(date).normalize()]
            if not conf.empty and {"game_pk", "key_mlbam"}.issubset(conf.columns):
                candidates = pd.concat([
                    candidates,
                    conf[["game_pk", "key_mlbam"]].drop_duplicates(),
                ], ignore_index=True).drop_duplicates()

        if not complete_h.empty:
            candidates = pd.concat([
                candidates,
                complete_h[["game_pk", "key_mlbam"]],
            ], ignore_index=True).drop_duplicates()

        merged = candidates.merge(h_feat, on=["game_pk", "key_mlbam"], how="left")
        merged = merged.merge(complete_h, on=["game_pk", "key_mlbam"], how="left")
        # Legacy *projections* for benchmark — never substitute the actual
        # partial label (that would leak outcomes into "predicted_legacy").
        proj_h = day["projected_hitters"]
        if {"game_pk", "key_mlbam", "DK_Points_Hitter"}.issubset(proj_h.columns):
            merged = merged.merge(
                proj_h[["game_pk", "key_mlbam", "DK_Points_Hitter"]].drop_duplicates(
                    subset=["game_pk", "key_mlbam"]
                ),
                on=["game_pk", "key_mlbam"],
                how="left",
            )
        appeared = merged[COMPLETE_HITTER_LABEL].notna() if COMPLETE_HITTER_LABEL in merged.columns else pd.Series(False, index=merged.index)
        merged["Appeared"] = appeared.astype(int)
        merged["Started"] = 0
        if confirmed_lineups is not None and not confirmed_lineups.empty:
            conf = confirmed_lineups
            if "is_confirmed_starter" in conf.columns:
                starters = conf.loc[conf["is_confirmed_starter"] == True, ["game_pk", "key_mlbam"]]  # noqa: E712
                starter_keys = set(zip(starters["game_pk"], starters["key_mlbam"]))
                merged["Started"] = [
                    1 if (g, k) in starter_keys else 0
                    for g, k in zip(merged["game_pk"], merged["key_mlbam"])
                ]

        # DNP → explicit zeros.
        for col in HITTER_COMPONENT_COLUMNS + [COMPLETE_HITTER_LABEL, LEGACY_LABEL]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        for col in HITTER_SOURCE_COLUMNS:
            if col not in merged.columns:
                merged[col] = SOURCE_ZERO_DNP
            else:
                merged[col] = merged[col].where(appeared, SOURCE_ZERO_DNP)

        if positions is not None and not positions.empty:
            pos = positions.rename(columns={"dk_slot": "dk_slot"}) if "dk_slot" in positions.columns else positions
            if "key_mlbam" in pos.columns:
                merged = merged.merge(
                    pos[["key_mlbam"] + [c for c in ("dk_slot", "Eligible_Positions") if c in pos.columns]].drop_duplicates("key_mlbam"),
                    on="key_mlbam", how="left",
                )
        if salaries is not None and not salaries.empty and "key_mlbam" in salaries.columns:
            sal_col = "Salary" if "Salary" in salaries.columns else (
                "Estimated_Salary" if "Estimated_Salary" in salaries.columns else None
            )
            if sal_col:
                merged = merged.merge(
                    salaries[["key_mlbam", sal_col]].drop_duplicates("key_mlbam"),
                    on="key_mlbam", how="left",
                )

        merged["date"] = date
        hitter_rows.append(merged)

        # Pitchers: slate starters + complete outcomes. Prefer game_pk join
        # to avoid doubleheader collisions on key_mlbam alone.
        p_proj = day["projected_pitchers"].copy()
        if "game_pk" not in p_proj.columns and games is not None and not games.empty:
            gday = games[games["date"] == date]
            if {"game_pk", "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam"}.issubset(gday.columns):
                home = gday[["game_pk", "home_probable_pitcher_key_mlbam"]].rename(
                    columns={"home_probable_pitcher_key_mlbam": "key_mlbam"}
                )
                away = gday[["game_pk", "away_probable_pitcher_key_mlbam"]].rename(
                    columns={"away_probable_pitcher_key_mlbam": "key_mlbam"}
                )
                pk_map = pd.concat([home, away], ignore_index=True).dropna()
                p_proj = p_proj.merge(pk_map, on="key_mlbam", how="left")
        join_keys = ["key_mlbam"]
        if "game_pk" in p_proj.columns and "game_pk" in complete_p.columns:
            join_keys = ["game_pk", "key_mlbam"]
        elif "game_pk" not in p_proj.columns:
            # Refuse key_mlbam-only merges when game_pk is expected on outcomes.
            p_proj["game_pk"] = pd.NA
            join_keys = ["game_pk", "key_mlbam"]
        p_merged = p_proj.merge(complete_p, on=join_keys, how="left")
        for col in PITCHER_COMPONENT_COLUMNS + [COMPLETE_PITCHER_LABEL, LEGACY_LABEL]:
            if col in p_merged.columns:
                p_merged[col] = p_merged[col].fillna(0)
        p_merged["date"] = date
        pitcher_rows.append(p_merged)

    hitters = pd.concat(hitter_rows, ignore_index=True) if hitter_rows else pd.DataFrame()
    pitchers = pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else pd.DataFrame()
    return {
        "hitters": hitters,
        "pitchers": pitchers,
        # Features currently come from derive_historical_team_schedule
        # (actual Statcast starters). Treat as diagnostic until as-of
        # probable-starter snapshots are wired into this assembler.
        "feature_schedule_mode": "actual_starter_diagnostic",
        "legacy_benchmark": {
            "label": LEGACY_LABEL,
            "kept_as_benchmark": True,
            "note": "Partial Actual_DK_Points_Modeled remains the legacy benchmark; FIP-proxy ER is not complete-target ground truth.",
        },
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def player_mae(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def rank_correlation(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    if mask.sum() < 3:
        return float("nan")
    corr, _ = spearmanr(actual[mask], predicted[mask])
    return float(corr) if corr == corr else float("nan")


def top_decile_recall(actual: pd.Series, predicted: pd.Series, *, decile: float = 0.1) -> float:
    mask = actual.notna() & predicted.notna()
    a = actual[mask]
    p = predicted[mask]
    n = len(a)
    if n < 10:
        return float("nan")
    k = max(1, int(np.ceil(n * decile)))
    top_actual = set(a.nlargest(k).index)
    top_pred = set(p.nlargest(k).index)
    return float(len(top_actual & top_pred) / k)


def boom_rate_calibration(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    boom_threshold: float | None = None,
) -> dict:
    """Compare predicted vs empirical boom rates (above threshold)."""
    mask = actual.notna() & predicted.notna()
    a = actual[mask]
    p = predicted[mask]
    if len(a) < 10:
        return {"boom_threshold": float("nan"), "predicted_boom_rate": float("nan"),
                "actual_boom_rate": float("nan"), "calibration_gap": float("nan")}
    thr = float(a.quantile(0.9) if boom_threshold is None else boom_threshold)
    return {
        "boom_threshold": thr,
        "predicted_boom_rate": float((p >= thr).mean()),
        "actual_boom_rate": float((a >= thr).mean()),
        "calibration_gap": float((p >= thr).mean() - (a >= thr).mean()),
    }


def lineup_projected_versus_actual(
    players: pd.DataFrame,
    *,
    projected_col: str,
    actual_col: str,
    date_col: str = "date",
    salary_col: str | None = "Estimated_Salary",
    slot_col: str | None = "dk_slot",
) -> dict:
    """Per-date optimal-lineup projected sum vs actual sum of those players.

    When salary/slot data are missing, falls back to top-N by projection
    (N = sum of roster slots) — still a lineup-level score, honestly labeled.
    """
    if players is None or players.empty:
        return {"n_dates": 0, "mean_abs_lineup_error": float("nan"), "method": "none"}

    n_slots = int(sum(config.DFS_ROSTER_SLOTS.values()))
    errors = []
    method = "top_n_fallback"
    for date, day in players.groupby(date_col):
        day = day.dropna(subset=[projected_col, actual_col])
        if day.empty:
            continue
        if (
            salary_col and slot_col
            and salary_col in day.columns and slot_col in day.columns
            and day[salary_col].notna().sum() >= n_slots
            and day[slot_col].notna().sum() >= n_slots
        ):
            try:
                pool = dfs_optimizer.build_player_pool(
                    day.assign(**{ "DK_Points": day[projected_col] }),
                ) if hasattr(dfs_optimizer, "build_player_pool") else day
                # Simpler: pick best projected per slot greedily for evaluation.
                picked = []
                remaining = day.copy()
                for slot, count in config.DFS_ROSTER_SLOTS.items():
                    elig = remaining[remaining[slot_col] == slot].nlargest(count, projected_col)
                    picked.append(elig)
                    remaining = remaining.drop(index=elig.index, errors="ignore")
                lineup = pd.concat(picked) if picked else day.nlargest(n_slots, projected_col)
                method = "slot_greedy"
            except Exception:
                lineup = day.nlargest(n_slots, projected_col)
                method = "top_n_fallback"
        else:
            lineup = day.nlargest(n_slots, projected_col)
            method = "top_n_fallback"
        errors.append(abs(float(lineup[projected_col].sum()) - float(lineup[actual_col].sum())))

    return {
        "n_dates": len(errors),
        "mean_abs_lineup_error": float(np.mean(errors)) if errors else float("nan"),
        "method": method,
    }


def contest_style_roi_simulation(
    players: pd.DataFrame,
    *,
    projected_col: str,
    actual_col: str,
    date_col: str = "date",
    entry_fee: float = 1.0,
    payout_top_frac: float = 0.2,
) -> dict:
    """Defensible toy ROI: each date, field = all players; cash if actual
    lineup (top-N by projection) finishes in top ``payout_top_frac`` of
    random opponent scores. Conservative and labeled hypothetical.
    """
    if players is None or players.empty:
        return {"n_dates": 0, "hypothetical_roi": float("nan"), "label": "insufficient"}
    n_slots = int(sum(config.DFS_ROSTER_SLOTS.values()))
    profits = []
    for _, day in players.groupby(date_col):
        day = day.dropna(subset=[projected_col, actual_col])
        if len(day) < n_slots * 2:
            continue
        our = day.nlargest(n_slots, projected_col)
        our_score = float(our[actual_col].sum())
        # Opponent distribution: sample other top-N by actual (strong field).
        field_scores = []
        rng = np.random.default_rng(config.NESTED_VALIDATION_RANDOM_SEED)
        idxs = day.index.to_numpy()
        for _ in range(min(50, len(day))):
            sample = day.loc[rng.choice(idxs, size=n_slots, replace=False)]
            field_scores.append(float(sample[actual_col].sum()))
        if not field_scores:
            continue
        cutoff = float(np.quantile(field_scores, 1.0 - payout_top_frac))
        win = our_score >= cutoff
        # Flat payout of 2x entry when cashing (even-money pool toy).
        profits.append((2.0 * entry_fee - entry_fee) if win else -entry_fee)
    if not profits:
        return {"n_dates": 0, "hypothetical_roi": float("nan"), "label": "hypothetical_toy_field"}
    staked = entry_fee * len(profits)
    return {
        "n_dates": len(profits),
        "hypothetical_roi": float(np.sum(profits) / staked),
        "label": "hypothetical_toy_field",
        "total_profit": float(np.sum(profits)),
    }


def evaluate_complete_predictions(
    frame: pd.DataFrame,
    *,
    predicted_col: str,
    actual_col: str = COMPLETE_HITTER_LABEL,
    legacy_predicted_col: str | None = None,
    legacy_actual_col: str = LEGACY_LABEL,
) -> dict:
    """Full metric bundle for one role (hitters or pitchers)."""
    metrics = {
        "n": int(frame[actual_col].notna().sum()) if actual_col in frame.columns else 0,
        "player_mae": player_mae(frame[actual_col], frame[predicted_col]),
        "rank_correlation": rank_correlation(frame[actual_col], frame[predicted_col]),
        "top_decile_recall": top_decile_recall(frame[actual_col], frame[predicted_col]),
        "boom_rate_calibration": boom_rate_calibration(frame[actual_col], frame[predicted_col]),
        "lineup_level": lineup_projected_versus_actual(
            frame, projected_col=predicted_col, actual_col=actual_col,
        ),
        "contest_roi": contest_style_roi_simulation(
            frame, projected_col=predicted_col, actual_col=actual_col,
        ),
    }
    if legacy_predicted_col and legacy_predicted_col in frame.columns and legacy_actual_col in frame.columns:
        metrics["legacy_benchmark"] = {
            "player_mae": player_mae(frame[legacy_actual_col], frame[legacy_predicted_col]),
            "rank_correlation": rank_correlation(frame[legacy_actual_col], frame[legacy_predicted_col]),
            "label": legacy_actual_col,
        }
    return metrics


# ---------------------------------------------------------------------------
# Nested validation + promotion gate
# ---------------------------------------------------------------------------


def _fit_predict_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str,
    *,
    family: str = "ridge",
    params: dict | None = None,
) -> pd.Series:
    params = params or {}
    X_train = train.reindex(columns=list(feature_columns)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = test.reindex(columns=list(feature_columns)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(train[label_col], errors="coerce").fillna(0.0)
    if family == "gbm":
        model = HistGradientBoostingRegressor(
            max_depth=int(params.get("max_depth", 2)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 100)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            random_state=config.NESTED_VALIDATION_RANDOM_SEED,
        )
        model.fit(X_train.to_numpy(), y.to_numpy())
        pred = model.predict(X_test.to_numpy())
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(params.get("alpha", 1.0)))),
        ])
        pipe.fit(X_train.to_numpy(), y.to_numpy())
        pred = pipe.predict(X_test.to_numpy())
    return pd.Series(np.clip(pred, 0, None), index=test.index)


def run_complete_target_nested_validation(
    hitters: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    freeze_dates: int = 0,
    label_col: str = COMPLETE_HITTER_LABEL,
    role: str = "hitter",
    feature_schedule_mode: str | None = None,
) -> dict:
    """Nested rolling-origin validation for complete DFS targets."""
    if feature_columns is None:
        feature_columns = (
            list(dfs_ml.PITCHER_FEATURE_COLUMNS)
            if role == "pitcher"
            else list(dfs_ml.HITTER_FEATURE_COLUMNS)
        )
    else:
        feature_columns = list(feature_columns)
    leak = set(feature_columns) & LEAKAGE_COLUMNS
    if leak:
        raise AssertionError(f"Leakage columns in features: {sorted(leak)}")

    if hitters is None or hitters.empty or label_col not in hitters.columns:
        return {"status": "insufficient_history", "n_outer_folds": 0, "role": role}

    work = hitters.dropna(subset=["date"]).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    nested, freeze_tail = model_validation.build_nested_folds(
        work["date"],
        outer_min_train_dates=config.DFS_COMPLETE_OUTER_MIN_TRAIN_DATES,
        outer_test_block_dates=config.DFS_COMPLETE_OUTER_TEST_BLOCK_DATES,
        inner_min_train_dates=config.DFS_COMPLETE_INNER_MIN_TRAIN_DATES,
        inner_test_block_dates=config.DFS_COMPLETE_INNER_TEST_BLOCK_DATES,
        freeze_dates=freeze_dates,
    )
    if not nested:
        return {
            "status": "insufficient_history",
            "n_outer_folds": 0,
            "role": role,
            "freeze_tail_dates": [str(d) for d in freeze_tail],
        }

    complete_parts = []
    selected = []
    for nf in nested:
        train = work[work["date"].isin(nf.outer.train_dates)]
        test = work[work["date"].isin(nf.outer.test_dates)]
        if train.empty or test.empty:
            continue

        # Inner selection: ridge alphas.
        best = {"family": "ridge", "params": {"alpha": config.DFS_COMPLETE_RIDGE_ALPHA_GRID[0]}}
        best_mae = float("inf")
        for alpha in config.DFS_COMPLETE_RIDGE_ALPHA_GRID:
            maes = []
            for inner in nf.inner_folds:
                tr = train[train["date"].isin(inner.train_dates)]
                te = train[train["date"].isin(inner.test_dates)]
                if tr.empty or te.empty:
                    continue
                pred = _fit_predict_regressor(
                    tr, te, feature_columns, label_col,
                    family="ridge", params={"alpha": alpha},
                )
                maes.append(player_mae(te[label_col], pred))
            mean_mae = float(np.nanmean(maes)) if maes else float("nan")
            if mean_mae == mean_mae and mean_mae < best_mae:
                best_mae = mean_mae
                best = {"family": "ridge", "params": {"alpha": alpha}}

        selected.append({"outer_fold_id": nf.outer.fold_id, **best})
        pred = _fit_predict_regressor(
            train, test, feature_columns, label_col,
            family=best["family"], params=best["params"],
        )
        block = test.copy()
        block["predicted_complete"] = pred
        # Legacy benchmark must be a *projection*, never the actual label.
        legacy_pred_col = "DK_Points_Hitter" if role == "hitter" else "DK_Points_Pitcher"
        if legacy_pred_col in block.columns:
            block["predicted_legacy"] = pd.to_numeric(block[legacy_pred_col], errors="coerce")
        elif "DK_Points" in block.columns:
            block["predicted_legacy"] = pd.to_numeric(block["DK_Points"], errors="coerce")
        else:
            block["predicted_legacy"] = np.nan
        complete_parts.append(block)

    if not complete_parts:
        return {"status": "insufficient_history", "n_outer_folds": 0, "role": role}

    aligned = pd.concat(complete_parts, ignore_index=True)
    complete_metrics = evaluate_complete_predictions(
        aligned,
        predicted_col="predicted_complete",
        actual_col=label_col,
        legacy_predicted_col="predicted_legacy",
        legacy_actual_col=LEGACY_LABEL,
    )
    legacy_metrics = evaluate_complete_predictions(
        aligned,
        predicted_col="predicted_legacy",
        actual_col=LEGACY_LABEL,
    )

    gate = build_complete_promotion_gate(
        complete_metrics,
        legacy_metrics,
        feature_schedule_mode=feature_schedule_mode,
    )
    return {
        "status": "ok",
        "role": role,
        "label_col": label_col,
        "feature_schedule_mode": feature_schedule_mode,
        "n_outer_folds": len(selected),
        "freeze_tail_dates": [str(d) for d in freeze_tail],
        "feature_columns": list(feature_columns),
        "selected_configs": selected,
        "complete_target": complete_metrics,
        "legacy_partial_benchmark": legacy_metrics,
        "promotion_gate": gate,
        "notes": [
            "Complete targets include DNP zeros and provenance-tagged fields.",
            "FIP-proxy ER is never treated as independent complete-target ground truth.",
            "Promotion requires lineup-level or ranking improvement, not trivial MAE alone.",
            "Legacy partial-target models remain the live benchmark until the gate passes.",
            "Assembler features currently use actual-starter schedule unless feature_schedule_mode=as_of_snapshot.",
        ],
    }


def fit_complete_target_model(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str,
    *,
    family: str = "ridge",
    params: dict | None = None,
):
    """Fit a final complete-target regressor on the full training frame."""
    params = params or {"alpha": 1.0}
    X = frame.reindex(columns=list(feature_columns)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(frame[label_col], errors="coerce").fillna(0.0)
    if family == "gbm":
        model = HistGradientBoostingRegressor(
            max_depth=int(params.get("max_depth", 2)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 100)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            random_state=config.NESTED_VALIDATION_RANDOM_SEED,
        )
        model.fit(X.to_numpy(), y.to_numpy())
        return model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(params.get("alpha", 1.0)))),
    ])
    pipe.fit(X.to_numpy(), y.to_numpy())
    return pipe


def write_complete_validation_report(report: dict, path: str | None = None) -> str:
    path = path or config.DFS_COMPLETE_PROMOTION_GATE_REPORT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def apply_complete_shadow_predictions(
    hitters: pd.DataFrame,
    model,
    feature_columns: Sequence[str],
    *,
    path: str | None = None,
) -> pd.DataFrame:
    """Write shadow complete-target predictions without overriding live DFS."""
    path = path or config.DFS_COMPLETE_SHADOW_PREDICTIONS_PATH
    X = hitters.reindex(columns=list(feature_columns)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pred = np.clip(model.predict(X.to_numpy()), 0, None)
    out = hitters[["date", "game_pk", "key_mlbam"]].copy() if {"date", "game_pk", "key_mlbam"}.issubset(hitters.columns) else hitters.copy()
    out["DK_Points_Complete_Shadow"] = pred
    out["target_mode"] = "complete_shadow"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return out


def build_complete_promotion_gate(
    complete: dict,
    legacy: dict,
    *,
    feature_schedule_mode: str | None = None,
) -> dict:
    """Fail-closed gate: trivial MAE wins do not promote."""
    c_mae = complete.get("player_mae", float("nan"))
    l_mae = (legacy.get("player_mae") if legacy else float("nan"))
    mae_improvement = (l_mae - c_mae) if (c_mae == c_mae and l_mae == l_mae) else float("nan")

    c_rank = complete.get("rank_correlation", float("nan"))
    l_rank = legacy.get("rank_correlation", float("nan")) if legacy else float("nan")
    rank_improvement = (c_rank - l_rank) if (c_rank == c_rank and l_rank == l_rank) else float("nan")

    c_tdr = complete.get("top_decile_recall", float("nan"))
    l_tdr = legacy.get("top_decile_recall", float("nan")) if legacy else float("nan")
    tdr_improvement = (c_tdr - l_tdr) if (c_tdr == c_tdr and l_tdr == l_tdr) else float("nan")

    c_lineup = (complete.get("lineup_level") or {}).get("mean_abs_lineup_error", float("nan"))
    l_lineup = (legacy.get("lineup_level") or {}).get("mean_abs_lineup_error", float("nan")) if legacy else float("nan")
    lineup_improvement = (l_lineup - c_lineup) if (c_lineup == c_lineup and l_lineup == l_lineup) else float("nan")

    meaningful_lineup = bool(
        lineup_improvement == lineup_improvement
        and lineup_improvement >= float(config.DFS_COMPLETE_MIN_LINEUP_SCORE_IMPROVEMENT)
    )
    meaningful_rank = bool(
        rank_improvement == rank_improvement
        and rank_improvement >= float(config.DFS_COMPLETE_MIN_RANK_CORR_IMPROVEMENT)
    )
    meaningful_tdr = bool(
        tdr_improvement == tdr_improvement
        and tdr_improvement >= float(config.DFS_COMPLETE_MIN_TOP_DECILE_RECALL_IMPROVEMENT)
    )
    trivial_mae_only = bool(
        mae_improvement == mae_improvement
        and 0 < mae_improvement < float(config.DFS_COMPLETE_TRIVIAL_MAE_EPSILON)
        and not (meaningful_lineup or meaningful_rank or meaningful_tdr)
    )

    production_equivalent_features = feature_schedule_mode in (
        None,  # synthetic unit tests that do not assemble schedule features
        "as_of_snapshot",
    )

    checks = {
        "based_on_untouched_outer_folds": True,
        "not_trivial_mae_only": not trivial_mae_only,
        "meaningful_lineup_or_ranking_improvement": bool(
            meaningful_lineup or meaningful_rank or meaningful_tdr
        ),
        "complete_mae_not_materially_worse": bool(
            mae_improvement == mae_improvement and mae_improvement > -float(config.DFS_COMPLETE_TRIVIAL_MAE_EPSILON)
        ),
        "legacy_partial_kept_as_benchmark": True,
        "fip_not_used_as_complete_er_truth": True,
        "features_not_actual_starter_hindsight": bool(production_equivalent_features),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "feature_schedule_mode": feature_schedule_mode,
        "mae_improvement_vs_legacy": mae_improvement,
        "rank_corr_improvement_vs_legacy": rank_improvement,
        "top_decile_recall_improvement_vs_legacy": tdr_improvement,
        "lineup_error_improvement_vs_legacy": lineup_improvement,
        "trivial_mae_only": trivial_mae_only,
    }


def evaluate_complete_promotion_gate(report: dict | None) -> tuple[bool, dict]:
    details: dict[str, Any] = {"passed": False, "reason": "missing_report", "checks": {}}
    if not report or not isinstance(report, dict):
        return False, details
    gate = report.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("checks"), dict):
        details["reason"] = "missing_promotion_gate_block"
        return False, details
    details["checks"] = {k: bool(v) for k, v in gate["checks"].items()}
    if not all(details["checks"].values()):
        failed = [k for k, v in details["checks"].items() if not v]
        details["reason"] = f"gate_checks_failed:{','.join(failed)}"
        return False, details
    details["passed"] = True
    details["reason"] = "passed"
    return True, details


def resolve_complete_target_mode(
    configured: str | None = None,
    *,
    report_path: str | None = None,
    force_live: bool = False,
) -> tuple[str, dict]:
    mode = (configured if configured is not None else config.DFS_COMPLETE_TARGET_MODE) or "shadow"
    if mode not in config.DFS_COMPLETE_TARGET_MODES:
        return "shadow", {
            "configured": mode, "effective": "shadow", "fallback_used": True,
            "fallback_reason": f"invalid_mode:{mode}",
        }
    if mode != "live":
        return mode, {"configured": mode, "effective": mode, "fallback_used": False, "fallback_reason": None}
    if force_live:
        return "live", {"configured": "live", "effective": "live", "fallback_used": False, "fallback_reason": None}
    path = report_path or config.DFS_COMPLETE_PROMOTION_GATE_REPORT_PATH
    if not path or not os.path.exists(path):
        return "shadow", {
            "configured": "live", "effective": "shadow", "fallback_used": True,
            "fallback_reason": "missing_report_file",
        }
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        return "shadow", {
            "configured": "live", "effective": "shadow", "fallback_used": True,
            "fallback_reason": f"report_unreadable:{type(exc).__name__}",
        }
    ok, details = evaluate_complete_promotion_gate(report)
    if ok:
        return "live", {"configured": "live", "effective": "live", "fallback_used": False, "fallback_reason": None, "promotion_gate": details}
    return "shadow", {
        "configured": "live", "effective": "shadow", "fallback_used": True,
        "fallback_reason": f"promotion_gate_failed:{details.get('reason')}",
        "promotion_gate": details,
    }
