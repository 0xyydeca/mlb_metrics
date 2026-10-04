"""Development-only three-state hitter model with official boxscore labels.

This module never serves predictions or reads registered evaluation outcomes.
Development labels fetched after games are outcomes, not pregame features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlb_metrics import config
from mlb_metrics.venues.polymarket_us import normalize_team_abbr

KEYS = ["date", "game_pk", "key_mlbam"]
CLASS_NAMES = ["nonqualifying", "qualifying_no_hit", "qualifying_hit"]


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return int(x) if np.isfinite(x) and x >= 0 and x.is_integer() else None


def official_player_label(game, boxscore, row):
    """Return state only from an exact final game/team/player; never infer DNP."""
    pk, key = _integer(row.get("game_pk")), _integer(row.get("key_mlbam"))
    if pk is None or key is None or pk <= 0 or key <= 0 or game.get("gamePk") != pk:
        return None, "invalid_or_mismatched_identity"
    if game.get("officialDate") != str(row["date"]):
        return None, "official_date_mismatch"
    if game.get("gameType") != "R" or game.get("status", {}).get("abstractGameState") != "Final":
        return None, "not_final_regular_season"
    if any(game.get(k) for k in ("resumeDate", "resumeGameDate", "rescheduledFromDate", "rescheduledGameDate")):
        return None, "rescheduled_completion_cutoff_unverified"
    team = normalize_team_abbr(row.get("team"))
    sides = [v for v in boxscore.get("teams", {}).values()
             if normalize_team_abbr(v.get("team", {}).get("abbreviation")) == team]
    if not team or len(sides) != 1:
        return None, "team_mismatch"
    side = sides[0]
    player = side.get("players", {}).get(f"ID{key}")
    if not player or player.get("person", {}).get("id") != key:
        return None, "player_missing_from_official_roster"
    order = _integer(player.get("battingOrder"))
    status = player.get("gameStatus", {})
    if order is None:
        if status.get("isOnBench") is True and key not in side.get("batters", []):
            return 0, "official_unused_bench"
        return None, "starting_status_unknown"
    if not 100 <= order <= 999:
        return None, "invalid_batting_order"
    started = order % 100 == 0
    substitute = status.get("isSubstitute")
    if not isinstance(substitute, bool) or substitute == started:
        return None, "conflicting_or_missing_substitution_status"
    # A substitute may get a hit, but does not meet the starting-lineup rule.
    if not started:
        return 0, "official_substitute"
    batting = player.get("stats", {}).get("batting", {})
    pa, ab, hits = (_integer(batting.get(k)) for k in ("plateAppearances", "atBats", "hits"))
    if pa is None or ab is None or hits is None or not hits <= ab <= pa:
        return None, "unknown_or_invalid_official_counts"
    if pa == 0:
        return 0, "official_starter_zero_pa"
    return (2 if hits > 0 else 1), "official_qualifying_pa"


def prepare_development_frame(frame, schedule, boxscore_dir):
    """Validate input provenance and return labels plus a complete exclusion log."""
    required = set(KEYS + ["team", "feature_as_of_timestamp"] + list(config.HIT_PROP_DEV_FEATURES))
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing development columns: {sorted(missing)}")
    work = frame.copy()
    dates = pd.to_datetime(work["date"], errors="coerce")
    if dates.isna().any() or (dates != dates.dt.normalize()).any():
        raise ValueError("Dates must be valid normalized calendar dates")
    work["date"] = dates.dt.strftime("%Y-%m-%d")
    if (work["date"] >= config.HIT_PROP_DEV_END_EXCLUSIVE).any():
        raise ValueError("Input overlaps protected post-development dates")
    for k in ("game_pk", "key_mlbam"):
        values = work[k].map(_integer)
        if values.isna().any() or (values <= 0).any():
            raise ValueError(f"Invalid identity: {k}")
        work[k] = values.astype(int)
    if work.duplicated(KEYS).any():
        raise ValueError("Duplicate date/game/player keys")
    stamps = pd.to_datetime(work["feature_as_of_timestamp"], utc=True, errors="coerce")
    morning = pd.to_datetime(work["date"], utc=True)
    if stamps.isna().any() or (stamps > morning).any():
        raise ValueError("Feature timestamp missing or later than morning cutoff")
    games = {g["gamePk"]: g for day in schedule.get("dates", []) for g in day.get("games", [])}
    cache, labels, reasons, hashes = {}, [], [], {}
    for row in work.to_dict("records"):
        pk = row["game_pk"]
        if pk not in cache:
            path = Path(boxscore_dir) / f"{pk}.json"
            if path.exists():
                raw = path.read_bytes()
                record = json.loads(raw)
                if record.get("game_pk") != pk:
                    raise ValueError("Cached boxscore identity mismatch")
                cache[pk] = record["payload"]
                hashes[str(pk)] = hashlib.sha256(raw).hexdigest()
            else:
                cache[pk] = None
        if cache[pk] is None or pk not in games:
            label, reason = None, "missing_official_game_or_boxscore"
        else:
            label, reason = official_player_label(games[pk], cache[pk], row)
        labels.append(label)
        reasons.append(reason)
    work["target_state"] = labels
    work["label_reason"] = reasons
    excluded = work[work.target_state.isna()][KEYS + ["label_reason"]].copy()
    included = work[work.target_state.notna()].copy()
    included["target_state"] = included.target_state.astype(int)
    # Deliberately retain only allowlisted pregame columns and audit metadata.
    included = included[KEYS + ["feature_as_of_timestamp", "target_state", "label_reason"] + list(config.HIT_PROP_DEV_FEATURES)]
    return included, excluded, hashes


def _features(frame):
    return frame[list(config.HIT_PROP_DEV_FEATURES)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_model(train):
    if set(train.target_state.unique()) != {0, 1, 2}:
        raise ValueError("Training requires all three outcome states")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=config.HIT_PROP_DEV_LOGISTIC_C, max_iter=config.HIT_PROP_DEV_MAX_ITER),
    )
    model.fit(_features(train), train.target_state)
    return model


def predict_states(model, frame):
    probs = model.predict_proba(_features(frame))
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Unexpected class ordering")
    return probs


def payout_bounds(probabilities):
    """Yes-share expected payout bounds when nonqualification LFMP is unknown.

    The bounds condition on this three-state target; game interruptions and
    other venue-specific nonbinary outcomes are not modeled here.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Expected a normalized three-state probability matrix")
    return np.column_stack([p[:, 2], p[:, 2] + p[:, 0]])


def chronological_development(frame):
    """Fixed forward blocks; no tuning, formal holdout, or market-return scoring."""
    dates = sorted(frame.date.unique())
    first, width = config.HIT_PROP_DEV_INITIAL_DATES, config.HIT_PROP_DEV_TEST_DATES
    predictions, folds = [], []
    for offset in range(first, len(dates), width):
        train = frame[frame.date.isin(dates[:offset])]
        test = frame[frame.date.isin(dates[offset:offset + width])]
        if test.empty or set(train.target_state.unique()) != {0, 1, 2}:
            continue
        model = fit_model(train)
        probs = predict_states(model, test)
        base = train.target_state.value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
        bounds = payout_bounds(probs)
        part = test[KEYS + ["target_state"]].copy()
        for i, name in enumerate(CLASS_NAMES):
            part["p_" + name] = probs[:, i]
            part["baseline_p_" + name] = base[i]
        part["expected_yes_payout_lower"] = bounds[:, 0]
        part["expected_yes_payout_upper"] = bounds[:, 1]
        part["training_end_date"] = train.date.max()
        part["fold"] = len(folds)
        predictions.append(part)
        folds.append({"training_start": train.date.min(), "training_end": train.date.max(),
                      "test_start": test.date.min(), "test_end": test.date.max(),
                      "n_training_rows": len(train), "n_test_rows": len(test)})
    if not predictions:
        return pd.DataFrame(), {"status": "insufficient_data", "reason": "no_complete_training_blocks", "folds": folds}
    pred = pd.concat(predictions, ignore_index=True)
    y = pred.target_state.to_numpy()
    p = pred[["p_"+n for n in CLASS_NAMES]].to_numpy()
    b = pred[["baseline_p_"+n for n in CLASS_NAMES]].to_numpy()
    eps = config.HIT_PROP_DEV_LOG_EPSILON
    loss = -np.log(np.clip(p[np.arange(len(y)), y], eps, 1))
    baseline_loss = -np.log(np.clip(b[np.arange(len(y)), y], eps, 1))
    by_date = pd.DataFrame({"date": pred.date, "difference": loss-baseline_loss}).groupby("date").difference.agg(["sum", "count"])
    rng = np.random.default_rng(config.HIT_PROP_DEV_SEED)
    idx = rng.integers(0, len(by_date), size=(config.HIT_PROP_DEV_BOOTSTRAP_SAMPLES, len(by_date)))
    bootstrap = by_date["sum"].to_numpy()[idx].sum(axis=1)/by_date["count"].to_numpy()[idx].sum(axis=1)
    onehot = np.eye(3)[y]
    report = {"status": "development_only", "folds": folds, "n_rows": len(pred), "n_dates": pred.date.nunique(),
              "log_loss": float(loss.mean()), "baseline_log_loss": float(baseline_loss.mean()),
              "multiclass_brier_sum": float(np.mean(np.sum((p-onehot)**2, axis=1))),
              "baseline_multiclass_brier_sum": float(np.mean(np.sum((b-onehot)**2, axis=1))),
              "paired_log_loss_difference": float(np.mean(loss-baseline_loss)),
              "date_block_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "bootstrap_samples": config.HIT_PROP_DEV_BOOTSTRAP_SAMPLES,
              "uncertainty_limit": "Resamples dates; repeated-player and serial dependence across dates are not fully modeled.",
              "formal_holdout": False, "market_comparison": "unavailable_no_matched_historical_executable_prices",
              "returns": None, "real_money_ready": False}
    return pred, report
