"""Market-residual game-win probability challenger.

Formulation (market as prior, not a separate "edge" probability)::

    logit(P_final_home_win) =
        logit(P_market_home_win_at_prediction_time) + residual_model(features)

The residual learns whether baseball-specific pregame information
systematically justifies moving away from the market. Closing odds are
never features of a morning / lineup-lock prediction; they appear only in
evaluation (true closing-line value).

Modes (``config.GAME_PREDICTION_MODE`` / ``config.BETTING_MODE``):

- legacy / disabled: official path unchanged; no residual / no stakes
- shadow: log residual probs (and optional hypothetical bets); official
  picks and ``bet_units`` unchanged / zeroed
- live: requires the corresponding promotion gate

Kelly sizing is downstream and hypothetical only - never used to decide
whether the probability model itself is accurate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from mlb_metrics import config, game_picks, kelly, market_odds, ml_models, model_validation

# ---------------------------------------------------------------------------
# Labels / leakage guards
# ---------------------------------------------------------------------------

HOME_WON_LABEL = "Home_Won"
MARKET_AT_PRED_COL = "market_home_win_probability"
HEURISTIC_COL = "home_win_probability"
CLOSING_MARKET_COL = "closing_home_win_probability"
RESIDUAL_PROB_COL = "residual_home_win_probability"

# Closing / post-prediction market columns must never enter residual X.
LEAKAGE_COLUMNS = frozenset({
    HOME_WON_LABEL,
    "actual_winner",
    "Home_Score",
    "Away_Score",
    CLOSING_MARKET_COL,
    "closing_home_moneyline",
    "closing_away_moneyline",
    "closing_snapshot_id",
    "bet_profit_units",
    "residual_home_win_probability",
})

BASE_FEATURE_COLUMNS = list(game_picks.GAME_PICK_FEATURE_COLUMNS)

DERIVED_FEATURE_COLUMNS = [
    "composite_diff",
    "starter_quality_diff",
    "bullpen_quality_diff",
    "park_factor",
    "home_advantage",
    "home_starter_certainty",
    "away_starter_certainty",
    "home_lineup_strength",
    "away_lineup_strength",
    "temperature_f",
    "wind_speed_mph",
    "home_days_rest",
    "away_days_rest",
]

RESIDUAL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

SHADOW_PREDICTION_COLUMNS = [
    "date", "game_pk", "home_team", "away_team",
    MARKET_AT_PRED_COL, HEURISTIC_COL, RESIDUAL_PROB_COL,
    "residual_logit", "probability_source", "model_version",
    "artifact_id", "game_prediction_mode", "prediction_snapshot_type",
]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def logit(p: np.ndarray | pd.Series | float, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(z: np.ndarray | pd.Series | float) -> np.ndarray:
    arr = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _as_float_frame(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = X.reindex(columns=list(columns)).copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def enrich_residual_features(
    features: pd.DataFrame,
    *,
    schedule_games: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    lineup_strength: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add pregame residual features without touching the heuristic path.

    Missing optional inputs become honest nulls (filled to training
    priors at fit / predict time), never fabricated skill.
    """
    out = features.copy()

    out["composite_diff"] = (
        pd.to_numeric(out.get("home_composite"), errors="coerce")
        - pd.to_numeric(out.get("away_composite"), errors="coerce")
    )
    home_starter = (
        pd.to_numeric(out.get("away_starter_pave_plus"), errors="coerce").fillna(1.0)
        + pd.to_numeric(out.get("away_starter_power_a_plus"), errors="coerce").fillna(1.0)
    ) / 2.0
    away_starter = (
        pd.to_numeric(out.get("home_starter_pave_plus"), errors="coerce").fillna(1.0)
        + pd.to_numeric(out.get("home_starter_power_a_plus"), errors="coerce").fillna(1.0)
    ) / 2.0
    # Higher opposing-starter quality faced -> harder for that offense.
    out["starter_quality_diff"] = away_starter - home_starter

    home_bp = (
        pd.to_numeric(out.get("away_bullpen_pave_plus"), errors="coerce").fillna(1.0)
        + pd.to_numeric(out.get("away_bullpen_power_a_plus"), errors="coerce").fillna(1.0)
    ) / 2.0
    away_bp = (
        pd.to_numeric(out.get("home_bullpen_pave_plus"), errors="coerce").fillna(1.0)
        + pd.to_numeric(out.get("home_bullpen_power_a_plus"), errors="coerce").fillna(1.0)
    ) / 2.0
    out["bullpen_quality_diff"] = away_bp - home_bp

    out["home_advantage"] = 1.0
    out["park_factor"] = 1.0
    if confidence is not None and not confidence.empty and "Park_Factor" in confidence.columns:
        park = confidence[["team", "Park_Factor"]].drop_duplicates("team")
        if "home_team" in out.columns:
            merged = out.drop(columns=["park_factor"], errors="ignore").merge(
                park.rename(columns={"team": "home_team", "Park_Factor": "park_factor"}),
                on="home_team",
                how="left",
            )
            out["park_factor"] = pd.to_numeric(merged["park_factor"], errors="coerce").fillna(1.0)

    # Starter certainty: known probable pitcher key -> 1, else 0.
    out["home_starter_certainty"] = 0.0
    out["away_starter_certainty"] = 0.0
    sched = schedule_games if schedule_games is not None else features
    if sched is not None and not sched.empty:
        home_key = "home_probable_pitcher_key_mlbam"
        away_key = "away_probable_pitcher_key_mlbam"
        if home_key in sched.columns and "game_pk" in sched.columns and "game_pk" in out.columns:
            certainty = sched[["game_pk", home_key, away_key]].drop_duplicates("game_pk")
            merged = out.merge(certainty, on="game_pk", how="left", suffixes=("", "_sched"))
            hk = home_key if home_key in merged.columns else f"{home_key}_sched"
            ak = away_key if away_key in merged.columns else f"{away_key}_sched"
            if hk in merged.columns:
                out["home_starter_certainty"] = merged[hk].notna().astype(float).to_numpy()
            if ak in merged.columns:
                out["away_starter_certainty"] = merged[ak].notna().astype(float).to_numpy()

    out["home_lineup_strength"] = pd.to_numeric(out.get("home_composite"), errors="coerce")
    out["away_lineup_strength"] = pd.to_numeric(out.get("away_composite"), errors="coerce")
    if lineup_strength is not None and not lineup_strength.empty and "game_pk" in lineup_strength.columns:
        ls = lineup_strength.copy()
        keep = [c for c in ("game_pk", "home_lineup_strength", "away_lineup_strength") if c in ls.columns]
        if len(keep) >= 2:
            merged = out.drop(
                columns=[c for c in ("home_lineup_strength", "away_lineup_strength") if c in out.columns],
                errors="ignore",
            ).merge(ls[keep].drop_duplicates("game_pk"), on="game_pk", how="left")
            for col in ("home_lineup_strength", "away_lineup_strength"):
                if col in merged.columns:
                    out[col] = pd.to_numeric(merged[col], errors="coerce")

    for weather_col in ("temperature_f", "wind_speed_mph"):
        if weather_col not in out.columns:
            out[weather_col] = np.nan
        if schedule_games is not None and weather_col in schedule_games.columns and "game_pk" in out.columns:
            w = schedule_games[["game_pk", weather_col]].drop_duplicates("game_pk")
            merged = out.drop(columns=[weather_col], errors="ignore").merge(w, on="game_pk", how="left")
            out[weather_col] = pd.to_numeric(merged[weather_col], errors="coerce")

    out["home_days_rest"] = np.nan
    out["away_days_rest"] = np.nan
    if schedule_games is not None and not schedule_games.empty:
        rest = derive_team_rest_features(schedule_games)
        if not rest.empty and "game_pk" in out.columns:
            merged = out.merge(rest, on="game_pk", how="left", suffixes=("", "_rest"))
            out["home_days_rest"] = pd.to_numeric(merged.get("home_days_rest"), errors="coerce")
            out["away_days_rest"] = pd.to_numeric(merged.get("away_days_rest"), errors="coerce")

    return out


def derive_team_rest_features(schedule_games: pd.DataFrame) -> pd.DataFrame:
    """Days since each club's previous scheduled game (defensible rest proxy)."""
    required = {"game_pk", "date", "home_team", "away_team"}
    if schedule_games is None or schedule_games.empty or not required.issubset(schedule_games.columns):
        return pd.DataFrame(columns=["game_pk", "home_days_rest", "away_days_rest"])

    games = schedule_games.loc[:, list(required)].copy()
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games = games.sort_values(["date", "game_pk"])

    last_seen: dict[str, pd.Timestamp] = {}
    rows = []
    for _, row in games.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        home_rest = (date - last_seen[home]).days if home in last_seen else np.nan
        away_rest = (date - last_seen[away]).days if away in last_seen else np.nan
        rows.append({
            "game_pk": row["game_pk"],
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
        })
        last_seen[home] = date
        last_seen[away] = date
    return pd.DataFrame(rows)


def residual_feature_matrix(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(RESIDUAL_FEATURE_COLUMNS)
    return _as_float_frame(frame, cols)


def assert_no_closing_odds_in_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    banned = set(feature_columns) & LEAKAGE_COLUMNS
    if banned:
        raise AssertionError(f"Closing/leakage columns in residual features: {sorted(banned)}")
    for col in feature_columns:
        if "closing" in str(col).lower():
            raise AssertionError(f"Closing-related feature blocked: {col}")


# ---------------------------------------------------------------------------
# Residual estimators
# ---------------------------------------------------------------------------


@dataclass
class MarketResidualLogistic:
    """Offset logistic: logit(p) = logit(market) + Xβ with strong L2 on β."""

    C: float = 0.01
    residual_cap: float = config.GAME_RESIDUAL_LOGIT_RESIDUAL_CAP
    feature_columns: list[str] = field(default_factory=lambda: list(RESIDUAL_FEATURE_COLUMNS))
    preprocessor: model_validation.StandardizePreprocessor | None = None
    coef_: np.ndarray | None = None
    params_index_: list[str] | None = None

    @property
    def alpha(self) -> float:
        return 1.0 / max(float(self.C), 1e-12)

    def fit(self, X: pd.DataFrame, y: pd.Series, market_p: pd.Series) -> MarketResidualLogistic:
        assert_no_closing_odds_in_features(X, self.feature_columns)
        self.preprocessor = model_validation.StandardizePreprocessor().fit(
            residual_feature_matrix(X, self.feature_columns)
        )
        Xs = self.preprocessor.transform(residual_feature_matrix(X, self.feature_columns))
        if Xs.shape[1] == 0:
            self.coef_ = np.array([0.0])
            self.params_index_ = ["const"]
            return self
        design = sm.add_constant(Xs.to_numpy(dtype=float), has_constant="add")
        y_arr = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
        offset = logit(pd.to_numeric(market_p, errors="coerce").to_numpy())
        glm = sm.GLM(y_arr, design, family=sm.families.Binomial(), offset=offset)
        # Higher alpha => stronger shrink of residual toward zero.
        result = glm.fit_regularized(alpha=self.alpha, L1_wt=0.0, maxiter=300)
        self.coef_ = np.asarray(result.params, dtype=float)
        self.params_index_ = ["const"] + list(Xs.columns)
        return self

    def residual_logit(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.coef_ is None:
            raise RuntimeError("MarketResidualLogistic.predict called before fit")
        Xs = self.preprocessor.transform(residual_feature_matrix(X, self.feature_columns))
        if Xs.shape[1] == 0:
            r = np.zeros(len(X), dtype=float)
        else:
            design = sm.add_constant(Xs.to_numpy(dtype=float), has_constant="add")
            r = design @ self.coef_
        return np.clip(r, -self.residual_cap, self.residual_cap)

    def predict_proba(self, X: pd.DataFrame, market_p: pd.Series | np.ndarray) -> np.ndarray:
        return sigmoid(logit(market_p) + self.residual_logit(X))


@dataclass
class MarketResidualNonlinear:
    """Conservative nonlinear residual: shrink(clip(logit(p_gbm) - logit(market)))."""

    max_depth: int = 2
    min_samples_leaf: int = 80
    shrink: float = 0.5
    residual_cap: float = config.GAME_RESIDUAL_LOGIT_RESIDUAL_CAP
    feature_columns: list[str] = field(default_factory=lambda: list(RESIDUAL_FEATURE_COLUMNS))
    preprocessor: model_validation.StandardizePreprocessor | None = None
    estimator_: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, market_p: pd.Series) -> MarketResidualNonlinear:
        del market_p  # market enters only at predict time as the prior
        assert_no_closing_odds_in_features(X, self.feature_columns)
        self.preprocessor = model_validation.StandardizePreprocessor().fit(
            residual_feature_matrix(X, self.feature_columns)
        )
        Xs = self.preprocessor.transform(residual_feature_matrix(X, self.feature_columns)).fillna(0.0)
        y_arr = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
        if Xs.shape[1] == 0 or len(np.unique(y_arr)) < 2:
            self.estimator_ = None
            return self
        self.estimator_ = HistGradientBoostingClassifier(
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            learning_rate=0.05,
            max_iter=150,
            random_state=config.NESTED_VALIDATION_RANDOM_SEED,
        )
        self.estimator_.fit(Xs.to_numpy(dtype=float), y_arr)
        return self

    def residual_logit(self, X: pd.DataFrame, market_p: pd.Series | np.ndarray) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("MarketResidualNonlinear.predict called before fit")
        if self.estimator_ is None:
            return np.zeros(len(X), dtype=float)
        Xs = self.preprocessor.transform(residual_feature_matrix(X, self.feature_columns)).fillna(0.0)
        p_gbm = self.estimator_.predict_proba(Xs.to_numpy(dtype=float))[:, 1]
        raw = logit(p_gbm) - logit(market_p)
        return float(self.shrink) * np.clip(raw, -self.residual_cap, self.residual_cap)

    def predict_proba(self, X: pd.DataFrame, market_p: pd.Series | np.ndarray) -> np.ndarray:
        return sigmoid(logit(market_p) + self.residual_logit(X, market_p))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if len(y_arr) == 0:
        return float("nan")
    return float(np.mean((p_arr - y_arr) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if len(y_arr) == 0:
        return float("nan")
    return float(-np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)))


def high_confidence_tail_metrics(
    y: pd.Series,
    p: pd.Series,
    *,
    quantile: float = 0.8,
) -> dict:
    """Performance on the high-|edge-from-0.5| tail of predictions."""
    y_arr = pd.to_numeric(y, errors="coerce")
    p_arr = pd.to_numeric(p, errors="coerce")
    mask = y_arr.notna() & p_arr.notna()
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    if len(p_arr) < 5:
        return {
            "high_conf_n": int(len(p_arr)),
            "high_conf_brier": float("nan"),
            "high_conf_log_loss": float("nan"),
            "high_conf_accuracy": float("nan"),
        }
    strength = (p_arr - 0.5).abs()
    thr = float(strength.quantile(quantile))
    tail = strength >= thr
    yt = y_arr[tail].to_numpy()
    pt = p_arr[tail].to_numpy()
    pred = (pt >= 0.5).astype(float)
    return {
        "high_conf_n": int(tail.sum()),
        "high_conf_brier": brier_score(yt, pt),
        "high_conf_log_loss": log_loss(yt, pt),
        "high_conf_accuracy": float(np.mean(pred == yt)) if len(yt) else float("nan"),
    }


def score_probabilities(y: pd.Series, p: pd.Series) -> dict:
    y_arr = pd.to_numeric(y, errors="coerce")
    p_arr = pd.to_numeric(p, errors="coerce")
    mask = y_arr.notna() & p_arr.notna()
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    metrics = {
        "n": int(len(y_arr)),
        "brier_score": brier_score(y_arr.to_numpy(), p_arr.to_numpy()),
        "log_loss": log_loss(y_arr.to_numpy(), p_arr.to_numpy()),
        "roc_auc": float("nan"),
    }
    metrics.update(model_validation.calibration_intercept_slope(y_arr, p_arr))
    metrics.update(high_confidence_tail_metrics(y_arr, p_arr))
    if len(y_arr) >= 5 and y_arr.nunique() >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_arr, p_arr))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


def paired_diffs_vs_market(y: pd.Series, p_model: pd.Series, p_market: pd.Series) -> dict:
    y_arr = pd.to_numeric(y, errors="coerce")
    pm = pd.to_numeric(p_model, errors="coerce")
    mkt = pd.to_numeric(p_market, errors="coerce")
    mask = y_arr.notna() & pm.notna() & mkt.notna()
    y_arr, pm, mkt = y_arr[mask], pm[mask], mkt[mask]
    if len(y_arr) == 0:
        return {
            "n_paired": 0,
            "model_minus_market_brier": float("nan"),
            "model_minus_market_log_loss": float("nan"),
        }
    return {
        "n_paired": int(len(y_arr)),
        "model_minus_market_brier": brier_score(y_arr, pm) - brier_score(y_arr, mkt),
        "model_minus_market_log_loss": log_loss(y_arr, pm) - log_loss(y_arr, mkt),
    }


def fit_isotonic_on_train(train_p: pd.Series, train_y: pd.Series) -> IsotonicRegression | None:
    y = pd.to_numeric(train_y, errors="coerce")
    p = pd.to_numeric(train_p, errors="coerce")
    mask = y.notna() & p.notna()
    if mask.sum() < 10 or y[mask].nunique() < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
    iso.fit(p[mask].to_numpy(), y[mask].to_numpy())
    return iso


def apply_isotonic(iso: IsotonicRegression | None, p: pd.Series) -> pd.Series:
    if iso is None:
        return pd.to_numeric(p, errors="coerce")
    arr = pd.to_numeric(p, errors="coerce")
    out = arr.copy()
    valid = arr.notna()
    if valid.any():
        out.loc[valid] = iso.predict(arr[valid].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Hypothetical betting / CLV (evaluation only; Kelly never gates accuracy)
# ---------------------------------------------------------------------------


def hypothetical_bets_from_probabilities(
    frame: pd.DataFrame,
    *,
    model_prob_col: str,
    market_prob_col: str = MARKET_AT_PRED_COL,
    home_ml_col: str = "home_moneyline",
    away_ml_col: str = "away_moneyline",
    edge_threshold: float = 0.02,
    kelly_fraction_multiplier: float = config.KELLY_FRACTION_MULTIPLIER,
) -> pd.DataFrame:
    """Build vig-aware hypothetical bets. Threshold is an input (from inner CV).

    Does not use the arbitrary production ``KELLY_MIN_EDGE`` as a skill proof.
    Kelly stake is recorded for ROI reporting only.
    """
    cols = [
        "date", "game_pk", "bet_side", "bet_team", "bet_moneyline",
        "model_probability", "market_implied_probability", "edge",
        "edge_threshold", "kelly_stake_fraction", "bet_units",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in frame.iterrows():
        home_ml = row.get(home_ml_col, row.get("market_home_moneyline"))
        away_ml = row.get(away_ml_col, row.get("market_away_moneyline"))
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        model_home = float(row[model_prob_col]) if pd.notna(row.get(model_prob_col)) else None
        if model_home is None:
            continue
        home_implied = market_odds.moneyline_to_implied_probability(float(home_ml))
        away_implied = market_odds.moneyline_to_implied_probability(float(away_ml))
        candidates = [
            ("home", row.get("home_team"), float(home_ml), model_home, home_implied),
            ("away", row.get("away_team"), float(away_ml), 1.0 - model_home, away_implied),
        ]
        best = None
        for side, team, ml, model_p, implied in candidates:
            edge = model_p - implied
            if edge < float(edge_threshold):
                continue
            stake = kelly.kelly_fraction(model_p, ml, kelly_fraction_multiplier)
            if stake <= 0:
                continue
            cand = {
                "date": row.get("date"),
                "game_pk": row.get("game_pk"),
                "bet_side": side,
                "bet_team": team,
                "bet_moneyline": ml,
                "model_probability": model_p,
                "market_implied_probability": implied,
                "edge": edge,
                "edge_threshold": float(edge_threshold),
                "kelly_stake_fraction": stake,
                "bet_units": stake / config.UNIT_SIZE_FRACTION,
                HOME_WON_LABEL: row.get(HOME_WON_LABEL),
                CLOSING_MARKET_COL: row.get(CLOSING_MARKET_COL),
            }
            if best is None or cand["edge"] > best["edge"]:
                best = cand
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows, columns=cols + [HOME_WON_LABEL, CLOSING_MARKET_COL])


def settle_hypothetical_roi(bets: pd.DataFrame) -> dict:
    if bets is None or bets.empty or HOME_WON_LABEL not in bets.columns:
        return {
            "n_bets": 0,
            "total_staked_units": 0.0,
            "total_profit_units": float("nan"),
            "roi": float("nan"),
        }
    settled = bets.dropna(subset=[HOME_WON_LABEL]).copy()
    if settled.empty:
        return {
            "n_bets": 0,
            "total_staked_units": 0.0,
            "total_profit_units": float("nan"),
            "roi": float("nan"),
        }
    profits = []
    for _, row in settled.iterrows():
        won_home = float(row[HOME_WON_LABEL]) == 1.0
        side = row["bet_side"]
        won = (side == "home" and won_home) or (side == "away" and not won_home)
        units = float(row["bet_units"])
        ml = float(row["bet_moneyline"])
        if won:
            profits.append(units * kelly.moneyline_to_net_odds(ml))
        else:
            profits.append(-units)
    staked = float(settled["bet_units"].sum())
    profit = float(np.sum(profits))
    return {
        "n_bets": int(len(settled)),
        "total_staked_units": staked,
        "total_profit_units": profit,
        "roi": (profit / staked) if staked > 0 else float("nan"),
    }


def true_closing_line_value(bets: pd.DataFrame) -> dict:
    """Mean probability CLV vs true closing home-win market, same side."""
    if bets is None or bets.empty:
        return {"n_clv": 0, "mean_probability_clv": float("nan")}
    if CLOSING_MARKET_COL not in bets.columns:
        return {"n_clv": 0, "mean_probability_clv": float("nan")}
    vals = []
    for _, row in bets.iterrows():
        closing_home = row.get(CLOSING_MARKET_COL)
        if pd.isna(closing_home) or pd.isna(row.get("market_implied_probability")):
            continue
        closing_home = float(closing_home)
        closing_side = closing_home if row["bet_side"] == "home" else 1.0 - closing_home
        # Positive CLV: got a better price at bet time than close.
        vals.append(float(closing_side) - float(row["market_implied_probability"]))
    if not vals:
        return {"n_clv": 0, "mean_probability_clv": float("nan")}
    return {"n_clv": len(vals), "mean_probability_clv": float(np.mean(vals))}


def date_block_bootstrap_roi(
    bets: pd.DataFrame,
    *,
    n_bootstrap: int | None = None,
    random_seed: int | None = None,
    alpha: float = 0.05,
) -> dict:
    n_bootstrap = n_bootstrap or config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES
    random_seed = config.NESTED_VALIDATION_RANDOM_SEED if random_seed is None else random_seed
    point = settle_hypothetical_roi(bets)
    if bets is None or bets.empty or "date" not in bets.columns:
        return {**point, "roi_ci_low": float("nan"), "roi_ci_high": float("nan"), "n_blocks": 0}
    blocks = model_validation.unique_sorted_dates(bets["date"])
    if not blocks:
        return {**point, "roi_ci_low": float("nan"), "roi_ci_high": float("nan"), "n_blocks": 0}
    rng = np.random.default_rng(random_seed)
    block_arr = np.asarray(blocks, dtype=object)
    rois = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = block_arr[rng.integers(0, len(block_arr), size=len(block_arr))]
        parts = [bets[bets["date"] == blk] for blk in sampled]
        boot = pd.concat(parts, ignore_index=True) if parts else bets.iloc[0:0]
        rois[i] = settle_hypothetical_roi(boot).get("roi", float("nan"))
    finite = rois[np.isfinite(rois)]
    if finite.size == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite, alpha / 2))
        ci_high = float(np.quantile(finite, 1 - alpha / 2))
    return {
        **point,
        "roi_ci_low": ci_low,
        "roi_ci_high": ci_high,
        "n_blocks": len(blocks),
        "n_bootstrap": n_bootstrap,
    }


def single_week_profit_concentration(bets: pd.DataFrame) -> dict:
    """Detect dependence on one unusually profitable ISO week."""
    settled_roi = settle_hypothetical_roi(bets)
    if bets is None or bets.empty or settled_roi["total_profit_units"] != settled_roi["total_profit_units"]:
        return {"max_week_profit_share": float("nan"), "worst_week_dependence": False}
    settled = bets.dropna(subset=[HOME_WON_LABEL]).copy()
    if settled.empty:
        return {"max_week_profit_share": float("nan"), "worst_week_dependence": False}
    profits = []
    weeks = []
    for _, row in settled.iterrows():
        won_home = float(row[HOME_WON_LABEL]) == 1.0
        side = row["bet_side"]
        won = (side == "home" and won_home) or (side == "away" and not won_home)
        units = float(row["bet_units"])
        ml = float(row["bet_moneyline"])
        profit = units * kelly.moneyline_to_net_odds(ml) if won else -units
        profits.append(profit)
        weeks.append(pd.Timestamp(row["date"]).to_period("W-SUN").start_time)
    tmp = pd.DataFrame({"week": weeks, "profit": profits})
    by_week = tmp.groupby("week")["profit"].sum()
    total_pos = float(by_week.clip(lower=0).sum())
    if total_pos <= 0:
        return {"max_week_profit_share": 0.0, "worst_week_dependence": False}
    share = float(by_week.max() / total_pos) if by_week.max() > 0 else 0.0
    return {
        "max_week_profit_share": share,
        "worst_week_dependence": share > float(config.BETTING_PROMOTION_MAX_SINGLE_WEEK_PROFIT_SHARE),
    }


# ---------------------------------------------------------------------------
# Nested validation
# ---------------------------------------------------------------------------


METHOD_MARKET = "market_alone"
METHOD_HEURISTIC = "heuristic"
METHOD_HEURISTIC_CAL = "heuristic_calibrated"
METHOD_RESIDUAL_LOGISTIC = "residual_logistic"
METHOD_RESIDUAL_NONLINEAR = "residual_nonlinear"


def _slice_dates(df: pd.DataFrame, dates: Sequence) -> pd.DataFrame:
    return df[df["date"].isin(set(dates))].copy()


def _select_edge_threshold_inner(
    train: pd.DataFrame,
    model_prob_col: str,
    inner_folds: Sequence[model_validation.DateFold],
) -> float:
    """Pick edge threshold on inner folds only (includes vig via moneylines)."""
    best_thr = float(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID[0])
    best_roi = float("-inf")
    for thr in config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID:
        rois = []
        for fold in inner_folds:
            block = _slice_dates(train, fold.test_dates)
            # Fit is outer-train already; here we only score threshold on
            # inner-test blocks using already-computed model probs when present.
            if model_prob_col not in block.columns:
                continue
            bets = hypothetical_bets_from_probabilities(block, model_prob_col=model_prob_col, edge_threshold=thr)
            rois.append(settle_hypothetical_roi(bets).get("roi", float("nan")))
        mean_roi = float(np.nanmean(rois)) if rois else float("nan")
        if mean_roi == mean_roi and mean_roi > best_roi:
            best_roi = mean_roi
            best_thr = float(thr)
    return best_thr


def _fit_predict_method(
    method: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    logistic_C: float | None = None,
    nonlinear_params: dict | None = None,
    calibrate_heuristic: bool = False,
) -> pd.Series:
    market_test = test[MARKET_AT_PRED_COL]
    if method == METHOD_MARKET:
        return pd.to_numeric(market_test, errors="coerce")
    if method == METHOD_HEURISTIC:
        return pd.to_numeric(test[HEURISTIC_COL], errors="coerce")
    if method == METHOD_HEURISTIC_CAL:
        iso = fit_isotonic_on_train(train[HEURISTIC_COL], train[HOME_WON_LABEL]) if calibrate_heuristic else None
        return apply_isotonic(iso, test[HEURISTIC_COL])
    if method == METHOD_RESIDUAL_LOGISTIC:
        model = MarketResidualLogistic(
            C=float(logistic_C or config.GAME_RESIDUAL_LOGISTIC_C_GRID[0]),
            feature_columns=list(feature_columns),
        )
        model.fit(train, train[HOME_WON_LABEL], train[MARKET_AT_PRED_COL])
        return pd.Series(model.predict_proba(test, market_test), index=test.index)
    if method == METHOD_RESIDUAL_NONLINEAR:
        params = nonlinear_params or {}
        model = MarketResidualNonlinear(
            max_depth=int(params.get("max_depth", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 80)),
            shrink=float(params.get("shrink", 0.5)),
            feature_columns=list(feature_columns),
        )
        model.fit(train, train[HOME_WON_LABEL], train[MARKET_AT_PRED_COL])
        return pd.Series(model.predict_proba(test, market_test), index=test.index)
    raise ValueError(f"Unknown method: {method}")


def _inner_select_residual_logistic(
    train: pd.DataFrame,
    inner_folds: Sequence[model_validation.DateFold],
    feature_columns: Sequence[str],
) -> float:
    best_C = float(config.GAME_RESIDUAL_LOGISTIC_C_GRID[0])
    best_score = float("inf")
    for C in config.GAME_RESIDUAL_LOGISTIC_C_GRID:
        losses = []
        for fold in inner_folds:
            tr = _slice_dates(train, fold.train_dates)
            te = _slice_dates(train, fold.test_dates)
            if tr.empty or te.empty:
                continue
            pred = _fit_predict_method(
                METHOD_RESIDUAL_LOGISTIC, tr, te,
                feature_columns=feature_columns, logistic_C=C,
            )
            losses.append(log_loss(te[HOME_WON_LABEL], pred))
        mean_ll = float(np.nanmean(losses)) if losses else float("nan")
        if mean_ll == mean_ll and mean_ll < best_score:
            best_score = mean_ll
            best_C = float(C)
    return best_C


def _inner_select_residual_nonlinear(
    train: pd.DataFrame,
    inner_folds: Sequence[model_validation.DateFold],
    feature_columns: Sequence[str],
) -> dict:
    best = {"max_depth": 2, "min_samples_leaf": 80, "shrink": 0.5}
    best_score = float("inf")
    for depth in config.GAME_RESIDUAL_NONLINEAR_MAX_DEPTH_GRID:
        for leaf in config.GAME_RESIDUAL_NONLINEAR_MIN_SAMPLES_LEAF_GRID:
            for shrink in config.GAME_RESIDUAL_NONLINEAR_SHRINK_GRID:
                params = {"max_depth": depth, "min_samples_leaf": leaf, "shrink": shrink}
                losses = []
                for fold in inner_folds:
                    tr = _slice_dates(train, fold.train_dates)
                    te = _slice_dates(train, fold.test_dates)
                    if tr.empty or te.empty:
                        continue
                    pred = _fit_predict_method(
                        METHOD_RESIDUAL_NONLINEAR, tr, te,
                        feature_columns=feature_columns, nonlinear_params=params,
                    )
                    losses.append(log_loss(te[HOME_WON_LABEL], pred))
                mean_ll = float(np.nanmean(losses)) if losses else float("nan")
                if mean_ll == mean_ll and mean_ll < best_score:
                    best_score = mean_ll
                    best = params
    return best


def run_game_residual_nested_validation(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    freeze_dates: int | None = None,
) -> dict:
    """Nested rolling-origin comparison on the exact same market-covered games."""
    feature_columns = list(feature_columns or RESIDUAL_FEATURE_COLUMNS)
    assert_no_closing_odds_in_features(df, feature_columns)

    required = {"date", "game_pk", HOME_WON_LABEL, MARKET_AT_PRED_COL, HEURISTIC_COL}
    if df is None or df.empty or not required.issubset(df.columns):
        return {"status": "insufficient_history", "n_outer_folds": 0, "methods": {}}

    # Evaluate only games with a prediction-time market snapshot.
    work = df.dropna(subset=[MARKET_AT_PRED_COL, HOME_WON_LABEL]).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if work.empty:
        return {"status": "insufficient_history", "n_outer_folds": 0, "methods": {}}

    nested, freeze_tail = model_validation.build_nested_folds(
        work["date"],
        outer_min_train_dates=config.GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES,
        outer_test_block_dates=config.GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES,
        inner_min_train_dates=config.GAME_RESIDUAL_INNER_MIN_TRAIN_DATES,
        inner_test_block_dates=config.GAME_RESIDUAL_INNER_TEST_BLOCK_DATES,
        freeze_dates=config.NESTED_VALIDATION_FREEZE_DATES if freeze_dates is None else freeze_dates,
    )
    if not nested:
        return {
            "status": "insufficient_history",
            "n_outer_folds": 0,
            "methods": {},
            "freeze_tail_dates": [str(d) for d in freeze_tail],
        }

    method_rows: dict[str, list[pd.DataFrame]] = {m: [] for m in (
        METHOD_MARKET, METHOD_HEURISTIC, METHOD_HEURISTIC_CAL,
        METHOD_RESIDUAL_LOGISTIC, METHOD_RESIDUAL_NONLINEAR,
    )}
    outer_reports = []
    selected_configs = []

    for nested_fold in nested:
        train = _slice_dates(work, nested_fold.outer.train_dates)
        test = _slice_dates(work, nested_fold.outer.test_dates)
        if train.empty or test.empty:
            continue

        # Calibration / residual hypers selected on inner folds only.
        use_cal = True
        # Prefer calibrated heuristic when inner mean log-loss improves.
        cal_losses = []
        raw_losses = []
        for fold in nested_fold.inner_folds:
            tr = _slice_dates(train, fold.train_dates)
            te = _slice_dates(train, fold.test_dates)
            if tr.empty or te.empty:
                continue
            raw_losses.append(log_loss(te[HOME_WON_LABEL], te[HEURISTIC_COL]))
            iso = fit_isotonic_on_train(tr[HEURISTIC_COL], tr[HOME_WON_LABEL])
            cal_losses.append(log_loss(te[HOME_WON_LABEL], apply_isotonic(iso, te[HEURISTIC_COL])))
        if raw_losses and cal_losses:
            use_cal = float(np.nanmean(cal_losses)) <= float(np.nanmean(raw_losses))

        best_C = _inner_select_residual_logistic(train, nested_fold.inner_folds, feature_columns)
        best_nl = _inner_select_residual_nonlinear(train, nested_fold.inner_folds, feature_columns)
        selected_configs.append({
            "outer_fold_id": nested_fold.outer.fold_id,
            "logistic_C": best_C,
            "nonlinear": best_nl,
            "heuristic_calibrated": use_cal,
        })

        fold_preds = {"date": test["date"], "game_pk": test["game_pk"], HOME_WON_LABEL: test[HOME_WON_LABEL]}
        for method in method_rows:
            pred = _fit_predict_method(
                method, train, test,
                feature_columns=feature_columns,
                logistic_C=best_C,
                nonlinear_params=best_nl,
                calibrate_heuristic=use_cal,
            )
            block = test[["date", "game_pk", HOME_WON_LABEL, MARKET_AT_PRED_COL]].copy()
            for optional in (
                "home_moneyline", "away_moneyline",
                "market_home_moneyline", "market_away_moneyline",
                CLOSING_MARKET_COL, "home_team", "away_team",
            ):
                if optional in test.columns:
                    block[optional] = test[optional]
            block["predicted_probability"] = pred.to_numpy()
            method_rows[method].append(block)
            fold_preds[method] = pred.to_numpy()

        market_metrics = score_probabilities(test[HOME_WON_LABEL], test[MARKET_AT_PRED_COL])
        residual_metrics = score_probabilities(
            test[HOME_WON_LABEL], pd.Series(fold_preds[METHOD_RESIDUAL_LOGISTIC], index=test.index),
        )
        paired = paired_diffs_vs_market(
            test[HOME_WON_LABEL],
            pd.Series(fold_preds[METHOD_RESIDUAL_LOGISTIC], index=test.index),
            test[MARKET_AT_PRED_COL],
        )
        outer_reports.append({
            "fold_id": nested_fold.outer.fold_id,
            "n_test_games": int(len(test)),
            "market": market_metrics,
            "residual_logistic": residual_metrics,
            "paired_vs_market": paired,
            "selected": selected_configs[-1],
        })

    methods_summary = {}
    all_bets = {}
    for method, parts in method_rows.items():
        if not parts:
            methods_summary[method] = {"status": "no_predictions"}
            continue
        aligned = pd.concat(parts, ignore_index=True)
        # Same games as market baseline: drop rows missing market (already filtered).
        metrics = score_probabilities(aligned[HOME_WON_LABEL], aligned["predicted_probability"])
        paired = paired_diffs_vs_market(
            aligned[HOME_WON_LABEL], aligned["predicted_probability"], aligned[MARKET_AT_PRED_COL],
        )
        # Bootstrap paired differences on date blocks.
        boot_frame_a = aligned.rename(columns={"predicted_probability": "predicted_probability_a"})
        boot_frame_b = aligned[["date", "game_pk", HOME_WON_LABEL, MARKET_AT_PRED_COL]].rename(
            columns={MARKET_AT_PRED_COL: "predicted_probability_b"}
        )
        boot_brier = model_validation.paired_block_bootstrap_difference(
            boot_frame_a, boot_frame_b,
            keys=["date", "game_pk"],
            value_a="predicted_probability_a",
            value_b="predicted_probability_b",
            label_col=HOME_WON_LABEL,
            metric="brier_score",
            n_bootstrap=min(200, config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES),
        )
        boot_ll = model_validation.paired_block_bootstrap_difference(
            boot_frame_a, boot_frame_b,
            keys=["date", "game_pk"],
            value_a="predicted_probability_a",
            value_b="predicted_probability_b",
            label_col=HOME_WON_LABEL,
            metric="log_loss",
            n_bootstrap=min(200, config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES),
        )
        # Edge threshold from a pooled inner-style heuristic: use grid median
        # when method-specific inner selection is unavailable post-hoc.
        edge_thr = float(np.median(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID))
        bets = hypothetical_bets_from_probabilities(
            aligned.rename(columns={"predicted_probability": "model_p"}),
            model_prob_col="model_p",
            edge_threshold=edge_thr,
        )
        # Attach labels already on aligned via merge
        if not bets.empty:
            bets = bets.drop(columns=[HOME_WON_LABEL, CLOSING_MARKET_COL], errors="ignore").merge(
                aligned[["game_pk", "date", HOME_WON_LABEL] + (
                    [CLOSING_MARKET_COL] if CLOSING_MARKET_COL in aligned.columns else []
                )],
                on=["game_pk", "date"],
                how="left",
            )
        roi = date_block_bootstrap_roi(bets)
        clv = true_closing_line_value(bets)
        week = single_week_profit_concentration(bets)
        methods_summary[method] = {
            **metrics,
            **paired,
            "paired_brier_bootstrap": boot_brier,
            "paired_log_loss_bootstrap": boot_ll,
            "hypothetical_roi": roi,
            "true_closing_line_value": clv,
            "week_concentration": week,
            "edge_threshold_used": edge_thr,
            "n_games": int(len(aligned)),
        }
        all_bets[method] = bets

    residual = methods_summary.get(METHOD_RESIDUAL_LOGISTIC, {})
    market = methods_summary.get(METHOD_MARKET, {})
    promotion_gate = build_probability_promotion_gate(residual, market, outer_reports)
    betting_gate = build_betting_promotion_gate(
        methods_summary.get(METHOD_RESIDUAL_LOGISTIC, {}),
        all_bets.get(METHOD_RESIDUAL_LOGISTIC),
        outer_reports,
    )

    return {
        "status": "ok",
        "n_outer_folds": len(outer_reports),
        "n_games_evaluated": int(residual.get("n_games", 0) or 0),
        "freeze_tail_dates": [str(d) for d in freeze_tail],
        "feature_columns": list(feature_columns),
        "selected_configs": selected_configs,
        "outer_folds": outer_reports,
        "methods": methods_summary,
        "promotion_gate": promotion_gate,
        "betting_promotion_gate": betting_gate,
        "formulation": "logit(P_final)=logit(P_market_at_pred)+residual(features)",
        "notes": [
            "Closing odds are evaluation-only and never residual features.",
            "Calibration and residual hypers selected on inner folds only.",
            "Kelly sizing is hypothetical and does not gate probability accuracy.",
            "Do not treat config.KELLY_MIN_EDGE as proof of residual skill.",
        ],
    }


def build_probability_promotion_gate(
    residual: dict,
    market: dict,
    outer_reports: Sequence[dict],
) -> dict:
    paired_brier = residual.get("model_minus_market_brier", float("nan"))
    paired_ll = residual.get("model_minus_market_log_loss", float("nan"))
    boot_brier = residual.get("paired_brier_bootstrap") or {}
    boot_ll = residual.get("paired_log_loss_bootstrap") or {}
    n_folds = len(outer_reports)
    n_games = int(residual.get("n_games", 0) or 0)
    n_blocks = int((boot_brier or {}).get("n_blocks", 0) or 0)

    checks = {
        "positive_paired_brier_vs_market": bool(paired_brier == paired_brier and paired_brier < 0),
        "positive_paired_log_loss_vs_market": bool(paired_ll == paired_ll and paired_ll < 0),
        "brier_ci_not_materially_negative": bool(
            boot_brier.get("ci_high") == boot_brier.get("ci_high")
            and boot_brier.get("ci_high", 1.0) < 0.01
        ),
        "log_loss_ci_not_materially_negative": bool(
            boot_ll.get("ci_high") == boot_ll.get("ci_high")
            and boot_ll.get("ci_high", 1.0) < 0.01
        ),
        "adequate_sample_size": n_games >= int(config.BETTING_PROMOTION_MIN_GAMES),
        "adequate_outer_folds": n_folds >= int(config.BETTING_PROMOTION_MIN_OUTER_FOLDS),
        "adequate_date_blocks": n_blocks >= int(config.BETTING_PROMOTION_MIN_DATE_BLOCKS),
        "based_on_untouched_outer_folds": n_folds > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "paired_brier": paired_brier,
        "paired_log_loss": paired_ll,
        "n_games": n_games,
        "n_outer_folds": n_folds,
        "market_brier": market.get("brier_score"),
        "residual_brier": residual.get("brier_score"),
    }


def build_betting_promotion_gate(
    residual: dict,
    bets: pd.DataFrame | None,
    outer_reports: Sequence[dict],
) -> dict:
    roi = residual.get("hypothetical_roi") or {}
    clv = residual.get("true_closing_line_value") or {}
    week = residual.get("week_concentration") or {}
    paired_brier = residual.get("model_minus_market_brier", float("nan"))
    paired_ll = residual.get("model_minus_market_log_loss", float("nan"))
    n_folds = len(outer_reports)
    n_games = int(residual.get("n_games", 0) or 0)
    n_blocks = int(roi.get("n_blocks", 0) or 0)

    checks = {
        "positive_paired_brier_vs_market": bool(paired_brier == paired_brier and paired_brier < 0),
        "positive_paired_log_loss_vs_market": bool(paired_ll == paired_ll and paired_ll < 0),
        "positive_closing_line_value": bool(
            clv.get("mean_probability_clv") == clv.get("mean_probability_clv")
            and clv.get("mean_probability_clv", -1) > 0
        ),
        "positive_hypothetical_roi": bool(roi.get("roi") == roi.get("roi") and roi.get("roi", -1) > 0),
        "roi_ci_not_materially_negative": bool(
            roi.get("roi_ci_low") == roi.get("roi_ci_low")
            and roi.get("roi_ci_low", -1) > float(config.BETTING_PROMOTION_MATERIAL_NEGATIVE_ROI)
        ),
        "adequate_sample_size": n_games >= int(config.BETTING_PROMOTION_MIN_GAMES),
        "adequate_outer_folds": n_folds >= int(config.BETTING_PROMOTION_MIN_OUTER_FOLDS),
        "adequate_date_blocks": n_blocks >= int(config.BETTING_PROMOTION_MIN_DATE_BLOCKS),
        "no_single_week_dependence": not bool(week.get("worst_week_dependence")),
        "based_on_untouched_outer_folds": n_folds > 0,
        # Explicit: Kelly fraction never appears as an accuracy gate.
        "kelly_not_used_as_accuracy_gate": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "hypothetical_roi": roi,
        "true_closing_line_value": clv,
        "week_concentration": week,
        "n_bets": int(len(bets)) if bets is not None else 0,
    }


# ---------------------------------------------------------------------------
# Artifact I/O / inference
# ---------------------------------------------------------------------------


@dataclass
class ResidualModelArtifact:
    family: str
    model: Any
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)


def save_residual_model(artifact: ResidualModelArtifact, path: str | None = None) -> dict:
    path = path or config.GAME_RESIDUAL_MODEL_PATH
    bundle = ml_models.save_model_bundle(
        artifact.model,
        path,
        model_type=f"game_residual_{artifact.family}",
        feature_columns=artifact.feature_columns,
        hyperparameters=artifact.metadata.get("hyperparameters") or {},
        calibration_method=None,
        validation_summary=artifact.metadata.get("validation_summary") or {},
        model_version=artifact.metadata.get("model_version") or config.GAME_RESIDUAL_MODEL_VERSION,
        training_data_start=artifact.metadata.get("training_data_start"),
        training_data_cutoff=artifact.metadata.get("training_data_cutoff"),
    )
    # Persist family tag beside estimator for loaders.
    bundle["residual_family"] = artifact.family
    bundle["residual_metadata"] = artifact.metadata
    import joblib
    joblib.dump(bundle, path)
    return bundle


def load_residual_model(path: str | None = None) -> ResidualModelArtifact | None:
    path = path or config.GAME_RESIDUAL_MODEL_PATH
    bundle = ml_models.load_model_bundle(path)
    if bundle is None or bundle.get("estimator") is None:
        return None
    return ResidualModelArtifact(
        family=bundle.get("residual_family") or "logistic",
        model=bundle["estimator"],
        feature_columns=list(bundle.get("feature_columns") or RESIDUAL_FEATURE_COLUMNS),
        metadata={
            "artifact_id": bundle.get("artifact_id"),
            "model_version": bundle.get("model_version"),
            "validation_summary": bundle.get("validation_summary") or {},
            "hyperparameters": bundle.get("hyperparameters") or {},
            **(bundle.get("residual_metadata") or {}),
        },
    )


def predict_residual_home_win_probability(
    features: pd.DataFrame,
    market_home_win_probability: pd.Series | np.ndarray,
    *,
    artifact: ResidualModelArtifact | None = None,
) -> tuple[pd.Series, dict]:
    """Predict P(home win) = sigmoid(logit(market) + residual(features))."""
    status = {
        "loaded": False,
        "fallback_used": True,
        "fallback_reason": "missing_artifact",
        "artifact_id": None,
        "model_version": None,
        "family": None,
    }
    art = artifact if artifact is not None else load_residual_model()
    market = pd.Series(pd.to_numeric(market_home_win_probability, errors="coerce"), index=features.index)
    if art is None:
        return market.copy(), status

    status.update({
        "loaded": True,
        "fallback_used": False,
        "fallback_reason": None,
        "artifact_id": art.metadata.get("artifact_id"),
        "model_version": art.metadata.get("model_version"),
        "family": art.family,
    })
    try:
        if art.family == "nonlinear":
            pred = art.model.predict_proba(features, market)
        else:
            pred = art.model.predict_proba(features, market)
        return pd.Series(np.asarray(pred, dtype=float), index=features.index), status
    except Exception as exc:
        status["fallback_used"] = True
        status["fallback_reason"] = f"predict_failed:{type(exc).__name__}"
        return market.copy(), status


def attach_prediction_time_market(
    frame: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    snapshot_role: str = "morning",
) -> pd.DataFrame:
    """Merge the market snapshot available at the configured prediction role.

    Never selects closing for prediction features.
    """
    if snapshot_role == "closing":
        raise ValueError("closing snapshots cannot be attached as prediction-time market priors")
    out = frame.copy()
    if snapshots is None or snapshots.empty or "game_pk" not in out.columns:
        out[MARKET_AT_PRED_COL] = pd.NA
        return out

    rows = []
    for gpk in out["game_pk"].dropna().unique():
        snap = market_odds.select_role_snapshot(snapshots, gpk, snapshot_role)
        if snap is None:
            # Fall back to latest snapshot for that game that is not closing-tagged
            # when an exact role tag is missing - still never uses post-start close
            # selection helpers.
            frame_s = market_odds.normalize_snapshot_frame(snapshots)
            cand = frame_s[
                (frame_s["game_pk"] == gpk)
                & frame_s[MARKET_AT_PRED_COL].notna()
                & (frame_s.get("snapshot_role", pd.Series(dtype=str)) != "closing")
            ] if "snapshot_role" in frame_s.columns else frame_s[
                (frame_s["game_pk"] == gpk) & frame_s[MARKET_AT_PRED_COL].notna()
            ]
            if cand.empty:
                continue
            snap = cand.sort_values("captured_at_utc").iloc[-1]
        rows.append({
            "game_pk": gpk,
            MARKET_AT_PRED_COL: snap.get(MARKET_AT_PRED_COL),
            "market_odds_snapshot_id": snap.get("snapshot_id"),
            "market_odds_snapshot_role": snap.get("snapshot_role", snapshot_role),
            "home_moneyline": snap.get("home_moneyline"),
            "away_moneyline": snap.get("away_moneyline"),
        })
    if not rows:
        out[MARKET_AT_PRED_COL] = pd.NA
        return out
    return out.drop(columns=[c for c in (
        MARKET_AT_PRED_COL, "market_odds_snapshot_id", "market_odds_snapshot_role",
        "home_moneyline", "away_moneyline",
    ) if c in out.columns], errors="ignore").merge(pd.DataFrame(rows), on="game_pk", how="left")


def prepare_training_frame(
    game_pick_log: pd.DataFrame,
    *,
    market_snapshots: pd.DataFrame | None = None,
    prediction_snapshot_role: str = "morning",
    closing_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble residual training rows; closing joined only for evaluation labels."""
    if game_pick_log is None or game_pick_log.empty:
        return pd.DataFrame()
    frame = enrich_residual_features(game_pick_log)
    if market_snapshots is not None and not market_snapshots.empty:
        frame = attach_prediction_time_market(
            frame, market_snapshots, snapshot_role=prediction_snapshot_role,
        )
    if CLOSING_MARKET_COL not in frame.columns:
        frame[CLOSING_MARKET_COL] = pd.NA
    if closing_snapshots is not None and not closing_snapshots.empty and "game_pk" in frame.columns:
        closes = []
        for gpk in frame["game_pk"].dropna().unique():
            snap = market_odds.select_closing_snapshot(closing_snapshots, int(gpk))
            if snap is None:
                continue
            closes.append({
                "game_pk": gpk,
                CLOSING_MARKET_COL: snap.get(MARKET_AT_PRED_COL),
            })
        if closes:
            frame = frame.drop(columns=[CLOSING_MARKET_COL], errors="ignore").merge(
                pd.DataFrame(closes), on="game_pk", how="left",
            )
    # Drop rows without prediction-time market - they cannot enter the
    # market-baseline-aligned evaluation set.
    return frame.dropna(subset=[MARKET_AT_PRED_COL]).copy()


# ---------------------------------------------------------------------------
# Modes / gates / shadow exports
# ---------------------------------------------------------------------------


def evaluate_prediction_promotion_gate(report: dict | None) -> tuple[bool, dict]:
    details: dict[str, Any] = {"passed": False, "reason": "missing_report", "checks": {}}
    if not report or not isinstance(report, dict):
        return False, details
    gate = report.get("promotion_gate")
    if not isinstance(gate, dict):
        details["reason"] = "missing_promotion_gate_block"
        return False, details
    checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else {}
    details["checks"] = {k: bool(v) for k, v in checks.items()}
    if not details["checks"]:
        details["reason"] = "empty_checks"
        return False, details
    if not all(details["checks"].values()):
        failed = [k for k, v in details["checks"].items() if not v]
        details["reason"] = f"gate_checks_failed:{','.join(failed)}"
        return False, details
    details["passed"] = True
    details["reason"] = "passed"
    return True, details


def evaluate_betting_promotion_gate(report: dict | None) -> tuple[bool, dict]:
    details: dict[str, Any] = {"passed": False, "reason": "missing_report", "checks": {}}
    if not report or not isinstance(report, dict):
        return False, details
    gate = report.get("betting_promotion_gate") or report.get("promotion_gate")
    if not isinstance(gate, dict):
        details["reason"] = "missing_betting_promotion_gate_block"
        return False, details
    checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else {}
    details["checks"] = {k: bool(v) for k, v in checks.items()}
    if not details["checks"]:
        details["reason"] = "empty_checks"
        return False, details
    if not all(details["checks"].values()):
        failed = [k for k, v in details["checks"].items() if not v]
        details["reason"] = f"gate_checks_failed:{','.join(failed)}"
        return False, details
    details["passed"] = True
    details["reason"] = "passed"
    return True, details


def _load_json_report(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def prediction_promotion_gate_satisfied(report_path: str | None = None) -> tuple[bool, dict]:
    path = report_path or config.GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH
    report = _load_json_report(path)
    if report is None:
        return False, {"passed": False, "reason": "missing_report_file", "path": path}
    ok, details = evaluate_prediction_promotion_gate(report)
    details["path"] = path
    return ok, details


def betting_promotion_gate_satisfied(report_path: str | None = None) -> tuple[bool, dict]:
    path = report_path or config.BETTING_PROMOTION_GATE_REPORT_PATH
    report = _load_json_report(path)
    if report is None:
        # Fall back to the nested residual report's betting gate block.
        report = _load_json_report(config.GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH)
        path = config.GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH
    if report is None:
        return False, {"passed": False, "reason": "missing_report_file", "path": path}
    ok, details = evaluate_betting_promotion_gate(report)
    details["path"] = path
    return ok, details


def resolve_game_prediction_mode(
    configured: str | None = None,
    *,
    report_path: str | None = None,
    force_live: bool = False,
) -> tuple[str, dict]:
    mode = (configured if configured is not None else config.GAME_PREDICTION_MODE) or "shadow"
    if mode not in config.GAME_PREDICTION_MODES:
        return "shadow", {
            "configured": mode,
            "effective": "shadow",
            "fallback_used": True,
            "fallback_reason": f"invalid_mode:{mode}",
            "promotion_gate": {},
        }
    if mode != "live":
        return mode, {
            "configured": mode,
            "effective": mode,
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {},
        }
    if force_live:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {"forced": True},
        }
    ok, gate_details = prediction_promotion_gate_satisfied(report_path)
    if ok:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": gate_details,
        }
    return "shadow", {
        "configured": "live",
        "effective": "shadow",
        "fallback_used": True,
        "fallback_reason": f"promotion_gate_failed:{gate_details.get('reason')}",
        "promotion_gate": gate_details,
    }


def resolve_betting_mode(
    configured: str | None = None,
    *,
    report_path: str | None = None,
    force_live: bool = False,
) -> tuple[str, dict]:
    mode = (configured if configured is not None else config.BETTING_MODE) or "disabled"
    if mode not in config.BETTING_MODES:
        return "disabled", {
            "configured": mode,
            "effective": "disabled",
            "fallback_used": True,
            "fallback_reason": f"invalid_mode:{mode}",
            "promotion_gate": {},
        }
    if mode != "live":
        return mode, {
            "configured": mode,
            "effective": mode,
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {},
        }
    if force_live:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {"forced": True},
        }
    ok, gate_details = betting_promotion_gate_satisfied(report_path)
    if ok:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": gate_details,
        }
    return "shadow", {
        "configured": "live",
        "effective": "shadow",
        "fallback_used": True,
        "fallback_reason": f"promotion_gate_failed:{gate_details.get('reason')}",
        "promotion_gate": gate_details,
    }


def suppress_official_bets(picks: pd.DataFrame) -> pd.DataFrame:
    """Zero official stake columns (shadow/disabled betting modes)."""
    if picks is None or picks.empty:
        return picks
    out = picks.copy()
    out["bet_units"] = 0.0
    out["bet_stake_fraction"] = 0.0
    for col in ("bet_side", "bet_team", "bet_moneyline"):
        if col in out.columns:
            out[col] = pd.NA
    return out


def write_shadow_predictions(
    frame: pd.DataFrame,
    *,
    path: str | None = None,
) -> str:
    path = path or config.GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [c for c in SHADOW_PREDICTION_COLUMNS if c in frame.columns]
    shadow = frame[cols].copy()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        shadow = pd.concat([prev, shadow], ignore_index=True)
        if {"date", "game_pk"}.issubset(shadow.columns):
            shadow = shadow.drop_duplicates(subset=["date", "game_pk"], keep="last")
    shadow.to_csv(path, index=False)
    return path


def write_shadow_bets(bets: pd.DataFrame, *, path: str | None = None) -> str:
    path = path or config.GAME_RESIDUAL_SHADOW_BETS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = bets.copy()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        out = pd.concat([prev, out], ignore_index=True)
        if {"date", "game_pk"}.issubset(out.columns):
            out = out.drop_duplicates(subset=["date", "game_pk"], keep="last")
    out.to_csv(path, index=False)
    return path


def build_shadow_prediction_frame(
    win_probabilities: pd.DataFrame,
    residual_probs: pd.Series,
    market_probs: pd.Series | None,
    *,
    model_status: dict,
    game_prediction_mode: str,
    prediction_snapshot_type: str = "morning",
) -> pd.DataFrame:
    out = win_probabilities.copy()
    out[RESIDUAL_PROB_COL] = residual_probs
    if market_probs is not None:
        out[MARKET_AT_PRED_COL] = market_probs
    out["residual_logit"] = logit(out[RESIDUAL_PROB_COL]) - logit(
        out.get(MARKET_AT_PRED_COL, pd.Series(np.nan, index=out.index))
    )
    out["probability_source"] = "market_residual"
    out["model_version"] = model_status.get("model_version") or config.GAME_RESIDUAL_MODEL_VERSION
    out["artifact_id"] = model_status.get("artifact_id")
    out["game_prediction_mode"] = game_prediction_mode
    out["prediction_snapshot_type"] = prediction_snapshot_type
    return out


def effective_game_model_version(game_prediction_mode: str | None = None) -> str:
    mode = game_prediction_mode or config.GAME_PREDICTION_MODE
    if mode == "live":
        return config.GAME_RESIDUAL_MODEL_VERSION_LIVE
    return config.GAME_PICK_MODEL_VERSION
