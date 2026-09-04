"""Game-pick backtest scoring - the game-level analog of evaluation.py's
Beat the Streak Tracker export (build_beat_the_streak_export). See
game_picks.py/game_predictions.py.

Market comparison uses a real timestamped closing snapshot (latest valid
pregame price) when available. Primary skill metrics are paired scoring-rule
differences (model − market Brier / log loss) with date-block bootstrap CIs;
``beat_closing_line_rate`` is retained only as a secondary win-rate style
readout and does not treat a 0.0001 edge the same as a 0.20 edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_metrics import config, evaluation, market_odds


def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    """Per-pick outcome: "pending", "not_played", "win", or "loss"."""
    game_played = pd.to_numeric(df["game_played"], errors="coerce")

    outcome = pd.Series("pending", index=df.index)
    outcome[game_played == 0] = "not_played"
    played = game_played == 1
    correct = df["predicted_winner"] == df["actual_winner"]
    outcome[played & correct] = "win"
    outcome[played & ~correct] = "loss"
    return outcome


def _model_home_probability(frame: pd.DataFrame) -> pd.Series:
    model_favors_home = frame["predicted_winner"] == frame["home_team"]
    return frame["predicted_probability"].where(
        model_favors_home, 1 - frame["predicted_probability"]
    )


def _resolve_closing_market_probability(
    picks: pd.DataFrame,
    odds_snapshots: pd.DataFrame | None = None,
) -> pd.Series:
    """Per-row closing de-vigged home probability.

    Prefers the latest valid pregame snapshot from ``odds_snapshots``.
    Falls back to the prediction's own ``market_home_win_probability``
    (legacy morning-only logs) when no closing snapshot exists.
    """
    out = pd.Series(pd.NA, index=picks.index, dtype="Float64")
    if odds_snapshots is not None and not odds_snapshots.empty and "game_pk" in picks.columns:
        snaps = market_odds.normalize_snapshot_frame(odds_snapshots)
        for idx, row in picks.iterrows():
            gpk = row.get("game_pk")
            if pd.isna(gpk):
                continue
            closing = market_odds.select_closing_snapshot(
                snaps, int(gpk), game_datetime=row.get("game_datetime"),
            )
            if closing is not None and pd.notna(closing.get("market_home_win_probability")):
                out.at[idx] = float(closing["market_home_win_probability"])
    if "market_home_win_probability" in picks.columns:
        legacy = pd.to_numeric(picks["market_home_win_probability"], errors="coerce")
        out = out.fillna(legacy)
    return out


def paired_market_scoring_differences(
    picks: pd.DataFrame,
    *,
    odds_snapshots: pd.DataFrame | None = None,
    n_bootstrap: int | None = None,
    random_seed: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """Paired model-vs-closing-market scoring on the same resolved games."""
    n_bootstrap = int(
        config.MARKET_ODDS_BOOTSTRAP_SAMPLES if n_bootstrap is None else n_bootstrap
    )
    random_seed = int(
        config.MARKET_ODDS_BOOTSTRAP_SEED if random_seed is None else random_seed
    )

    empty = {
        "n_compared": 0,
        "brier_diff": float("nan"),
        "log_loss_diff": float("nan"),
        "mean_prob_diff": float("nan"),
        "mean_squared_error_diff": float("nan"),
        "pct_model_error_lower": float("nan"),
        "brier_diff_ci_low": float("nan"),
        "brier_diff_ci_high": float("nan"),
        "log_loss_diff_ci_low": float("nan"),
        "log_loss_diff_ci_high": float("nan"),
        "n_bootstrap": n_bootstrap,
        "n_date_blocks": 0,
        "beat_closing_line_rate": float("nan"),
        "n_beat_closing_line_compared": 0,
    }
    if picks is None or picks.empty:
        return empty

    frame = picks.copy()
    frame["closing_market_home_probability"] = _resolve_closing_market_probability(
        frame, odds_snapshots=odds_snapshots,
    )
    scoped = frame[
        frame["closing_market_home_probability"].notna()
        & frame["actual_winner"].notna()
        & frame["predicted_probability"].notna()
    ].copy()
    if scoped.empty:
        return empty

    y = (scoped["actual_winner"] == scoped["home_team"]).astype(float)
    model_p = _model_home_probability(scoped).astype(float)
    market_p = pd.to_numeric(scoped["closing_market_home_probability"], errors="coerce").astype(float)
    valid = model_p.notna() & market_p.notna() & y.notna()
    scoped = scoped.loc[valid].copy()
    y, model_p, market_p = y.loc[valid], model_p.loc[valid], market_p.loc[valid]
    if scoped.empty:
        return empty

    model_se = (model_p - y) ** 2
    market_se = (market_p - y) ** 2
    eps = 1e-6
    model_ll = -(
        y * np.log(model_p.clip(eps, 1 - eps))
        + (1 - y) * np.log((1 - model_p).clip(eps, 1 - eps))
    )
    market_ll = -(
        y * np.log(market_p.clip(eps, 1 - eps))
        + (1 - y) * np.log((1 - market_p).clip(eps, 1 - eps))
    )

    brier_diff = float(model_se.mean() - market_se.mean())
    log_loss_diff = float(model_ll.mean() - market_ll.mean())
    mean_prob_diff = float((model_p - market_p).mean())
    mean_se_diff = float((model_se - market_se).mean())

    compared = model_se != market_se
    n_tie_excluded = int(compared.sum())
    pct_lower = float((model_se < market_se).sum() / n_tie_excluded) if n_tie_excluded else float("nan")

    scoped = scoped.copy()
    scoped["_model_se"] = model_se.to_numpy()
    scoped["_market_se"] = market_se.to_numpy()
    scoped["_model_ll"] = model_ll.to_numpy()
    scoped["_market_ll"] = market_ll.to_numpy()
    scoped["date"] = pd.to_datetime(scoped["date"])
    blocks = scoped["date"].drop_duplicates().sort_values().to_numpy()
    n_blocks = len(blocks)
    rng = np.random.default_rng(random_seed)
    brier_boots = np.empty(n_bootstrap, dtype=float)
    ll_boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        if n_blocks == 0:
            brier_boots[i] = ll_boots[i] = float("nan")
            continue
        sampled = blocks[rng.integers(0, n_blocks, size=n_blocks)]
        parts = [scoped[scoped["date"] == blk] for blk in sampled]
        boot = pd.concat(parts, ignore_index=True) if parts else scoped.iloc[0:0]
        if boot.empty:
            brier_boots[i] = ll_boots[i] = float("nan")
        else:
            brier_boots[i] = float(boot["_model_se"].mean() - boot["_market_se"].mean())
            ll_boots[i] = float(boot["_model_ll"].mean() - boot["_market_ll"].mean())

    def _ci(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2))

    brier_lo, brier_hi = _ci(brier_boots)
    ll_lo, ll_hi = _ci(ll_boots)

    return {
        "n_compared": int(len(scoped)),
        "brier_diff": brier_diff,
        "log_loss_diff": log_loss_diff,
        "mean_prob_diff": mean_prob_diff,
        "mean_squared_error_diff": mean_se_diff,
        "pct_model_error_lower": pct_lower,
        "brier_diff_ci_low": brier_lo,
        "brier_diff_ci_high": brier_hi,
        "log_loss_diff_ci_low": ll_lo,
        "log_loss_diff_ci_high": ll_hi,
        "n_bootstrap": n_bootstrap,
        "n_date_blocks": n_blocks,
        "beat_closing_line_rate": pct_lower,
        "n_beat_closing_line_compared": n_tie_excluded,
    }


def closing_line_value_table(
    picks: pd.DataFrame,
    odds_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per advised-bet CLV vs the true closing snapshot for the same side."""
    cols = [
        "date", "game_pk", "bet_side", "bet_team", "bet_moneyline",
        "bet_implied_probability", "closing_implied_probability",
        "probability_clv", "moneyline_clv", "closing_moneyline",
        "market_odds_snapshot_id", "closing_snapshot_id",
    ]
    if picks is None or picks.empty or "bet_units" not in picks.columns:
        return pd.DataFrame(columns=cols)

    advised = picks[pd.to_numeric(picks["bet_units"], errors="coerce").fillna(0) > 0].copy()
    if advised.empty:
        return pd.DataFrame(columns=cols)

    snaps = (
        market_odds.normalize_snapshot_frame(odds_snapshots)
        if odds_snapshots is not None
        else market_odds.empty_snapshot_frame()
    )
    rows = []
    for _, row in advised.iterrows():
        side = row.get("bet_side")
        if side not in ("home", "away"):
            continue
        bet_ml = row.get("bet_moneyline")
        if pd.isna(bet_ml):
            continue
        bet_implied = market_odds.moneyline_to_implied_probability(float(bet_ml))

        closing_ml = pd.NA
        closing_implied = pd.NA
        closing_id = pd.NA
        gpk = row.get("game_pk")
        if pd.notna(gpk) and not snaps.empty:
            closing = market_odds.select_closing_snapshot(
                snaps, int(gpk), game_datetime=row.get("game_datetime"),
            )
            if closing is not None:
                closing_id = closing.get("snapshot_id")
                closing_ml = closing.get("home_moneyline") if side == "home" else closing.get("away_moneyline")
                if pd.notna(closing_ml):
                    closing_implied = market_odds.moneyline_to_implied_probability(float(closing_ml))

        rows.append({
            "date": row.get("date"),
            "game_pk": gpk,
            "bet_side": side,
            "bet_team": row.get("bet_team"),
            "bet_moneyline": bet_ml,
            "bet_implied_probability": bet_implied,
            "closing_implied_probability": closing_implied,
            "probability_clv": (
                market_odds.probability_clv(bet_implied, closing_implied)
                if pd.notna(closing_implied) else pd.NA
            ),
            "moneyline_clv": (
                market_odds.moneyline_clv(float(bet_ml), float(closing_ml))
                if pd.notna(closing_ml) else pd.NA
            ),
            "closing_moneyline": closing_ml,
            "market_odds_snapshot_id": row.get("market_odds_snapshot_id"),
            "closing_snapshot_id": closing_id,
        })
    return pd.DataFrame(rows, columns=cols)


def build_game_picks_export(
    predictions: pd.DataFrame,
    metric: str = "GamePick_Win_Probability",
    min_probability: float = config.GAME_PICK_MIN_PROBABILITY,
    model_version: str | None = None,
    odds_snapshots: pd.DataFrame | None = None,
):
    """Build dashboard tables; primary market skill = paired score diffs vs closing."""
    picks = predictions[predictions["metric"] == metric].copy()
    if model_version is not None:
        picks = picks[picks["model_version"] == model_version] if "model_version" in picks.columns else picks.iloc[0:0]
    if "above_threshold" not in picks.columns:
        picks["above_threshold"] = picks["predicted_probability"] >= min_probability
    if "market_home_win_probability" not in picks.columns:
        picks["market_home_win_probability"] = pd.NA
    if "bet_units" not in picks.columns:
        picks["bet_units"] = 0.0
        for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units"):
            picks[col] = pd.NA
    if "bet_profit_units" not in picks.columns:
        picks["bet_profit_units"] = pd.NA
    for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction"):
        if col not in picks.columns:
            picks[col] = pd.NA
    picks["status"] = _classify_outcome(picks)
    picks["actual_correct"] = pd.NA
    picks.loc[picks["status"] == "win", "actual_correct"] = 1.0
    picks.loc[picks["status"] == "loss", "actual_correct"] = 0.0

    recommended = picks[picks["above_threshold"]]
    (
        market_accuracy, market_brier, market_ll, n_market_resolved,
        market_accuracy_ci_low, market_accuracy_ci_high,
    ) = _market_comparison_metrics(recommended, odds_snapshots=odds_snapshots)
    paired = paired_market_scoring_differences(recommended, odds_snapshots=odds_snapshots)
    beat_closing_line_rate = paired["pct_model_error_lower"]
    n_beat_closing_line_compared = paired["n_beat_closing_line_compared"]
    if n_beat_closing_line_compared > 0 and beat_closing_line_rate == beat_closing_line_rate:
        n_beat = int(round(beat_closing_line_rate * n_beat_closing_line_compared))
        beat_ci_low, beat_ci_high = evaluation.wilson_confidence_interval(
            n_beat, n_beat_closing_line_compared,
        )
        beat_p = evaluation.binomial_significance(
            n_beat, n_beat_closing_line_compared, null_probability=0.5,
        )
    else:
        beat_ci_low = beat_ci_high = beat_p = float("nan")

    (
        n_bets_advised, bets_won, bets_lost, win_rate_on_advised_bets,
        total_staked_units, total_profit_units, roi, current_bet_streak, best_bet_streak,
        win_rate_on_advised_bets_ci_low, win_rate_on_advised_bets_ci_high, roi_p_value,
    ) = _bet_pnl_metrics(picks)

    clv = closing_line_value_table(picks, odds_snapshots=odds_snapshots)
    mean_prob_clv = (
        float(pd.to_numeric(clv["probability_clv"], errors="coerce").mean())
        if not clv.empty else float("nan")
    )
    n_clv = (
        int(pd.to_numeric(clv["probability_clv"], errors="coerce").notna().sum())
        if not clv.empty else 0
    )

    home_favored = picks["predicted_winner"] == picks["home_team"]
    picks["predicted_loser"] = picks["away_team"].where(home_favored, picks["home_team"])
    picks["market_predicted_winner_probability"] = picks["market_home_win_probability"].where(
        home_favored, 1 - picks["market_home_win_probability"]
    )

    out_cols = [
        "date", "game_pk", "home_team", "away_team", "predicted_winner", "predicted_loser",
        "predicted_probability", "above_threshold", "status", "market_home_win_probability",
        "market_predicted_winner_probability",
        "market_odds_snapshot_id", "market_odds_snapshot_role",
        "bet_units", "bet_side", "bet_team", "bet_moneyline", "bet_profit_units",
    ]
    for col in out_cols:
        if col not in picks.columns:
            picks[col] = pd.NA
    picks_out = picks[out_cols].sort_values("date", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "model_version": model_version if model_version is not None else "all_time",
                "metric": metric,
                "n_bets_advised": n_bets_advised,
                "bets_won": bets_won,
                "bets_lost": bets_lost,
                "win_rate_on_advised_bets": win_rate_on_advised_bets,
                "win_rate_on_advised_bets_ci_low": win_rate_on_advised_bets_ci_low,
                "win_rate_on_advised_bets_ci_high": win_rate_on_advised_bets_ci_high,
                "total_staked_units": total_staked_units,
                "total_profit_units": total_profit_units,
                "roi": roi,
                "roi_p_value": roi_p_value,
                "current_bet_streak": current_bet_streak,
                "best_bet_streak": best_bet_streak,
                "n_market_resolved": n_market_resolved,
                "market_accuracy": market_accuracy,
                "market_accuracy_ci_low": market_accuracy_ci_low,
                "market_accuracy_ci_high": market_accuracy_ci_high,
                "market_brier_score": market_brier,
                "market_log_loss": market_ll,
                "n_paired_market_compared": paired["n_compared"],
                "model_minus_market_brier": paired["brier_diff"],
                "model_minus_market_brier_ci_low": paired["brier_diff_ci_low"],
                "model_minus_market_brier_ci_high": paired["brier_diff_ci_high"],
                "model_minus_market_log_loss": paired["log_loss_diff"],
                "model_minus_market_log_loss_ci_low": paired["log_loss_diff_ci_low"],
                "model_minus_market_log_loss_ci_high": paired["log_loss_diff_ci_high"],
                "mean_model_minus_market_probability": paired["mean_prob_diff"],
                "mean_squared_error_diff": paired["mean_squared_error_diff"],
                "paired_bootstrap_n_date_blocks": paired["n_date_blocks"],
                "n_beat_closing_line_compared": n_beat_closing_line_compared,
                "beat_closing_line_rate": beat_closing_line_rate,
                "beat_closing_line_rate_ci_low": beat_ci_low,
                "beat_closing_line_rate_ci_high": beat_ci_high,
                "beat_closing_line_rate_p_value": beat_p,
                "n_clv_bets": n_clv,
                "mean_probability_clv": mean_prob_clv,
            }
        ]
    )
    return picks_out, summary


def _bet_pnl_metrics(picks: pd.DataFrame):
    """Real units won/lost, scoped to advised bets (bet_units > 0)."""
    resolved = picks[picks["bet_profit_units"].notna()].copy()
    n_bets_advised = len(resolved)
    if n_bets_advised == 0:
        return 0, 0, 0, float("nan"), 0.0, 0.0, float("nan"), 0, 0, 0.0, 1.0, float("nan")

    resolved["bet_profit_units"] = resolved["bet_profit_units"].astype(float)
    resolved["bet_units"] = resolved["bet_units"].astype(float)

    bets_won = int((resolved["bet_profit_units"] > 0).sum())
    bets_lost = int((resolved["bet_profit_units"] < 0).sum())
    win_rate = bets_won / n_bets_advised
    win_rate_ci_low, win_rate_ci_high = evaluation.wilson_confidence_interval(bets_won, n_bets_advised)
    total_staked = float(resolved["bet_units"].sum())
    total_profit = float(resolved["bet_profit_units"].sum())
    roi = total_profit / total_staked if total_staked else float("nan")
    roi_p_value = evaluation.mean_significance(resolved["bet_profit_units"], null_value=0.0)

    daily = (
        resolved.groupby("date", as_index=False)["bet_profit_units"]
        .sum()
        .sort_values("date")
    )
    current = best = streak = 0
    for profit in daily["bet_profit_units"]:
        if profit > 0:
            streak += 1
            current = streak
            best = max(best, streak)
        else:
            streak = 0
            current = 0
    return (
        n_bets_advised, bets_won, bets_lost, win_rate,
        total_staked, total_profit, roi, current, best,
        win_rate_ci_low, win_rate_ci_high, roi_p_value,
    )


def _market_comparison_metrics(recommended: pd.DataFrame, odds_snapshots: pd.DataFrame | None = None):
    """Market accuracy/Brier/log-loss using closing probabilities when available."""
    if recommended is None or recommended.empty:
        return float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan")

    with_market = recommended.copy()
    with_market["closing_market_home_probability"] = _resolve_closing_market_probability(
        with_market, odds_snapshots=odds_snapshots,
    )
    with_market = with_market[with_market["closing_market_home_probability"].notna()].copy()
    if with_market.empty:
        return float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan")

    market_home_prob = pd.to_numeric(with_market["closing_market_home_probability"], errors="coerce")
    favors_home = market_home_prob >= 0.5
    with_market["market_predicted_winner"] = with_market["home_team"].where(favors_home, with_market["away_team"])
    with_market["predicted_probability"] = market_home_prob.where(favors_home, 1 - market_home_prob)
    with_market["market_correct"] = pd.NA
    played = with_market["market_predicted_winner"].notna() & with_market["actual_winner"].notna()
    correct = with_market["market_predicted_winner"] == with_market["actual_winner"]
    with_market.loc[played & correct, "market_correct"] = 1.0
    with_market.loc[played & ~correct, "market_correct"] = 0.0

    resolved = evaluation.resolved_only(with_market, outcome_col="market_correct")
    n_resolved = len(resolved)
    accuracy = float(resolved["market_correct"].mean()) if n_resolved else float("nan")
    brier = evaluation.brier_score(with_market, outcome_col="market_correct")
    ll = evaluation.log_loss(with_market, outcome_col="market_correct")
    ci_low, ci_high = (
        evaluation.wilson_confidence_interval(int(resolved["market_correct"].sum()), n_resolved)
        if n_resolved else (float("nan"), float("nan"))
    )
    return accuracy, brier, ll, n_resolved, ci_low, ci_high


def _beat_closing_line_rate(recommended: pd.DataFrame, odds_snapshots: pd.DataFrame | None = None):
    """Secondary metric: fraction of games where model SE < closing-market SE."""
    paired = paired_market_scoring_differences(recommended, odds_snapshots=odds_snapshots)
    rate = paired["pct_model_error_lower"]
    n_compared = paired["n_beat_closing_line_compared"]
    if n_compared == 0 or rate != rate:
        return float("nan"), 0, float("nan"), float("nan"), float("nan")
    n_beat = int(round(rate * n_compared))
    ci_low, ci_high = evaluation.wilson_confidence_interval(n_beat, n_compared)
    p_value = evaluation.binomial_significance(n_beat, n_compared, null_probability=0.5)
    return rate, n_compared, ci_low, ci_high, p_value
