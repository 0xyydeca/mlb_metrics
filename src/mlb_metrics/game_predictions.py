"""Daily game-pick logging and outcome resolution - the game-level analog
of predictions.py. Kept as a separate module rather than extending
predictions.py: the entity (a game, keyed on game_pk) and its resolution
data source (final scores via schedule.fetch_game_results, not Statcast)
both differ enough from the batter-pick case that threading two entity
types through one module would hurt readability more than sharing code
would help.

Also owns `advise_bets` (real Kelly-edge bet-advice logic, formerly
scripts/recommend_bets.py's own private copy) - deliberately placed here
rather than in kelly.py (which stays pure scalar math, no pandas/config)
or a new module, since this IS the module that decides what goes into
each day's logged game-pick row, and bet advice is now one more real,
resolvable field on that same row (bet_units/bet_side/bet_moneyline/
bet_profit_units) - not a separate concern with its own lifecycle.
scripts/recommend_bets.py calls this same function rather than keeping
its own copy, so there's exactly one real implementation of "should a bet
be advised on this game, and how many units."

Bet sizing is tracked in UNITS, not dollars - the standard sports-betting
convention (bettors report/compare performance in bankroll-agnostic
"units risked/won" rather than dollars, since real bankroll size varies
per person - see config.UNIT_SIZE_FRACTION). `bet_units` is also the
single real signal for "was a bet advised at all": 0.0 means no bet, any
positive value is real units to risk - no separate boolean, so a numeric
column always tells the whole story on its own.
"""

import os
from datetime import datetime, timezone

import pandas as pd

from mlb_metrics import config, game_picks, kelly, market_odds, ml_models

GAME_PREDICTION_COLUMNS = [
    "date", "game_pk", "home_team", "away_team", "predicted_winner",
    "predicted_probability", "above_threshold", "metric", "actual_winner", "game_played", "model_version",
    "market_home_win_probability",
    "market_odds_snapshot_id", "market_odds_snapshot_role",
    "market_provider_event_id", "market_sportsbook",
    "market_captured_at_utc", "market_home_moneyline", "market_away_moneyline",
    "bet_units", "bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units",
    # Real per-game conservative probabilities (game_picks.apply_kelly_uncertainty,
    # 2026-08-25 - "we need the units risked to not be arbitrary") - persisted
    # (not just used transiently inside select_game_picks) so
    # scripts/recommend_bets.py's later re-derivation of advise_bets from
    # THIS log reuses the exact same real sizing the live pipeline already
    # computed for that game, rather than silently falling back to the flat
    # kelly_fraction_multiplier because the team-confidence context that
    # produced them isn't cheaply recomputable from a lightweight report
    # script. NaN for any game where `confidence` wasn't given/empty.
    "home_win_probability_pessimistic", "away_win_probability_pessimistic",
    # Provenance - see PROVENANCE_COLUMNS.
    "prediction_timestamp_utc", "prediction_code_sha", "model_artifact_id",
    "training_data_cutoff", "feature_schema_hash", "selection_logic_version",
    "selection_metric", "selection_score", "probability_source",
    "fallback_used", "fallback_reason",
    "prediction_snapshot_type", "lineup_status", "starter_status",
]

PROVENANCE_COLUMNS = [
    "prediction_timestamp_utc", "prediction_code_sha", "model_artifact_id",
    "training_data_cutoff", "feature_schema_hash", "selection_logic_version",
    "selection_metric", "selection_score", "probability_source",
    "fallback_used", "fallback_reason",
    "prediction_snapshot_type", "lineup_status", "starter_status",
]

_PROVENANCE_MIGRATION_DEFAULTS = {
    "prediction_timestamp_utc": pd.NA,
    "prediction_code_sha": "legacy",
    "model_artifact_id": pd.NA,
    "training_data_cutoff": pd.NA,
    "feature_schema_hash": pd.NA,
    "selection_logic_version": "legacy",
    "selection_metric": pd.NA,
    "selection_score": pd.NA,
    "probability_source": pd.NA,
    "fallback_used": pd.NA,
    "fallback_reason": pd.NA,
    "prediction_snapshot_type": "legacy",
    "lineup_status": "legacy",
    "starter_status": "legacy",
}

BET_ADVICE_COLUMNS = [
    "date", "game_pk", "side", "team", "opponent", "moneyline",
    "model_probability", "market_implied_probability", "edge", "kelly_stake_fraction",
]

# Same purpose as predictions.LEGACY_MODEL_VERSION - tagged onto any row
# logged before model_version existed, or reconstructed by a historical
# replay (see game_picks_backtest.py).
LEGACY_MODEL_VERSION = "legacy"


def advise_bets(
    todays_picks: pd.DataFrame,
    market: pd.DataFrame,
    kelly_fraction_multiplier: float,
    min_edge: float,
) -> pd.DataFrame:
    """Pure, testable core shared by both the live daily pipeline
    (select_game_picks below) and the standalone scripts/recommend_bets.py
    report - single source of truth for "should a real bet be advised on
    this game." No network. For each logged game, derives the model's real
    probability for BOTH sides from its own logged predicted_winner/
    predicted_probability (predicted_probability is always the model's OWN
    favored side's probability, >= 0.5 by construction - the other side's
    model probability is the complement). Compares each side against the
    REAL vigged market price (market_odds.moneyline_to_implied_probability
    on the real raw moneyline) - deliberately NOT the de-vigged
    market_home_win_probability column, which exists for a different
    purpose (fairly scoring forecast skill with the book's own edge
    removed). A real bet is paid off at the real, vigged price, so Kelly's
    net-odds term has to come from that same real price too. Where
    edge >= min_edge, sizes a stake via kelly.kelly_fraction.

    Provably never recommends both sides of one game when min_edge > 0:
    edge_home + edge_away = 1 - (home_implied + away_implied) = -vig, and a
    real book's implied probabilities always sum to > 1 (see
    market_odds.devig's own test), so the two edges can never both be
    positive at once. Kept as a defensive check anyway, not silently
    trusted - if a row's real data is malformed enough that both sides
    somehow clear min_edge together, that's a real data-quality anomaly
    (e.g. a stale matching collision), not a real arbitrage in this
    pipeline's actual data - both sides are dropped for that game rather
    than either one being guessed at. Returns at most one row per
    game_pk.

    Real bet-sizing follow-up (2026-08-25 - "we need the units risked to
    not be arbitrary"): if a row carries real (non-null)
    home_win_probability_pessimistic/away_win_probability_pessimistic
    values (see game_picks.apply_kelly_uncertainty, merged in by
    select_game_picks below), that row's stake is sized off THAT real,
    per-game conservative probability instead of the raw point estimate.
    Gated per ROW, not just per column: select_game_picks always persists
    these two columns (so scripts/recommend_bets.py's later re-derivation
    from the log reuses the same real sizing), but leaves them null for
    any game `confidence` didn't cover - a caller/test that never
    supplies them at all (game_picks_backtest.py's real historical
    replay, older fixtures) is unaffected, exactly like a caller that
    supplies the columns but with null values for a specific game.
    `min_edge` eligibility still uses the raw model_probability either
    way - "is there a real edge worth considering" stays a separate
    question from "how much to actually risk given that edge."

    `kelly_fraction_multiplier` is ALWAYS applied on top of whichever
    probability ends up sizing the bet (2026-08-26 - real follow-up:
    "the short-priced-favorite blowup risk" - a heavy favorite's small
    payout per unit staked means a small error in the sizing probability
    can flip real profit to real loss, and a team's season-long win-rate
    CI can stay tight - and so barely discount a confident raw estimate -
    even for a single game whose own real matchup-specific uncertainty
    the CI can't see). The CI-derived pessimistic probability and the
    flat multiplier are two INDEPENDENT layers of conservatism, not a
    swap of one for the other - a confident raw model probability run
    through even a real, non-arbitrary pessimistic adjustment can still
    imply a double-digit-percent-of-bankroll full-Kelly stake on a heavy
    favorite (see the README's own worked Dodgers/Rockies example), and
    the flat multiplier (half-Kelly by default) is what actually reins
    that in.

    Also enforces two real, user-set portfolio-level risk limits, in
    order (2026-08-26 - direct response to "maybe we scaled it down too
    much... this is now taking out games where the line is genuinely
    appealing"): first config.KELLY_MAX_SINGLE_BET_UNIT_CAP clips any ONE
    game's stake directly - the real, targeted defense against Kelly's
    own amplification of a thin sizing-probability margin into a large
    stake for short-priced favorites (stake sensitivity to that
    probability is proportional to 1 + 1/b, large when the net odds b
    are small), which the CI-based pessimistic probability and
    kelly_fraction_multiplier reduce but don't fully eliminate on their
    own - see config.KELLY_UNCERTAINTY_CI_ALPHA's own docstring for why
    that CI was widened rather than relied on alone for this. Then
    config.KELLY_DAILY_UNIT_CAP - if the day's total (post-per-bet-cap)
    stake fraction still exceeds the daily cap, every advised stake for
    that date is scaled down proportionally so the day's total lands
    exactly at the cap, preserving each bet's relative size rather than
    favoring whichever game happened to be evaluated first."""
    # Prefer game_pk join when the market frame carries matched game_pk
    # (timestamped odds snapshots) — avoids doubleheader collisions.
    if (
        "game_pk" in market.columns
        and market["game_pk"].notna().any()
        and "game_pk" in todays_picks.columns
    ):
        market_by_pk = market.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"], keep="last")
        pk_cols = [
            c for c in (
                "game_pk", "home_moneyline", "away_moneyline",
                "market_home_win_probability", "home_team", "away_team",
            )
            if c in market_by_pk.columns
        ]
        base = todays_picks.drop(
            columns=[c for c in ("home_moneyline", "away_moneyline", "market_home_win_probability") if c in todays_picks.columns],
            errors="ignore",
        )
        merged = base.merge(market_by_pk[pk_cols], on="game_pk", how="left")
    else:
        # Drop moneylines already on picks so a second merge (select_game_picks
        # already attached odds) does not create home_moneyline_x/_y ghosts.
        base = todays_picks.drop(
            columns=[c for c in ("home_moneyline", "away_moneyline") if c in todays_picks.columns],
            errors="ignore",
        )
        money_cols = [c for c in ("home_team", "away_team", "home_moneyline", "away_moneyline") if c in market.columns]
        merged = base.merge(market[money_cols], on=["home_team", "away_team"], how="left")

    rows = []
    for _, pick in merged.iterrows():
        if pd.isna(pick.get("home_moneyline")) or pd.isna(pick.get("away_moneyline")):
            continue  # no real market data for this game - skip, not fatal

        home_favored = pick["predicted_winner"] == pick["home_team"]
        home_model_probability = pick["predicted_probability"] if home_favored else 1 - pick["predicted_probability"]
        away_model_probability = 1 - home_model_probability

        home_pessimistic = pick.get("home_win_probability_pessimistic")
        away_pessimistic = pick.get("away_win_probability_pessimistic")
        if pd.notna(home_pessimistic) and pd.notna(away_pessimistic):
            home_sizing_probability = home_pessimistic
            away_sizing_probability = away_pessimistic
        else:
            home_sizing_probability = home_model_probability
            away_sizing_probability = away_model_probability

        sides = [
            ("home", pick["home_team"], pick["away_team"], home_model_probability, home_sizing_probability, pick["home_moneyline"]),
            ("away", pick["away_team"], pick["home_team"], away_model_probability, away_sizing_probability, pick["away_moneyline"]),
        ]

        candidates = []
        for side, team, opponent, model_probability, sizing_probability, moneyline in sides:
            market_implied = market_odds.moneyline_to_implied_probability(moneyline)
            edge = model_probability - market_implied
            if edge >= min_edge:
                stake_fraction = kelly.kelly_fraction(sizing_probability, moneyline, kelly_fraction_multiplier)
                candidates.append({
                    "date": pick["date"], "game_pk": pick["game_pk"], "side": side,
                    "team": team, "opponent": opponent, "moneyline": moneyline,
                    "model_probability": model_probability, "market_implied_probability": market_implied,
                    "edge": edge, "kelly_stake_fraction": stake_fraction,
                })

        if len(candidates) > 1:
            print(
                f"WARNING: both sides of game_pk={pick['game_pk']} cleared min_edge "
                f"simultaneously - a real data-quality anomaly (see advise_bets' own "
                f"docstring), not a real arbitrage. Dropping both sides for this game."
            )
            continue
        rows.extend(candidates)

    result = pd.DataFrame(rows, columns=BET_ADVICE_COLUMNS)
    if result.empty:
        return result

    single_bet_cap_fraction = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    result["kelly_stake_fraction"] = result["kelly_stake_fraction"].clip(upper=single_bet_cap_fraction)

    daily_cap_fraction = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_total = result.groupby("date")["kelly_stake_fraction"].transform("sum")
    over_cap = daily_total > daily_cap_fraction
    result.loc[over_cap, "kelly_stake_fraction"] = (
        result.loc[over_cap, "kelly_stake_fraction"] * daily_cap_fraction / daily_total[over_cap]
    )
    return result


def _ensure_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, default in _PROVENANCE_MIGRATION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    return df


def _ensure_market_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "market_odds_snapshot_id": pd.NA,
        "market_odds_snapshot_role": pd.NA,
        "market_provider_event_id": pd.NA,
        "market_sportsbook": pd.NA,
        "market_captured_at_utc": pd.NA,
        "market_home_moneyline": pd.NA,
        "market_away_moneyline": pd.NA,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def _infer_game_starter_status(row_frame: pd.DataFrame) -> pd.Series:
    if "starter_status" in row_frame.columns:
        return row_frame["starter_status"]
    home_col = "home_probable_pitcher_key_mlbam"
    away_col = "away_probable_pitcher_key_mlbam"
    if home_col in row_frame.columns and away_col in row_frame.columns:
        both = row_frame[home_col].notna() & row_frame[away_col].notna()
        either = row_frame[home_col].notna() | row_frame[away_col].notna()
        status = pd.Series("missing", index=row_frame.index)
        status = status.mask(either, "probable")
        status = status.mask(both, "probable")
        return status
    return pd.Series("probable", index=row_frame.index)


def select_game_picks(
    win_probabilities: pd.DataFrame,
    date,
    min_probability: float = config.GAME_PICK_MIN_PROBABILITY,
    metric: str = "GamePick_Win_Probability",
    model_version: str = config.GAME_PICK_MODEL_VERSION,
    market_probabilities: pd.DataFrame | None = None,
    kelly_fraction_multiplier: float = config.KELLY_FRACTION_MULTIPLIER,
    min_edge: float = config.KELLY_MIN_EDGE,
    confidence: pd.DataFrame | None = None,
    model_status: dict | None = None,
    fallback_used: bool | None = None,
    fallback_reason: str | None = None,
    probability_source: str = "home_win_probability",
    prediction_snapshot_type: str = "morning",
) -> pd.DataFrame:
    """Turn game_picks.compute_game_win_probabilities' output into the
    day's logged games - EVERY scheduled game, not just the ones that clear
    `min_probability`. `above_threshold` flags whether the favored side's
    win probability clears `min_probability` ("real separation between the
    two teams") - the dashboard still publishes the complete slate and
    highlights the flagged games, but game_evaluation.build_game_picks_export's
    real tracking is scoped to `bet_units > 0` (below), not `above_threshold` -
    a model being confident in its own favorite is a different question
    from the market actually disagreeing with it. `predicted_winner` is
    whichever side (home_team/away_team) has the higher probability;
    `predicted_probability` is that side's probability (always >= 0.5).
    `model_version` (see config.GAME_PICK_MODEL_VERSION) is stamped onto
    every row so game_evaluation.py/the dashboard can segment by which
    win-probability logic actually produced a pick - same reasoning as
    predictions.select_picks' own model_version.

    `market_probabilities` (default None - unchanged behavior for every
    existing caller/test, including game_picks_backtest.py's real calls
    that never pass one) is market_odds.fetch_market_home_win_probabilities'
    output, a real DataFrame keyed by (home_team, away_team) with a
    market_home_win_probability column - left-merged in when given, so a
    day with no market data (fetch failed, or the caller doesn't pass one)
    still logs picks normally with that column all-NaN.

    When `market_probabilities` also carries real `home_moneyline`/
    `away_moneyline` (the raw, vigged price - a defensive column check,
    not just an empty check, so any older caller/test still passing the
    original 3-column de-vigged-only frame keeps working with `bet_units`
    simply staying 0.0), calls `advise_bets` to decide whether a real bet
    is advised on each game and logs the result (`bet_units`/`bet_side`/
    `bet_team`/`bet_moneyline`/`bet_stake_fraction` - `bet_profit_units`
    stays null here, filled in only once resolve_game_predictions later
    knows the real outcome). `bet_units = bet_stake_fraction / config.UNIT_SIZE_FRACTION`
    - the standard sports-betting "units risked" convention, bankroll-
    agnostic by design (see config.UNIT_SIZE_FRACTION's own docstring).
    `bet_units` IS the real "was a bet advised" signal: 0.0 for every game
    with no real edge, a positive number of units otherwise - no separate
    boolean.

    advise_bets returns at most one row per game_pk in the normal case,
    but `market_probabilities` matches by (home_team, away_team) only (see
    market_odds.py's own doubleheader caveat) - a real doubleheader with
    two ESPN-reported events for the same two teams could in principle
    produce two advice rows for the same game_pk. Defended against
    explicitly here (not silently trusted): any game_pk appearing more
    than once in the advice is dropped from consideration entirely (with
    a warning) rather than picking one arbitrarily, so the final merge
    below is always safely one-to-one.

    `confidence` (default None - unchanged behavior for every existing
    caller/test that doesn't pass one) is teams.assemble_team_metrics'
    output, carrying each team's real win_rate_CI_Low/CI_High - when
    given, game_picks.apply_kelly_uncertainty computes a real, per-game
    conservative probability that `advise_bets` sizes stakes off, with
    `kelly_fraction_multiplier` still applied on top of it (see that
    function's own docstring for why - "we need the units risked to not
    be arbitrary," and its 2026-08-26 follow-up on why the flat
    multiplier isn't dropped once this pessimistic probability exists).

    Provenance kwargs (`model_status` / `fallback_used` / `fallback_reason`
    / `probability_source` / `prediction_snapshot_type`) do not change which
    games are logged or how bets are sized - they only stamp
    PROVENANCE_COLUMNS onto returned rows.
    """
    df = win_probabilities.copy()
    favors_home = df["home_win_probability"] >= 0.5
    df["predicted_winner"] = df["home_team"].where(favors_home, df["away_team"])
    df["predicted_probability"] = df["home_win_probability"].where(favors_home, 1 - df["home_win_probability"])
    df["above_threshold"] = df["predicted_probability"] >= min_probability

    picks = df.copy()
    picks["date"] = pd.Timestamp(date)
    picks["metric"] = metric
    picks["actual_winner"] = pd.NA
    picks["game_played"] = pd.NA
    picks["model_version"] = model_version

    # Attach the exact odds snapshot this prediction used.
    for col in (
        "market_home_win_probability", "market_odds_snapshot_id", "market_odds_snapshot_role",
        "market_provider_event_id", "market_sportsbook", "market_captured_at_utc",
        "market_home_moneyline", "market_away_moneyline",
    ):
        if col not in picks.columns:
            picks[col] = pd.NA

    market_for_advice = None
    if market_probabilities is not None and not market_probabilities.empty:
        market = market_probabilities.copy()
        # Normalize sportsbook / snapshot column names from odds snapshots.
        if "sportsbook" in market.columns:
            market["market_sportsbook"] = market["sportsbook"]
            if "market_provider" not in market.columns:
                market["market_provider"] = market["sportsbook"]
        if "market_provider" in market.columns and "market_sportsbook" not in market.columns:
            market["market_sportsbook"] = market["market_provider"]
        if "snapshot_id" in market.columns and "market_odds_snapshot_id" not in market.columns:
            market["market_odds_snapshot_id"] = market["snapshot_id"]
        if "snapshot_role" in market.columns and "market_odds_snapshot_role" not in market.columns:
            market["market_odds_snapshot_role"] = market["snapshot_role"]
        if "provider_event_id" in market.columns and "market_provider_event_id" not in market.columns:
            market["market_provider_event_id"] = market["provider_event_id"]
        if "captured_at_utc" in market.columns and "market_captured_at_utc" not in market.columns:
            market["market_captured_at_utc"] = market["captured_at_utc"]
        if "home_moneyline" in market.columns and "market_home_moneyline" not in market.columns:
            market["market_home_moneyline"] = market["home_moneyline"]
        if "away_moneyline" in market.columns and "market_away_moneyline" not in market.columns:
            market["market_away_moneyline"] = market["away_moneyline"]
        if "home_moneyline" not in market.columns and "market_home_moneyline" in market.columns:
            market["home_moneyline"] = market["market_home_moneyline"]
            market["away_moneyline"] = market["market_away_moneyline"]

        attach_cols = [
            c for c in (
                "market_home_win_probability", "market_odds_snapshot_id", "market_odds_snapshot_role",
                "market_provider_event_id", "market_sportsbook", "market_captured_at_utc",
                "market_home_moneyline", "market_away_moneyline",
                "home_moneyline", "away_moneyline", "game_pk", "home_team", "away_team",
            )
            if c in market.columns
        ]
        if "game_pk" in market.columns and market["game_pk"].notna().any():
            mkt = market.dropna(subset=["game_pk"])[attach_cols].drop_duplicates("game_pk", keep="last")
            picks = picks.drop(columns=[c for c in attach_cols if c in picks.columns and c != "game_pk"], errors="ignore")
            picks = picks.merge(mkt, on="game_pk", how="left")
        else:
            key_cols = ["home_team", "away_team"]
            mkt = market[attach_cols].drop_duplicates(key_cols, keep="last")
            picks = picks.drop(
                columns=[c for c in attach_cols if c in picks.columns and c not in key_cols],
                errors="ignore",
            )
            picks = picks.merge(mkt, on=key_cols, how="left")

        if "market_odds_snapshot_role" in picks.columns:
            picks["market_odds_snapshot_role"] = picks["market_odds_snapshot_role"].fillna(prediction_snapshot_type)
        else:
            picks["market_odds_snapshot_role"] = prediction_snapshot_type
        market_for_advice = market
    else:
        picks["market_home_win_probability"] = pd.NA
        picks["market_odds_snapshot_id"] = pd.NA
        picks["market_odds_snapshot_role"] = pd.NA
        picks["market_provider_event_id"] = pd.NA
        picks["market_sportsbook"] = pd.NA
        picks["market_captured_at_utc"] = pd.NA
        picks["market_home_moneyline"] = pd.NA
        picks["market_away_moneyline"] = pd.NA

    # advise_bets still expects home_moneyline/away_moneyline column names.
    if "home_moneyline" not in picks.columns and "market_home_moneyline" in picks.columns:
        picks["home_moneyline"] = picks["market_home_moneyline"]
    if "away_moneyline" not in picks.columns and "market_away_moneyline" in picks.columns:
        picks["away_moneyline"] = picks["market_away_moneyline"]

    has_moneylines = (
        market_for_advice is not None
        and not market_for_advice.empty
        and (
            {"home_moneyline", "away_moneyline"}.issubset(market_for_advice.columns)
            or {"market_home_moneyline", "market_away_moneyline"}.issubset(market_for_advice.columns)
        )
    )

    if confidence is not None and not confidence.empty:
        pessimistic = game_picks.apply_kelly_uncertainty(win_probabilities, confidence)
        picks = picks.merge(
            pessimistic[["game_pk", "home_win_probability_pessimistic", "away_win_probability_pessimistic"]],
            on="game_pk",
            how="left",
        )
    else:
        picks["home_win_probability_pessimistic"] = pd.NA
        picks["away_win_probability_pessimistic"] = pd.NA

    if has_moneylines:
        advice = advise_bets(picks, market_for_advice, kelly_fraction_multiplier, min_edge)
        dupe_game_pks = advice.loc[advice["game_pk"].duplicated(keep=False), "game_pk"].unique()
        if len(dupe_game_pks) > 0:
            print(
                f"WARNING: advise_bets returned more than one row for game_pk(s) "
                f"{list(dupe_game_pks)} - a real doubleheader/matching collision (see "
                f"market_odds.py's own doubleheader caveat), not a data error to guess "
                f"through. Dropping those games from bet advice entirely."
            )
            advice = advice[~advice["game_pk"].isin(dupe_game_pks)]

        picks = picks.merge(
            advice[["game_pk", "side", "team", "moneyline", "kelly_stake_fraction"]].rename(
                columns={
                    "side": "bet_side", "team": "bet_team",
                    "moneyline": "bet_moneyline", "kelly_stake_fraction": "bet_stake_fraction",
                }
            ),
            on="game_pk",
            how="left",
        )
        # bet_units is the real "was a bet advised" signal - 0.0 (not NaN)
        # for every game with no real edge, so it's always a valid number,
        # never a separate boolean to also check.
        picks["bet_units"] = (picks["bet_stake_fraction"] / config.UNIT_SIZE_FRACTION).fillna(0.0)
    else:
        picks["bet_units"] = 0.0
        picks["bet_side"] = pd.NA
        picks["bet_team"] = pd.NA
        picks["bet_moneyline"] = pd.NA
        picks["bet_stake_fraction"] = pd.NA

    picks["bet_profit_units"] = pd.NA

    if model_status is not None:
        provenance = ml_models.provenance_fields_from_model_status(model_status)
    else:
        provenance = {
            "model_artifact_id": pd.NA,
            "training_data_cutoff": pd.NA,
            "feature_schema_hash": pd.NA,
            "fallback_used": False,
            "fallback_reason": None,
        }
    if fallback_used is not None:
        provenance["fallback_used"] = bool(fallback_used)
        provenance["fallback_reason"] = fallback_reason

    picks["prediction_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    picks["prediction_code_sha"] = ml_models.resolve_code_sha()
    picks["model_artifact_id"] = provenance["model_artifact_id"]
    picks["training_data_cutoff"] = provenance["training_data_cutoff"]
    picks["feature_schema_hash"] = provenance["feature_schema_hash"]
    picks["selection_logic_version"] = model_version
    # Game picks log the full slate (not a top-N rank). The score that
    # determines the predicted side is predicted_probability itself.
    picks["selection_metric"] = "predicted_probability"
    picks["selection_score"] = picks["predicted_probability"]
    picks["probability_source"] = probability_source
    picks["fallback_used"] = provenance["fallback_used"]
    picks["fallback_reason"] = provenance["fallback_reason"]
    picks["prediction_snapshot_type"] = prediction_snapshot_type
    picks["lineup_status"] = "unconfirmed"
    picks["starter_status"] = _infer_game_starter_status(picks)

    picks = _ensure_market_odds_columns(picks)
    for col in GAME_PREDICTION_COLUMNS:
        if col not in picks.columns:
            picks[col] = pd.NA
    return picks[GAME_PREDICTION_COLUMNS]


def append_game_prediction_audit(
    picks: pd.DataFrame,
    audit_path: str | None = None,
    *,
    published_log_path: str | None = None,
) -> None:
    """Append-only immutable audit of every game-prediction batch."""
    if audit_path is None:
        if published_log_path:
            root, ext = os.path.splitext(published_log_path)
            audit_path = f"{root}_audit{ext}"
        else:
            audit_path = config.GAME_PREDICTIONS_AUDIT_PATH
    frame = picks.copy()
    if frame.empty:
        return
    os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
    if os.path.exists(audit_path):
        existing = pd.read_csv(audit_path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    combined.to_csv(audit_path, index=False)


def append_game_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the game-predictions log at `log_path`, deduping
    on (date, game_pk, metric).

    Unresolved same-day (or same game_pk) rows may be refreshed by a newer
    batch. Already-resolved rows (``game_played`` not null) are immutable and
    always win over a re-logged pick for the same key — mirroring the
    hitter-side ``append_predictions`` supersede rules. Every batch is also
    written to the immutable audit log beside the published CSV.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    picks = picks.copy()
    picks["date"] = pd.to_datetime(picks["date"])
    append_game_prediction_audit(picks, published_log_path=log_path)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        if "above_threshold" not in existing.columns:
            # Every row logged before this column existed already cleared
            # the old hard filter select_game_picks used to apply - True is
            # the factually correct backfill, not an arbitrary default.
            existing["above_threshold"] = True
        if "market_home_win_probability" not in existing.columns:
            # A log written before slice 2 genuinely has no real market
            # data for those rows - NaN, not a guess.
            existing["market_home_win_probability"] = pd.NA
        if "bet_units" not in existing.columns:
            # A row logged before bet advice existed genuinely never had a
            # bet advised - 0.0 is the factually correct backfill (0 units
            # means no bet, same real signal a freshly-logged non-advised
            # row would get), not an arbitrary default (same reasoning as
            # above_threshold's own backfill above).
            existing["bet_units"] = 0.0
            for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units"):
                existing[col] = pd.NA
        if "home_win_probability_pessimistic" not in existing.columns:
            # A row logged before uncertainty-scaled Kelly existed genuinely
            # has no real per-game pessimistic probability to report - NaN,
            # not a guess (same reasoning as market_home_win_probability's
            # own backfill above).
            existing["home_win_probability_pessimistic"] = pd.NA
            existing["away_win_probability_pessimistic"] = pd.NA
        existing = _ensure_provenance_columns(existing)
        existing = _ensure_market_odds_columns(existing)
        picks = _ensure_provenance_columns(picks)
        picks = _ensure_market_odds_columns(picks)
        existing["date"] = pd.to_datetime(existing["date"])

        # Drop unresolved existing rows for keys present in the fresh batch
        # so a lineup-lock / re-run can refresh still-pending game picks.
        # Resolved rows (game_played notna) are never dropped. Key-level only
        # — do not wipe other same-day games that were not recomputed.
        fresh_keys = picks[["date", "game_pk", "metric"]].drop_duplicates()
        existing = existing.merge(
            fresh_keys.assign(_fresh=1),
            on=["date", "game_pk", "metric"],
            how="left",
        )
        is_resolved = (
            existing["game_played"].notna()
            if "game_played" in existing.columns
            else pd.Series(False, index=existing.index)
        )
        drop_mask = existing["_fresh"].eq(1) & ~is_resolved
        existing = existing.loc[~drop_mask].drop(columns=["_fresh"], errors="ignore")

        # picks first, then existing: keep="last" lets resolved rows that
        # remained in `existing` win over a re-logged unresolved duplicate.
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        picks = _ensure_provenance_columns(picks)
        picks = _ensure_market_odds_columns(picks)
        combined = picks

    combined = combined.drop_duplicates(subset=["date", "game_pk", "metric"], keep="last")
    combined = combined.sort_values(["date", "game_pk"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_game_predictions(log_path: str, fetch_results_fn, as_of_date) -> pd.DataFrame:
    """Fill in `actual_winner`/`game_played` for any still-pending rows
    (`game_played` is null) with a date strictly before `as_of_date`, by
    calling `fetch_results_fn(date)` (i.e. schedule.fetch_game_results) once
    per distinct pending date. `as_of_date` is explicit rather than reading
    the wall clock - same no-implicit-"now" philosophy as pipeline.run()
    itself (see its module docstring), and it keeps this function's
    behavior fully determined by its inputs, not by when it happens to run.
    Unlike Statcast (one bulk range fetch), statsapi has no bulk date-range
    endpoint, so this is necessarily N fetches for N distinct pending dates
    - each one is try/except-guarded so one bad/unreachable date can't
    block resolving the others or raise out of this function.

    Only a `status == "Final"` game is resolved (winner = whichever side
    scored more). Any other status (Scheduled, In Progress, postponed,
    etc.) is left pending - the exact status strings MLB Stats API uses for
    a postponed/cancelled game aren't confirmed in this codebase, so rather
    than guess at them and risk mis-resolving a game, those picks simply
    stay pending indefinitely in this first pass (a safe failure mode, not
    a wrong-data one)."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=GAME_PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    if "game_played" not in log.columns:
        log["game_played"] = pd.NA  # migrate a log written before game_played existed
    if "model_version" not in log.columns:
        log["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
    if "above_threshold" not in log.columns:
        log["above_threshold"] = True  # migrate a log written before above_threshold existed (see append_game_predictions)
    if "market_home_win_probability" not in log.columns:
        log["market_home_win_probability"] = pd.NA  # migrate a log written before slice 2's market column existed
    if "bet_units" not in log.columns:
        # A row logged before bet advice existed genuinely never had a bet
        # advised - 0.0 units is the factually correct backfill, not an
        # arbitrary default (see append_game_predictions's identical guard).
        log["bet_units"] = 0.0
        for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units"):
            log[col] = pd.NA
    log = _ensure_provenance_columns(log)
    log = _ensure_market_odds_columns(log)
    # A log with no resolved games yet round-trips actual_winner as an
    # all-null float64 column (empty strings -> NaN on read) - cast back to
    # object so assigning a team abbreviation string into it doesn't raise.
    log["actual_winner"] = log["actual_winner"].astype("object")

    pending_dates = log.loc[
        log["game_played"].isna() & (log["date"] < pd.Timestamp(as_of_date)), "date"
    ].unique()

    for date in pending_dates:
        date = pd.Timestamp(date)
        try:
            results = fetch_results_fn(date.date())
        except Exception as exc:
            print(
                f"WARNING: failed to fetch game results for {date.date()} ({exc}); "
                f"leaving that date's game picks pending."
            )
            continue
        if results.empty:
            continue

        results = results[["game_pk", "status", "home_score", "away_score"]].rename(
            columns={"status": "_status", "home_score": "_home_score", "away_score": "_away_score"}
        )
        merged = log.merge(results, on="game_pk", how="left")

        day_mask = (log["date"] == date) & log["game_played"].isna()
        final = day_mask & (merged["_status"] == "Final")
        home_won = final & (merged["_home_score"] > merged["_away_score"])
        away_won = final & (merged["_away_score"] > merged["_home_score"])

        log.loc[final, "game_played"] = 1
        log.loc[home_won, "actual_winner"] = log.loc[home_won, "home_team"]
        log.loc[away_won, "actual_winner"] = log.loc[away_won, "away_team"]

        # Real units won/lost, only for newly-finalized rows where a real
        # bet was advised (bet_units > 0 - a real positive stake, the
        # single "was a bet advised" signal; a row can never be negative
        # and pd.to_numeric handles any object-dtype float/NaN mix a real
        # CSV round-trip can produce).
        advised_final = final & (pd.to_numeric(log["bet_units"], errors="coerce") > 0)
        team_won = (home_won & (log["bet_team"] == log["home_team"])) | (
            away_won & (log["bet_team"] == log["away_team"])
        )
        bet_won = advised_final & team_won
        bet_lost = advised_final & ~team_won
        log.loc[bet_won, "bet_profit_units"] = log.loc[bet_won, "bet_units"] * log.loc[
            bet_won, "bet_moneyline"
        ].apply(kelly.moneyline_to_net_odds)
        log.loc[bet_lost, "bet_profit_units"] = -log.loc[bet_lost, "bet_units"]

    log.to_csv(log_path, index=False)
    return log
