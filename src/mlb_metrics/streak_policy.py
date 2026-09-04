"""Streak-action decision layer: sit / play one / play two.

Chooses among sitting out, one hitter, or two hitters given the current
streak, remaining horizon, and each candidate's hit / miss / void
probabilities. Does **not** assume two picks are always better than one.

Outcome triple for each pick (must sum to 1)::

    P(hit)  = Final_Hit_Probability
    P(void) = 1 - P_Appear   (no appearance / no_game)
    P(miss) = 1 - P(hit) - P(void)

Two-pick day transitions (Beat the Streak rules)::

    - any miss  -> streak resets to 0
    - no misses, two hits -> streak += 2
    - no misses, one hit  -> streak += 1
    - two voids -> streak unchanged ( += 0 )

Joint outcomes start from independence, then an optional conservative
dependence penalty (same game / team / park / opposing same game) may
inflate joint-miss mass. The penalty defaults to 0 until earned inside
nested validation — never presented as a learned correlation.

Live wiring is shadow by default (``config.STREAK_POLICY_MODE``); promote
only after untouched outer-fold improvement vs legacy fixed policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from mlb_metrics import config

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ACTIONS = ("sit", "one", "two")
UTILITIES = (
    "expected_final_streak",
    "probability_reach_57",
    "probability_reach_target",
    "expected_streak_gain_with_reset_penalty",
)


@dataclass(frozen=True)
class OutcomeTriple:
    """P(hit), P(miss), P(void) — always renormalized to sum to 1."""

    p_hit: float
    p_miss: float
    p_void: float

    def __post_init__(self):
        total = float(self.p_hit) + float(self.p_miss) + float(self.p_void)
        if total <= 0:
            object.__setattr__(self, "p_hit", 0.0)
            object.__setattr__(self, "p_miss", 0.0)
            object.__setattr__(self, "p_void", 1.0)
            return
        object.__setattr__(self, "p_hit", float(self.p_hit) / total)
        object.__setattr__(self, "p_miss", float(self.p_miss) / total)
        object.__setattr__(self, "p_void", float(self.p_void) / total)

    def as_array(self) -> np.ndarray:
        return np.array([self.p_hit, self.p_miss, self.p_void], dtype=float)


@dataclass
class Candidate:
    key_mlbam: Any
    name: str = ""
    team: str | None = None
    game_pk: Any = None
    opponent: str | None = None
    park_id: Any = None
    is_home: float | None = None
    final_hit_probability: float = 0.0
    p_appear: float | None = None
    outcomes: OutcomeTriple | None = None

    def resolved_outcomes(self) -> OutcomeTriple:
        if self.outcomes is not None:
            return self.outcomes
        return outcome_triple_from_probabilities(
            self.final_hit_probability, self.p_appear,
        )


@dataclass
class ActionDecision:
    action: str  # sit | one | two
    expected_utility: float
    streak: int
    days_remaining: int
    utility_name: str
    picks: list[Candidate] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "expected_utility": self.expected_utility,
            "streak": self.streak,
            "days_remaining": self.days_remaining,
            "utility_name": self.utility_name,
            "n_picks": len(self.picks),
            "pick_keys": [c.key_mlbam for c in self.picks],
            "pick_names": [c.name for c in self.picks],
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------------


def outcome_triple_from_probabilities(
    final_hit_probability: float,
    p_appear: float | None = None,
    *,
    default_appear: float = 0.85,
) -> OutcomeTriple:
    """Build (hit, miss, void) from Final_Hit_Probability and P(Appear).

    ``Final_Hit_Probability = P_Appear × P(Hit|Appear)``, so::

        P(hit)  = Final_Hit_Probability
        P(void) = 1 - P_Appear
        P(miss) = P_Appear - Final_Hit_Probability
    """
    p_hit = float(np.clip(final_hit_probability, 0.0, 1.0))
    if p_appear is None or (isinstance(p_appear, float) and p_appear != p_appear):
        # If we only know Final_Hit, infer a soft appearance floor so void
        # mass stays distinguishable from miss without inventing correlation.
        appear = max(float(default_appear), p_hit)
    else:
        appear = float(np.clip(p_appear, 0.0, 1.0))
        appear = max(appear, p_hit)  # appear cannot be below hit
    p_void = 1.0 - appear
    p_miss = max(0.0, appear - p_hit)
    return OutcomeTriple(p_hit=p_hit, p_miss=p_miss, p_void=p_void)


def assert_outcomes_sum_to_one(triple: OutcomeTriple, tol: float = 1e-9) -> None:
    s = triple.p_hit + triple.p_miss + triple.p_void
    if abs(s - 1.0) > tol:
        raise AssertionError(f"Outcome probabilities must sum to 1, got {s}")


# ---------------------------------------------------------------------------
# Day transitions (exact rules)
# ---------------------------------------------------------------------------


def apply_one_pick_transition(streak: int, outcome: str) -> int:
    """outcome in {hit, miss, void}."""
    if outcome == "miss":
        return 0
    if outcome == "hit":
        return int(streak) + 1
    if outcome == "void":
        return int(streak)
    raise ValueError(f"Unknown outcome: {outcome}")


def apply_two_pick_transition(streak: int, outcome_a: str, outcome_b: str) -> int:
    """Exact two-pick day transition rules."""
    outcomes = (outcome_a, outcome_b)
    if "miss" in outcomes:
        return 0
    hits = sum(1 for o in outcomes if o == "hit")
    if hits == 2:
        return int(streak) + 2
    if hits == 1:
        return int(streak) + 1
    # two voids
    return int(streak)


def one_pick_next_streak_distribution(streak: int, triple: OutcomeTriple) -> dict[int, float]:
    """Map next_streak -> probability under one pick."""
    assert_outcomes_sum_to_one(triple)
    dist: dict[int, float] = {}
    for outcome, p in (("hit", triple.p_hit), ("miss", triple.p_miss), ("void", triple.p_void)):
        nxt = apply_one_pick_transition(streak, outcome)
        dist[nxt] = dist.get(nxt, 0.0) + p
    return dist


def independent_two_pick_joint(a: OutcomeTriple, b: OutcomeTriple) -> dict[tuple[str, str], float]:
    """Joint PMF over (outcome_a, outcome_b) under independence."""
    assert_outcomes_sum_to_one(a)
    assert_outcomes_sum_to_one(b)
    labels = ("hit", "miss", "void")
    pa = {"hit": a.p_hit, "miss": a.p_miss, "void": a.p_void}
    pb = {"hit": b.p_hit, "miss": b.p_miss, "void": b.p_void}
    return {(oa, ob): pa[oa] * pb[ob] for oa in labels for ob in labels}


def dependence_labels(
    a: Candidate,
    b: Candidate,
) -> dict[str, bool]:
    same_game = (
        a.game_pk is not None and b.game_pk is not None
        and pd.notna(a.game_pk) and pd.notna(b.game_pk)
        and a.game_pk == b.game_pk
    )
    same_team = (
        a.team is not None and b.team is not None
        and str(a.team) == str(b.team)
    )
    shared_park = (
        a.park_id is not None and b.park_id is not None
        and pd.notna(a.park_id) and pd.notna(b.park_id)
        and a.park_id == b.park_id
    )
    # Opposing players in the same game (different teams, same game_pk).
    opposing_same_game = bool(
        same_game and a.team is not None and b.team is not None and str(a.team) != str(b.team)
    )
    return {
        "same_game": bool(same_game),
        "same_team": bool(same_team),
        "shared_park": bool(shared_park),
        "opposing_same_game": opposing_same_game,
    }


def dependence_penalty(
    a: Candidate,
    b: Candidate,
    *,
    same_game: float | None = None,
    same_team: float | None = None,
    shared_park: float | None = None,
    opposing_same_game: float | None = None,
) -> float:
    """Sum of configured conservative penalties (default all zero)."""
    same_game = config.STREAK_POLICY_DEPENDENCE_SAME_GAME if same_game is None else same_game
    same_team = config.STREAK_POLICY_DEPENDENCE_SAME_TEAM if same_team is None else same_team
    shared_park = config.STREAK_POLICY_DEPENDENCE_SHARED_PARK if shared_park is None else shared_park
    opposing_same_game = (
        config.STREAK_POLICY_DEPENDENCE_OPPOSING_SAME_GAME
        if opposing_same_game is None else opposing_same_game
    )
    flags = dependence_labels(a, b)
    penalty = 0.0
    if flags["same_game"]:
        penalty += float(same_game)
    if flags["same_team"]:
        penalty += float(same_team)
    if flags["shared_park"]:
        penalty += float(shared_park)
    if flags["opposing_same_game"]:
        penalty += float(opposing_same_game)
    return max(0.0, penalty)


def apply_conservative_dependence(
    joint: dict[tuple[str, str], float],
    a: OutcomeTriple,
    b: OutcomeTriple,
    penalty: float,
) -> dict[tuple[str, str], float]:
    """Inflate joint-miss mass toward Frechet upper bound; default penalty=0 is identity.

    Not a learned correlation — a configurable conservative shift evaluated
    inside nested validation and defaulted to zero until earned.
    """
    if penalty <= 0:
        return dict(joint)
    # Paths with any miss
    miss_keys = [k for k in joint if "miss" in k]
    p_any_miss = sum(joint[k] for k in miss_keys)
    # Frechet-ish upper on both-miss
    both_miss_cap = min(a.p_miss, b.p_miss)
    both_miss_indep = joint.get(("miss", "miss"), 0.0)
    target_both = both_miss_indep + penalty * (both_miss_cap - both_miss_indep)
    target_both = float(np.clip(target_both, both_miss_indep, both_miss_cap))
    delta = target_both - both_miss_indep
    if delta <= 0 or p_any_miss <= both_miss_indep + 1e-15:
        return dict(joint)

    out = dict(joint)
    out[("miss", "miss")] = target_both
    # Take mass from other miss-containing cells proportionally
    donors = [k for k in miss_keys if k != ("miss", "miss")]
    donor_mass = sum(joint[k] for k in donors)
    if donor_mass <= 0:
        return dict(joint)
    for k in donors:
        share = joint[k] / donor_mass
        out[k] = max(0.0, joint[k] - delta * share)
    # Renormalize
    total = sum(out.values())
    if total <= 0:
        return dict(joint)
    return {k: v / total for k, v in out.items()}


def two_pick_next_streak_distribution(
    streak: int,
    a: Candidate,
    b: Candidate,
    *,
    dependence_penalty_value: float | None = None,
) -> tuple[dict[int, float], dict]:
    """Next-streak distribution for two picks (independence + optional penalty)."""
    oa = a.resolved_outcomes()
    ob = b.resolved_outcomes()
    joint = independent_two_pick_joint(oa, ob)
    penalty = (
        dependence_penalty(a, b)
        if dependence_penalty_value is None
        else float(dependence_penalty_value)
    )
    joint = apply_conservative_dependence(joint, oa, ob, penalty)
    dist: dict[int, float] = {}
    for (xa, xb), p in joint.items():
        nxt = apply_two_pick_transition(streak, xa, xb)
        dist[nxt] = dist.get(nxt, 0.0) + p
    meta = {
        "dependence_penalty": penalty,
        "dependence_flags": dependence_labels(a, b),
        "independence": penalty <= 0,
    }
    return dist, meta


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def terminal_utility(
    streak: int,
    name: str = "expected_final_streak",
    *,
    target: int | None = None,
    reset_penalty: float = 0.0,
    previous_streak: int | None = None,
    reset: bool = False,
) -> float:
    """Terminal (or one-step) utility value for a streak state."""
    if name == "expected_final_streak":
        return float(streak)
    if name == "probability_reach_57":
        return 1.0 if streak >= 57 else 0.0
    if name == "probability_reach_target":
        t = int(config.STREAK_POLICY_TARGET if target is None else target)
        return 1.0 if streak >= t else 0.0
    if name == "expected_streak_gain_with_reset_penalty":
        prev = int(streak if previous_streak is None else previous_streak)
        if reset:
            return float(-abs(reset_penalty)) - float(prev)  # lost the prior streak
        # gain relative to previous; if previous_streak provided use it
        base = int(previous_streak) if previous_streak is not None else int(streak)
        return float(streak - base)
    raise ValueError(f"Unknown utility: {name}")


def expected_utility_from_distribution(
    dist: dict[int, float],
    continuation: Callable[[int], float],
) -> float:
    return float(sum(p * continuation(s) for s, p in dist.items()))


# ---------------------------------------------------------------------------
# Dynamic programming / backward induction
# ---------------------------------------------------------------------------


@dataclass
class SolvedPolicy:
    """Value function V[r][s] = expected terminal utility with r days left at streak s."""

    V: list[list[float]]
    horizon: int
    max_streak: int
    utility_name: str
    target: int
    day_samples: list  # each sample: list[Candidate] slates used for E_max


def _continuation_lookup(solved: SolvedPolicy, days_remaining: int, streak: int) -> float:
    r = max(0, min(int(days_remaining), solved.horizon))
    s = max(0, min(int(streak), solved.max_streak))
    return float(solved.V[r][s])


def solve_streak_policy_values(
    day_slates: Sequence[Sequence[Candidate]],
    horizon: int,
    *,
    utility_name: str = "expected_final_streak",
    target: int | None = None,
    max_streak: int | None = None,
    reset_penalty: float = 0.0,
) -> SolvedPolicy:
    """Backward induction: each remaining day draws a slate uniformly from ``day_slates``.

    At every (r, s), the agent optimally chooses sit / best one / best two
    for that realized slate. Terminal utility is evaluated at r=0.
    """
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    if not day_slates:
        raise ValueError("day_slates must be non-empty")
    target_v = int(config.STREAK_POLICY_TARGET if target is None else target)
    # Streak can grow by at most 2 per day.
    max_s = int(max_streak if max_streak is not None else horizon * 2 + 2)
    max_s = max(max_s, target_v + 1, 2)

    V = [[0.0] * (max_s + 1) for _ in range(horizon + 1)]
    for s in range(max_s + 1):
        V[0][s] = terminal_utility(s, utility_name, target=target_v)

    def cont(r_left: int, streak: int) -> float:
        ss = min(max(0, streak), max_s)
        return V[r_left][ss]

    for r in range(1, horizon + 1):
        for s in range(max_s + 1):
            total = 0.0
            for slate in day_slates:
                decision = choose_action(
                    list(slate),
                    streak=s,
                    days_remaining=r - 1,  # continuation after today
                    utility_name=utility_name,
                    target=target_v,
                    reset_penalty=reset_penalty,
                    continuation=lambda nxt, _r=r - 1: cont(_r, nxt),
                    use_precomputed_continuation=True,
                )
                total += decision.expected_utility
            V[r][s] = total / len(day_slates)

    return SolvedPolicy(
        V=V,
        horizon=horizon,
        max_streak=max_s,
        utility_name=utility_name,
        target=target_v,
        day_samples=list(day_slates),
    )


def choose_action(
    candidates: Sequence[Candidate],
    *,
    streak: int,
    days_remaining: int,
    utility_name: str = "expected_final_streak",
    target: int | None = None,
    reset_penalty: float = 0.0,
    solved: SolvedPolicy | None = None,
    continuation: Callable[[int], float] | None = None,
    use_precomputed_continuation: bool = False,
    max_candidates: int | None = None,
) -> ActionDecision:
    """Compare sit / one / two for today's slate; return the best action.

    Continuation value for next-streak states comes from ``solved`` (DP),
    an explicit ``continuation`` callable, or terminal utility when
    ``days_remaining == 0``.
    """
    target_v = int(config.STREAK_POLICY_TARGET if target is None else target)
    streak = int(streak)
    days_remaining = int(days_remaining)

    def default_cont(nxt: int) -> float:
        if continuation is not None:
            return float(continuation(nxt))
        if solved is not None:
            return _continuation_lookup(solved, days_remaining, nxt)
        # Myopic terminal: evaluate utility at the resulting streak
        reset = nxt == 0 and streak > 0
        return terminal_utility(
            nxt, utility_name, target=target_v,
            reset_penalty=reset_penalty, previous_streak=streak, reset=reset,
        )

    cont = default_cont

    # Sit
    sit_eu = float(cont(streak))
    best = ActionDecision(
        action="sit",
        expected_utility=sit_eu,
        streak=streak,
        days_remaining=days_remaining,
        utility_name=utility_name,
        picks=[],
        details={"sit_eu": sit_eu},
    )

    pool = list(candidates)
    if max_candidates is None:
        max_candidates = int(config.STREAK_POLICY_MAX_CANDIDATES)
    if len(pool) > max_candidates:
        pool = sorted(
            pool,
            key=lambda c: c.resolved_outcomes().p_hit,
            reverse=True,
        )[:max_candidates]

    # One-pick options
    for c in pool:
        dist = one_pick_next_streak_distribution(streak, c.resolved_outcomes())
        eu = expected_utility_from_distribution(dist, cont)
        if eu > best.expected_utility + 1e-15:
            best = ActionDecision(
                action="one",
                expected_utility=eu,
                streak=streak,
                days_remaining=days_remaining,
                utility_name=utility_name,
                picks=[c],
                details={"dist": {str(k): v for k, v in dist.items()}, "p_hit": c.resolved_outcomes().p_hit},
            )

    # Two-pick options
    for a, b in combinations(pool, 2):
        dist, meta = two_pick_next_streak_distribution(streak, a, b)
        eu = expected_utility_from_distribution(dist, cont)
        if eu > best.expected_utility + 1e-15:
            best = ActionDecision(
                action="two",
                expected_utility=eu,
                streak=streak,
                days_remaining=days_remaining,
                utility_name=utility_name,
                picks=[a, b],
                details={
                    "dist": {str(k): v for k, v in dist.items()},
                    **meta,
                },
            )

    best.details["compared_sit_eu"] = sit_eu
    best.details["n_candidates"] = len(pool)
    return best


# ---------------------------------------------------------------------------
# Candidate frame helpers
# ---------------------------------------------------------------------------


def candidates_from_frame(df: pd.DataFrame) -> list[Candidate]:
    """Build Candidate list from a pick-pool / opportunity-like DataFrame."""
    if df is None or df.empty:
        return []
    rows = []
    for _, row in df.iterrows():
        final = row.get("Final_Hit_Probability", row.get("predicted_probability", np.nan))
        if pd.isna(final):
            continue
        p_appear = row.get("P_Appear", row.get("p_appear", np.nan))
        p_appear = None if pd.isna(p_appear) else float(p_appear)
        name = row.get("name")
        if pd.isna(name) if not isinstance(name, str) else False:
            name = ""
        if not name:
            nf = str(row.get("name_first", "") or "")
            nl = str(row.get("name_last", "") or "")
            name = f"{nf} {nl}".strip()
        rows.append(Candidate(
            key_mlbam=row.get("key_mlbam"),
            name=name,
            team=None if pd.isna(row.get("team", np.nan)) else row.get("team"),
            game_pk=row.get("game_pk", None),
            opponent=None if pd.isna(row.get("opponent", np.nan)) else row.get("opponent"),
            park_id=row.get("park_id", row.get("venue_team", None)),
            is_home=None if pd.isna(row.get("is_home", np.nan)) else float(row.get("is_home")),
            final_hit_probability=float(final),
            p_appear=p_appear,
        ))
    return rows


# ---------------------------------------------------------------------------
# Simulation / backtest metrics
# ---------------------------------------------------------------------------


def simulate_day_outcomes(
    picks: Sequence[Candidate],
    rng: np.random.Generator,
) -> list[str]:
    """Sample hit/miss/void for each pick independently (for Monte Carlo)."""
    out = []
    for c in picks:
        t = c.resolved_outcomes()
        out.append(rng.choice(["hit", "miss", "void"], p=t.as_array()))
    return out


def apply_action_outcomes(streak: int, action: str, outcomes: Sequence[str]) -> tuple[int, int]:
    """Return (new_streak, hits_added)."""
    if action == "sit" or not outcomes:
        return int(streak), 0
    if action == "one":
        nxt = apply_one_pick_transition(streak, outcomes[0])
        hits = 1 if outcomes[0] == "hit" else 0
        return nxt, hits
    if len(outcomes) < 2:
        # Degenerate two-pick
        return apply_action_outcomes(streak, "one", outcomes)
    nxt = apply_two_pick_transition(streak, outcomes[0], outcomes[1])
    if "miss" in outcomes:
        hits = 0
    else:
        hits = sum(1 for o in outcomes[:2] if o == "hit")
    return nxt, hits


@dataclass
class BacktestResult:
    action_counts: dict
    coverage: float
    day_survival_rate: float
    reset_rate: float
    average_hits_added: float
    final_streak: int
    longest_streak: int
    reach_target_rate: float
    daily: pd.DataFrame
    utility_name: str
    policy_name: str


def run_policy_backtest(
    daily_slates: Sequence[tuple[Any, Sequence[Candidate]]],
    *,
    policy: str = "dp",
    utility_name: str = "expected_final_streak",
    target: int | None = None,
    fixed_threshold: float | None = None,
    initial_streak: int = 0,
    horizon: int | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
    solved: SolvedPolicy | None = None,
) -> BacktestResult:
    """Walk forward through dated slates applying a policy.

    Policies:
      - ``dp``: choose_action with DP / myopic continuation
      - ``legacy_two``: always top-2 by p_hit (if available)
      - ``legacy_threshold``: sit unless best p_hit >= threshold; then take
        all candidates clearing threshold up to 2
    """
    target_v = int(config.STREAK_POLICY_TARGET if target is None else target)
    threshold = (
        float(config.DAILY_PICK_MIN_PROBABILITY)
        if fixed_threshold is None
        else float(fixed_threshold)
    )
    n_days = len(daily_slates)
    H = int(horizon if horizon is not None else n_days)
    streak = int(initial_streak)
    longest = streak
    rows = []
    action_counts = {"sit": 0, "one": 0, "two": 0}
    hits_added_total = 0
    played_days = 0
    survived = 0
    resets = 0

    # Optional DP solve on the same slates (i.i.d. approximation).
    if policy == "dp" and solved is None and n_days > 0:
        solved = solve_streak_policy_values(
            [list(s) for _, s in daily_slates],
            horizon=min(H, max(1, n_days)),
            utility_name=utility_name,
            target=target_v,
        )

    rng = np.random.default_rng(seed)

    for i, (date, slate) in enumerate(daily_slates):
        days_left = max(0, H - i - 1)
        slate = list(slate)

        if policy == "legacy_two":
            ordered = sorted(slate, key=lambda c: c.resolved_outcomes().p_hit, reverse=True)
            picks = ordered[:2]
            action = {0: "sit", 1: "one", 2: "two"}[len(picks)]
            decision = ActionDecision(
                action=action, expected_utility=float("nan"), streak=streak,
                days_remaining=days_left, utility_name=utility_name, picks=picks,
            )
        elif policy == "legacy_threshold":
            ordered = sorted(slate, key=lambda c: c.resolved_outcomes().p_hit, reverse=True)
            clearing = [c for c in ordered if c.resolved_outcomes().p_hit >= threshold][:2]
            action = {0: "sit", 1: "one", 2: "two"}[len(clearing)]
            decision = ActionDecision(
                action=action, expected_utility=float("nan"), streak=streak,
                days_remaining=days_left, utility_name=utility_name, picks=clearing,
            )
        else:
            decision = choose_action(
                slate, streak=streak, days_remaining=days_left,
                utility_name=utility_name, target=target_v, solved=solved,
            )

        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1

        # Realized outcomes: if candidates carry no labels, sample from triples.
        # Backtests that attach realized_outcome on Candidate.details aren't used;
        # sampling keeps the module self-contained. Callers can pass
        # Candidate with a fixed OutcomeTriple of 0/1 for deterministic paths.
        outcomes = simulate_day_outcomes(decision.picks, rng)
        prev = streak
        streak, hits = apply_action_outcomes(streak, decision.action, outcomes)
        hits_added_total += hits
        longest = max(longest, streak)
        reset = streak == 0 and prev > 0 and decision.action != "sit"
        if decision.action != "sit":
            played_days += 1
            if not reset:
                survived += 1
            else:
                resets += 1
        rows.append({
            "date": date,
            "action": decision.action,
            "streak_before": prev,
            "streak_after": streak,
            "hits_added": hits,
            "reset": reset,
            "n_picks": len(decision.picks),
            "expected_utility": decision.expected_utility,
            "outcomes": ",".join(outcomes),
        })

    daily = pd.DataFrame(rows)
    coverage = float(played_days / n_days) if n_days else float("nan")
    survival = float(survived / played_days) if played_days else float("nan")
    reset_rate = float(resets / played_days) if played_days else float("nan")
    avg_hits = float(hits_added_total / n_days) if n_days else float("nan")

    # Bootstrap reach-target rate via resampling day order of outcome draws
    reach_rate = float("nan")
    if n_bootstrap > 0 and n_days > 0:
        reaches = 0
        for b in range(n_bootstrap):
            s = int(initial_streak)
            order = rng.permutation(n_days)
            for j in order:
                row = daily.iloc[int(j)]
                # Re-draw from stored action's implied outcomes distribution
                # using the same policy actions (fixed path of actions).
                # Simpler: replay recorded outcomes with reshuffled days'
                # action+outcome pairs as a paired block surrogate.
                s, _ = apply_action_outcomes(
                    s, row["action"],
                    row["outcomes"].split(",") if row["outcomes"] else [],
                )
            if s >= target_v:
                reaches += 1
        reach_rate = reaches / n_bootstrap

    return BacktestResult(
        action_counts=action_counts,
        coverage=coverage,
        day_survival_rate=survival,
        reset_rate=reset_rate,
        average_hits_added=avg_hits,
        final_streak=int(streak),
        longest_streak=int(longest),
        reach_target_rate=reach_rate,
        daily=daily,
        utility_name=utility_name,
        policy_name=policy,
    )


def paired_bootstrap_policy_comparison(
    daily_slates: Sequence[tuple[Any, Sequence[Candidate]]],
    *,
    utility_name: str = "expected_final_streak",
    n_bootstrap: int = 500,
    seed: int = 0,
    metric: str = "final_streak",
) -> dict:
    """Paired date-block bootstrap: DP policy minus legacy_two."""
    rng = np.random.default_rng(seed)
    n = len(daily_slates)
    if n == 0:
        return {"metric": metric, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    def _metric(result: BacktestResult) -> float:
        if metric == "final_streak":
            return float(result.final_streak)
        if metric == "day_survival_rate":
            return float(result.day_survival_rate)
        if metric == "longest_streak":
            return float(result.longest_streak)
        if metric == "coverage":
            return float(result.coverage)
        raise ValueError(metric)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [daily_slates[int(i)] for i in idx]
        dp = run_policy_backtest(sample, policy="dp", utility_name=utility_name, seed=int(rng.integers(0, 1_000_000)))
        leg = run_policy_backtest(sample, policy="legacy_two", utility_name=utility_name, seed=int(rng.integers(0, 1_000_000)))
        diffs.append(_metric(dp) - _metric(leg))
    arr = np.asarray(diffs, dtype=float)
    return {
        "metric": metric,
        "mean_diff": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_bootstrap": n_bootstrap,
        "challenger": "dp",
        "champion": "legacy_two",
    }


# ---------------------------------------------------------------------------
# Shadow integration helpers
# ---------------------------------------------------------------------------


def resolve_streak_policy_mode(
    configured: str | None = None,
    *,
    force_live: bool = False,
) -> tuple[str, dict]:
    mode = configured if configured is not None else config.STREAK_POLICY_MODE
    if mode not in config.STREAK_POLICY_MODES:
        return "shadow", {
            "configured": mode,
            "effective": "shadow",
            "fallback_used": True,
            "fallback_reason": f"invalid_mode:{mode}",
        }
    if mode == "live" and not force_live:
        # Fail closed until an outer-fold promotion gate report exists.
        import json
        import os
        path = config.STREAK_POLICY_PROMOTION_GATE_REPORT_PATH
        if not path or not os.path.exists(path):
            return "shadow", {
                "configured": "live",
                "effective": "shadow",
                "fallback_used": True,
                "fallback_reason": "promotion_gate_missing",
            }
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            gate = report.get("promotion_gate") or {}
            if not gate.get("untouched_outer_fold_improvement"):
                return "shadow", {
                    "configured": "live",
                    "effective": "shadow",
                    "fallback_used": True,
                    "fallback_reason": "promotion_gate_failed",
                }
        except Exception as exc:
            return "shadow", {
                "configured": "live",
                "effective": "shadow",
                "fallback_used": True,
                "fallback_reason": f"promotion_gate_unreadable:{type(exc).__name__}",
            }
    return mode, {
        "configured": mode,
        "effective": mode,
        "fallback_used": False,
        "fallback_reason": None,
    }


def shadow_decision_record(
    decision: ActionDecision,
    *,
    date,
    current_streak: int,
    mode: str = "shadow",
) -> dict:
    rec = decision.to_dict()
    rec.update({
        "date": str(date),
        "current_streak": int(current_streak),
        "selection_mode": mode,
        "shadow": mode == "shadow",
    })
    return rec
