"""Hand-calculated streak policy transitions and sit/one/two decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, streak_policy as sp


def test_outcome_triple_sums_to_one():
    t = sp.outcome_triple_from_probabilities(0.5, p_appear=0.8)
    assert abs(t.p_hit + t.p_miss + t.p_void - 1.0) < 1e-12
    assert abs(t.p_hit - 0.5) < 1e-12
    assert abs(t.p_void - 0.2) < 1e-12
    assert abs(t.p_miss - 0.3) < 1e-12
    sp.assert_outcomes_sum_to_one(t)


def test_outcome_triple_appear_floor_at_hit():
    t = sp.outcome_triple_from_probabilities(0.9, p_appear=0.5)
    # Appear is floored up to hit mass so probabilities stay coherent
    assert t.p_hit == pytest.approx(0.9)
    assert t.p_void == pytest.approx(0.1)
    assert t.p_miss == pytest.approx(0.0)
    assert t.p_hit + t.p_miss + t.p_void == pytest.approx(1.0)


# --- Exact one-pick transitions -------------------------------------------------

@pytest.mark.parametrize("streak,outcome,expected", [
    (5, "hit", 6),
    (5, "miss", 0),
    (5, "void", 5),
    (0, "hit", 1),
    (0, "miss", 0),
    (0, "void", 0),
])
def test_one_pick_transitions_hand_calculated(streak, outcome, expected):
    assert sp.apply_one_pick_transition(streak, outcome) == expected


# --- Exact two-pick transitions -------------------------------------------------

@pytest.mark.parametrize("streak,a,b,expected", [
    (4, "hit", "hit", 6),
    (4, "hit", "void", 5),
    (4, "void", "hit", 5),
    (4, "void", "void", 4),
    (4, "miss", "hit", 0),
    (4, "hit", "miss", 0),
    (4, "miss", "miss", 0),
    (4, "miss", "void", 0),
    (4, "void", "miss", 0),
    (0, "hit", "hit", 2),
    (0, "void", "void", 0),
])
def test_two_pick_transitions_hand_calculated(streak, a, b, expected):
    assert sp.apply_two_pick_transition(streak, a, b) == expected


def test_one_pick_distribution_hand_calculated():
    triple = sp.OutcomeTriple(p_hit=0.5, p_miss=0.3, p_void=0.2)
    dist = sp.one_pick_next_streak_distribution(10, triple)
    assert dist[11] == pytest.approx(0.5)
    assert dist[0] == pytest.approx(0.3)
    assert dist[10] == pytest.approx(0.2)
    assert abs(sum(dist.values()) - 1.0) < 1e-12


def test_independent_two_pick_joint_sums_and_values():
    a = sp.OutcomeTriple(0.5, 0.3, 0.2)
    b = sp.OutcomeTriple(0.4, 0.4, 0.2)
    joint = sp.independent_two_pick_joint(a, b)
    assert abs(sum(joint.values()) - 1.0) < 1e-12
    assert joint[("hit", "hit")] == pytest.approx(0.5 * 0.4)
    assert joint[("miss", "void")] == pytest.approx(0.3 * 0.2)


def test_dependence_penalty_default_zero_is_identity():
    a = sp.Candidate(1, final_hit_probability=0.5, p_appear=0.8, team="NYY", game_pk=1)
    b = sp.Candidate(2, final_hit_probability=0.5, p_appear=0.8, team="NYY", game_pk=1)
    oa, ob = a.resolved_outcomes(), b.resolved_outcomes()
    joint = sp.independent_two_pick_joint(oa, ob)
    adjusted = sp.apply_conservative_dependence(joint, oa, ob, penalty=0.0)
    for k in joint:
        assert adjusted[k] == pytest.approx(joint[k])


def test_dependence_penalty_inflates_both_miss():
    a = sp.OutcomeTriple(0.4, 0.4, 0.2)
    b = sp.OutcomeTriple(0.4, 0.4, 0.2)
    joint = sp.independent_two_pick_joint(a, b)
    before = joint[("miss", "miss")]
    after = sp.apply_conservative_dependence(joint, a, b, penalty=0.5)
    assert after[("miss", "miss")] > before
    assert abs(sum(after.values()) - 1.0) < 1e-9


def test_policy_can_choose_one_instead_of_two():
    """High single-pick safety beats two high-miss picks (independence)."""
    safe = sp.Candidate(
        1, name="Safe",
        outcomes=sp.OutcomeTriple(p_hit=0.95, p_miss=0.05, p_void=0.0),
    )
    risky_a = sp.Candidate(
        2, name="RiskyA",
        outcomes=sp.OutcomeTriple(p_hit=0.55, p_miss=0.45, p_void=0.0),
    )
    risky_b = sp.Candidate(
        3, name="RiskyB",
        outcomes=sp.OutcomeTriple(p_hit=0.55, p_miss=0.45, p_void=0.0),
    )
    # Myopic terminal utility = final streak length
    decision = sp.choose_action(
        [safe, risky_a, risky_b],
        streak=5,
        days_remaining=0,
        utility_name="expected_final_streak",
    )
    assert decision.action == "one"
    assert decision.picks[0].key_mlbam == 1

    # Hand-check EU: one safe -> 0.95*6 + 0.05*0 = 5.7 (> sit=5)
    # two risky: P(+2)=0.55^2=0.3025 -> EU=0.3025*7 ≈ 2.12
    assert decision.expected_utility == pytest.approx(5.7)


def test_policy_can_choose_sit_when_both_options_are_poor():
    bad = sp.Candidate(
        1, outcomes=sp.OutcomeTriple(p_hit=0.2, p_miss=0.8, p_void=0.0),
    )
    decision = sp.choose_action(
        [bad],
        streak=20,
        days_remaining=0,
        utility_name="expected_final_streak",
    )
    # Sit EU = 20; play EU = 0.2*21 + 0.8*0 = 4.2
    assert decision.action == "sit"
    assert decision.expected_utility == pytest.approx(20.0)


def test_policy_can_choose_two_when_both_are_strong():
    a = sp.Candidate(1, outcomes=sp.OutcomeTriple(0.95, 0.05, 0.0))
    b = sp.Candidate(2, outcomes=sp.OutcomeTriple(0.95, 0.05, 0.0))
    decision = sp.choose_action(
        [a, b],
        streak=5,
        days_remaining=0,
        utility_name="expected_final_streak",
    )
    assert decision.action == "two"
    # P(+2)=0.95^2, P(reset)=1-0.9025; EU = 0.9025*7
    assert decision.expected_utility == pytest.approx(0.95 * 0.95 * 7.0)


def test_dp_solve_monotone_in_streak_for_expected_utility():
    slate = [
        sp.Candidate(1, outcomes=sp.OutcomeTriple(0.7, 0.2, 0.1)),
        sp.Candidate(2, outcomes=sp.OutcomeTriple(0.65, 0.25, 0.1)),
    ]
    solved = sp.solve_streak_policy_values([slate], horizon=5, utility_name="expected_final_streak")
    # Higher streak should be at least as valuable at any remaining horizon
    for r in range(solved.horizon + 1):
        for s in range(solved.max_streak):
            assert solved.V[r][s + 1] + 1e-9 >= solved.V[r][s]


def test_reach_target_utility():
    assert sp.terminal_utility(56, "probability_reach_57") == 0.0
    assert sp.terminal_utility(57, "probability_reach_57") == 1.0
    assert sp.terminal_utility(10, "probability_reach_target", target=10) == 1.0


def test_backtest_action_counts_and_headline_metrics():
    # Deterministic triples via extreme probabilities so sampling is stable
    days = []
    for d in range(8):
        date = pd.Timestamp("2026-06-01") + pd.Timedelta(days=d)
        days.append((date, [
            sp.Candidate(1, outcomes=sp.OutcomeTriple(0.99, 0.01, 0.0)),
            sp.Candidate(2, outcomes=sp.OutcomeTriple(0.98, 0.02, 0.0)),
        ]))
    result = sp.run_policy_backtest(
        days, policy="legacy_two", utility_name="expected_final_streak",
        n_bootstrap=20, seed=0,
    )
    assert result.action_counts["two"] == 8
    assert result.coverage == 1.0
    assert result.longest_streak >= result.final_streak
    assert "sit" in result.action_counts


def test_legacy_threshold_can_sit():
    days = [(pd.Timestamp("2026-06-01"), [
        sp.Candidate(1, outcomes=sp.OutcomeTriple(0.50, 0.50, 0.0)),
    ])]
    result = sp.run_policy_backtest(
        days, policy="legacy_threshold", fixed_threshold=0.77,
        utility_name="expected_final_streak", seed=0,
    )
    assert result.action_counts["sit"] == 1
    assert result.coverage == 0.0


def test_shadow_mode_default_and_live_gate_fallback():
    assert config.STREAK_POLICY_MODE == "shadow"
    mode, meta = sp.resolve_streak_policy_mode("live", force_live=False)
    assert mode == "shadow"
    assert meta["fallback_used"] is True


def test_candidates_from_frame():
    df = pd.DataFrame([
        {
            "key_mlbam": 1, "name": "A", "team": "NYY", "game_pk": 10,
            "Final_Hit_Probability": 0.6, "P_Appear": 0.9,
        },
        {
            "key_mlbam": 2, "name_first": "B", "name_last": "Bee", "team": "BOS",
            "game_pk": 10, "Final_Hit_Probability": 0.55, "P_Appear": 0.85,
        },
    ])
    cands = sp.candidates_from_frame(df)
    assert len(cands) == 2
    assert cands[0].resolved_outcomes().p_hit == pytest.approx(0.6)
    assert cands[0].resolved_outcomes().p_void == pytest.approx(0.1)


def test_config_dependence_defaults_are_zero():
    assert config.STREAK_POLICY_DEPENDENCE_SAME_GAME == 0.0
    assert config.STREAK_POLICY_DEPENDENCE_SAME_TEAM == 0.0
    assert config.STREAK_POLICY_DEPENDENCE_SHARED_PARK == 0.0
    assert config.STREAK_POLICY_DEPENDENCE_OPPOSING_SAME_GAME == 0.0
