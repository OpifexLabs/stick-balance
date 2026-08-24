from stick_balance.pid import BEST_GAINS, PidGains, evaluate_gains


def test_best_gains_hold_still_on_the_logged_suite():
    metrics = evaluate_gains(BEST_GAINS)
    assert metrics.survived_frac == 1.0
    assert metrics.settle_abs_theta < 1e-4
    assert metrics.settle_abs_x < 0.005


def test_best_gains_beat_the_untuned_seed():
    seed = PidGains(
        kp_theta=42.0,
        ki_theta=0.25,
        kd_theta=9.0,
        kp_x=2.8,
        ki_x=0.04,
        kd_x=3.2,
    )
    assert evaluate_gains(BEST_GAINS).cost < evaluate_gains(seed).cost
