from src.simulation.scenarios import FIRST_START, LAST_START, SCENARIOS, scenario_config, start_day


def test_start_day_is_fixed_by_the_seed_and_in_range():
    days = [start_day(seed) for seed in range(50)]
    assert days == [start_day(seed) for seed in range(50)]  # same seed, same day
    assert min(days) >= FIRST_START and max(days) <= LAST_START
    assert len(set(days)) > 10  # and different seeds spread over the range


def test_normal_scenario_has_no_disruption():
    config = scenario_config("S0 normal", seed=3)
    assert config.start_day is None
    assert not any(config.disrupted(day) for day in range(140))


def test_each_scenario_switches_on_its_own_change():
    s1 = scenario_config("S1 supplier outage", seed=3)
    assert s1.supplier_down and s1.duration == 14
    assert s1.disrupted(s1.start_day) and not s1.disrupted(s1.start_day + 14)
    assert scenario_config("S2 demand spike", 3).demand_multiplier == 1.5
    assert scenario_config("S3 DC capacity cut", 3).dc_capacity_multiplier == 0.5
    assert scenario_config("S4 longer lead times", 3).store_lead_multiplier == 2.0
    assert len(SCENARIOS) == 5
