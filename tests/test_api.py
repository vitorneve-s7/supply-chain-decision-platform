import pytest

from src.inventory.inputs import build_inputs
from src.simulation.api import simulate_scenario
from src.simulation.run_twin import sample_items


def test_unknown_names_list_the_valid_ones():
    with pytest.raises(ValueError, match="choose from"):
        simulate_scenario("(s, Q)", "S0 normal")
    with pytest.raises(ValueError, match="S1 supplier outage"):
        simulate_scenario("(s, S)", "S9 meteor")


def test_returns_a_small_dict_of_plain_numbers():
    inputs = sample_items(build_inputs(), 20)
    out = simulate_scenario("(s, S)", "S1 supplier outage", seed=3, inputs=inputs)
    assert out["disruption_start"] is not None
    assert 0 <= out["fill_rate"] <= 1
    assert all(isinstance(v, (str, int, float)) for v in out.values())
