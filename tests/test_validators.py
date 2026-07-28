"""Tests for `src/validators.py`.

`validate_complete_percentage_composition` guards a specific scientific claim:
when a paper reports a *complete* lipid composition on a percentage basis, the
stated fractions must actually add up. Anything else -- a molar-ratio basis, an
explicitly partial composition, an unknown basis -- is outside what the check
can decide, and the function must abstain by returning True rather than
inventing a failure.
"""

import pytest

from src.validators import validate_complete_percentage_composition


def check(percentages, **kwargs):
    kwargs.setdefault("composition_basis", "mol%")
    kwargs.setdefault("composition_is_complete", True)
    return validate_complete_percentage_composition(percentages, **kwargs)


# --------------------------------------------------------------------------- #
# Complete percentage compositions are actually summed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("basis", ["mol%", "weight%"])
def test_complete_composition_that_totals_100_passes(basis):
    assert check([50.0, 10.0, 38.5, 1.5], composition_basis=basis) is True


@pytest.mark.parametrize("total", [98.0, 100.0, 102.0])
def test_totals_on_the_tolerance_boundary_are_accepted(total):
    # Rounding in published tables rarely lands exactly on 100.
    assert check([total]) is True


@pytest.mark.parametrize("total", [97.9, 102.1, 0.0, 200.0])
def test_totals_outside_the_tolerance_are_rejected(total):
    assert check([total]) is False


def test_composition_that_totals_far_below_100_fails():
    assert check([50.0, 10.0, 20.0, 1.5]) is False


def test_tolerance_bounds_are_configurable():
    assert check([95.0], minimum_total=90.0, maximum_total=110.0) is True
    assert check([99.9], minimum_total=100.0, maximum_total=100.0) is False


def test_single_component_declared_as_the_whole_formulation_passes():
    assert check([100.0]) is True


def test_accepts_any_iterable_not_only_lists():
    assert check(iter([50.0, 50.0])) is True
    assert check((50.0, 10.0, 38.5, 1.5)) is True


# --------------------------------------------------------------------------- #
# Missing values in a composition claimed to be complete
# --------------------------------------------------------------------------- #


def test_complete_composition_with_a_missing_value_fails():
    # "Complete" plus an unknown percentage is self-contradictory, and the
    # remaining values must not be silently summed as if they were the whole.
    assert check([50.0, None, 38.5]) is False


def test_complete_composition_with_no_values_fails():
    assert check([]) is False


def test_complete_composition_of_only_missing_values_fails():
    assert check([None, None]) is False


# --------------------------------------------------------------------------- #
# Cases the check must abstain on
# --------------------------------------------------------------------------- #


def test_incomplete_composition_is_not_forced_to_total_100():
    assert check([50.0, 10.0], composition_is_complete=False) is True


def test_incomplete_composition_with_missing_values_still_abstains():
    assert check([50.0, None], composition_is_complete=False) is True


@pytest.mark.parametrize(
    "basis",
    ["molar_ratio", "mass_ratio", "mg/mL", "", "MOL%", "mol %", None],
)
def test_non_percentage_bases_are_out_of_scope(basis):
    # Only the two exact percentage bases are decidable; everything else --
    # including a differently-spelled variant -- must abstain rather than
    # guess that the numbers are percentages.
    assert check([50.0, 10.0], composition_basis=basis) is True


def test_basis_check_precedes_the_completeness_check():
    assert check([1.0, 1.0, 1.0], composition_basis="molar_ratio") is True
    assert check([], composition_basis="molar_ratio") is True


# --------------------------------------------------------------------------- #
# The input iterable is consumed exactly once
# --------------------------------------------------------------------------- #


def test_generator_input_is_materialised_before_being_inspected():
    calls = []

    def source():
        for value in (50.0, 10.0, 38.5, 1.5):
            calls.append(value)
            yield value

    assert check(source()) is True
    assert calls == [50.0, 10.0, 38.5, 1.5]
