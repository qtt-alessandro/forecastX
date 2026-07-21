"""Example tests for pnl.py.

These FAIL right now on purpose -- the functions in src/pnl.py raise
NotImplementedError. Implement them until the bar goes green. That red->green
loop is the whole point of having tests; a test is just an executable spec.

Run from the project root:   pytest -q
"""
import numpy as np
import pytest

from src.pnl import pnl_eur, equivalent_full_cycles


def test_pnl_worked_example():
    # charge, idle, charge, discharge -- see the docstring in pnl.py
    prices = [10, 50, 20, 40]
    net_export = [-1, 0, -1, 1]
    assert pnl_eur(prices, net_export, dt_h=0.25) == pytest.approx(2.5)


def test_pnl_sign_convention():
    # selling 1 MW for one full hour at 100 EUR/MWh earns +100
    assert pnl_eur([100.0], [1.0], dt_h=1.0) == pytest.approx(100.0)
    # buying the same costs 100
    assert pnl_eur([100.0], [-1.0], dt_h=1.0) == pytest.approx(-100.0)


def test_pnl_doing_nothing_is_zero():
    assert pnl_eur([10, 20, 30], [0, 0, 0], dt_h=0.25) == pytest.approx(0.0)


@pytest.mark.parametrize("discharge, energy, expected", [
    ([20.0], 20.0, 1.0),        # one full capacity discharged -> 1 cycle
    ([10.0, 10.0], 20.0, 1.0),  # split over two steps -> still 1 cycle
    ([5.0, 5.0], 20.0, 0.5),
])
def test_equivalent_full_cycles(discharge, energy, expected):
    assert equivalent_full_cycles(discharge, energy) == pytest.approx(expected)
