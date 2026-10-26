import pytest

from casingrel.units import PSI_PER_FT_PER_PPG, convert_spread, to_field


def test_plain_number_is_already_in_field_unit():
    assert to_field(25, "ft") == 25.0
    assert to_field(9.7, "ppg") == 9.7


def test_strings_convert():
    assert to_field("1000 m", "ft") == pytest.approx(3280.84, abs=0.01)
    assert to_field("244.475 mm", "inch") == pytest.approx(9.625, abs=1e-6)
    assert to_field("100 bar", "psi") == pytest.approx(1450.38, abs=0.01)
    assert to_field("24 degC", "degF") == pytest.approx(75.2)


def test_ppg_accepts_density_and_pressure_gradient():
    assert to_field("1.0 g/cm**3", "ppg") == pytest.approx(8.345, abs=0.001)
    assert to_field("0.52 psi/ft", "ppg") == pytest.approx(0.52 / PSI_PER_FT_PER_PPG)


def test_per_100ft_scale():
    assert to_field("3 delta_degC / (100 m)", "delta_degF/ft", 100.0) == pytest.approx(1.6459, abs=1e-4)


@pytest.mark.parametrize("bad", ["25", "25 psi", "abc", True])
def test_rejects_unitless_strings_wrong_dimension_and_junk(bad):
    with pytest.raises(ValueError):
        to_field(bad, "ft")


def test_spread_ignores_unit_offset():
    # A 2 degC standard deviation is 3.6 degF wide; the 32 degF offset must not apply.
    assert convert_spread(24.0, 2.0, "degC", "degF") == pytest.approx(3.6)
    assert convert_spread(1.0, 0.1, "g/cm**3", "ppg") == pytest.approx(0.8345, abs=1e-4)
