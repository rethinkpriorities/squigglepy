import pytest
import numpy as np

from ..squigglepy.utils import (_process_weights_values, event_occurs, event_happens,
                                event, get_percentiles, get_log_percentiles, geomean,
                                p_to_odds, odds_to_p, geomean_odds, laplace, roll_die,
                                flip_coin)
from ..squigglepy.rng import set_seed


def test_process_weights_values_simple_case():
    test = _process_weights_values([0.1, 0.9], [2, 3])
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_numpy_arrays():
    test = _process_weights_values(np.array([0.1, 0.9]), np.array([2, 3]))
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_length_one():
    test = _process_weights_values([1], [2])
    expected = ([1], [2])
    assert test == expected


def test_process_weights_values_weight_inference():
    test = _process_weights_values([0.9], [2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_not_list():
    test = _process_weights_values(0.9, [2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_no_weights():
    test = _process_weights_values(None, [2, 3])
    expected = ([0.5, 0.5], [2, 3])
    assert test == expected


def test_process_weights_values_weight_inference_no_weights_len4():
    test = _process_weights_values(None, [2, 3, 4, 5])
    expected = ([0.25, 0.25, 0.25, 0.25], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_weights_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values('error', [2, 3])
    assert 'passed weights must be a list' in str(excinfo.value)


def test_process_weights_values_values_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values([0.1, 0.9], 'error')
    assert 'passed values must be a list' in str(excinfo.value)


def test_process_weights_values_weights_must_sum_to_1_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values([0.2, 0.9], [2, 3])
    assert 'weights don\'t sum to 1 - they sum to 1.1' in str(excinfo.value)


def test_process_weights_values_length_mismatch_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values([0.1, 0.9], [2, 3, 4])
    assert 'weights and values not same length' in str(excinfo.value)


def test_event_occurs():
    set_seed(42)
    assert event_occurs(0.9)
    set_seed(42)
    assert not event_occurs(0.1)


def test_event_happens():
    set_seed(42)
    assert event_happens(0.9)
    set_seed(42)
    assert not event_happens(0.1)


def test_event():
    set_seed(42)
    assert event(0.9)
    set_seed(42)
    assert not event(0.1)


def test_get_percentiles():
    test = get_percentiles(range(1, 901))
    expected = {1: 9.99, 5: 45.95, 10: 90.9, 20: 180.8,
                30: 270.7, 40: 360.6, 50: 450.5,
                60: 540.4, 70: 630.3, 80: 720.2,
                90: 810.1, 95: 855.05, 99: 891.01}
    assert test == expected


def test_get_percentiles_change_percentiles():
    test = get_percentiles(range(1, 901), percentiles=[20, 80])
    expected = {20: 180.8, 80: 720.2}
    assert test == expected

    test = get_percentiles(range(1, 901), percentiles=[25, 75])
    expected = {25: 225.75, 75: 675.25}
    assert test == expected


def test_get_percentiles_reverse():
    test = get_percentiles(range(1, 901), percentiles=[20, 80], reverse=True)
    expected = {20: 720.2, 80: 180.8}
    assert test == expected


def test_get_percentiles_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=1)
    expected = {25: 225.8, 75: 675.2}
    assert test == expected


def test_get_percentiles_zero_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=0)
    expected = {25: 225, 75: 675}
    assert test == expected
    assert isinstance(expected[25], int)
    assert isinstance(expected[75], int)


def test_get_log_percentiles():
    test = get_log_percentiles([10 ** x for x in range(1, 10)])
    expected = {1: '10^1.2', 5: '10^1.7', 10: '10^1.9', 20: '10^2.8',
                30: '10^3.7', 40: '10^4.4', 50: '10^5.0', 60: '10^5.9',
                70: '10^6.8', 80: '10^7.7', 90: '10^8.4',
                95: '10^8.8', 99: '10^9.0'}
    assert test == expected


def test_get_log_percentiles_change_percentiles():
    test = get_log_percentiles([10 ** x for x in range(1, 10)], percentiles=[20, 80])
    expected = {20: '10^2.8', 80: '10^7.7'}
    assert test == expected


def test_get_log_percentiles_reverse():
    test = get_log_percentiles([10 ** x for x in range(1, 10)],
                               percentiles=[20, 80],
                               reverse=True)
    expected = {20: '10^7.7', 80: '10^2.8'}
    assert test == expected


def test_get_log_percentiles_no_display():
    test = get_log_percentiles([10 ** x for x in range(1, 10)],
                               percentiles=[20, 80],
                               display=False)
    expected = {20: 2.8, 80: 7.7}
    assert test == expected


def test_get_log_percentiles_zero_digits():
    test = get_log_percentiles([10 ** x for x in range(1, 10)],
                               percentiles=[20, 80],
                               display=False,
                               digits=0)
    expected = {20: 3, 80: 8}
    assert test == expected


def test_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.26


def test_geomean_numpy():
    assert round(geomean(np.array([0.1, 0.2, 0.3, 0.4, 0.5])), 2) == 0.26


def test_weighted_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5],
                         weights=[0.5, 0.1, 0.1, 0.1, 0.2]), 2) == 0.19


def test_p_to_odds():
    assert round(p_to_odds(0.1), 2) == 0.11


def test_odds_to_p():
    assert round(odds_to_p(1/9), 2) == 0.1


def test_geomean_odds():
    assert round(geomean_odds([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.28


def test_laplace_simple():
    test = laplace(0, 1)
    expected = 1/3
    assert test == expected


def test_laplace_s_is_1():
    test = laplace(1, 1)
    expected = 2/3
    assert test == expected


def test_laplace_s_gt_n():
    with pytest.raises(ValueError) as excinfo:
        laplace(3, 2)
    assert '`s` cannot be greater than `n`' in str(excinfo.value)


def test_time_invariant_laplace_zero_s():
    assert laplace(s=0, time_passed=2, time_remaining=2) == 0.5


def test_time_invariant_laplace_one_s_time_fixed():
    assert laplace(s=1, time_passed=2, time_remaining=2, time_fixed=True) == 0.75


def test_time_invariant_laplace_one_s_time_variable():
    assert laplace(s=1, time_passed=2, time_remaining=2) == 0.5


def test_time_invariant_laplace_infer_time_remaining():
    assert round(laplace(s=1, time_passed=2), 2) == 0.33


def test_time_invariant_laplace_two_s():
    assert laplace(s=2, time_passed=2, time_remaining=2) == 0.75


def test_laplace_only_s():
    with pytest.raises(ValueError) as excinfo:
        laplace(3)
    assert 'Must define `time_passed` or `n`' in str(excinfo.value)


def test_laplace_no_time_passed():
    with pytest.raises(ValueError) as excinfo:
        laplace(3, time_remaining=1)
    assert 'Must define `time_passed`' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        laplace(3, n=10, time_remaining=1)
    assert 'Must define `time_passed`' in str(excinfo.value)


def test_roll_die():
    set_seed(42)
    assert roll_die(6) == 5


def test_roll_die_different_sides():
    set_seed(42)
    assert roll_die(4) == 4


def test_roll_one_sided_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(1)
    assert 'cannot roll less than a 2-sided die' in str(excinfo.value)


def test_roll_nonint_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(2.5)
    assert 'can only roll an integer number of sides' in str(excinfo.value)


def test_roll_five_die():
    set_seed(42)
    assert list(roll_die(4, 4)) == [4, 2, 4, 3]


def test_flip_coin():
    set_seed(42)
    assert flip_coin() == 'heads'


def test_flip_five_coins():
    set_seed(42)
    assert flip_coin(5) == ['heads', 'tails', 'heads', 'heads', 'tails']
