import pytest
import numpy as np

from datetime import datetime, timedelta
from ..squigglepy.utils import (_process_weights_values, event_occurs, event_happens,
                                event, get_percentiles, get_log_percentiles, geomean,
                                p_to_odds, odds_to_p, geomean_odds, laplace, roll_die,
                                flip_coin, kelly, full_kelly, half_kelly, quarter_kelly,
                                one_in)
from ..squigglepy.rng import set_seed
from ..squigglepy.distributions import bernoulli, beta, norm, dist_round


def test_process_weights_values_simple_case():
    test = _process_weights_values(weights=[0.1, 0.9], values=[2, 3])
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_numpy_arrays():
    test = _process_weights_values(weights=np.array([0.1, 0.9]), values=np.array([2, 3]))
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_length_one():
    test = _process_weights_values(weights=[1], values=[2])
    expected = ([1], [2])
    assert test == expected


def test_process_weights_values_alt_format():
    test = _process_weights_values(values=[[0.1, 2],
                                           [0.2, 3],
                                           [0.3, 4],
                                           [0.4, 5]])
    expected = ([0.1, 0.2, 0.3, 0.4], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_alt2_format():
    test = _process_weights_values(values={2: 0.1,
                                           3: 0.2,
                                           4: 0.3,
                                           5: 0.4})
    expected = ([0.1, 0.2, 0.3, 0.4], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_dict_error():
    with pytest.raises(ValueError) as execinfo:
        _process_weights_values(weights=[0.1, 0.2, 0.3, 0.4],
                                values={2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4})
    assert 'cannot pass dict and weights separately' in str(execinfo.value)


def test_process_weights_values_entry_error():
    with pytest.raises(ValueError) as execinfo:
        _process_weights_values(values='troll')
    assert 'passed values must be a list, dict, or array' in str(execinfo.value)


def test_process_weights_values_weight_inference():
    test = _process_weights_values(weights=[0.9], values=[2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_not_list():
    test = _process_weights_values(weights=0.9, values=[2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_no_weights():
    test = _process_weights_values(values=[2, 3])
    expected = ([0.5, 0.5], [2, 3])
    assert test == expected


def test_process_weights_values_weight_inference_no_weights_len4():
    test = _process_weights_values(values=[2, 3, 4, 5])
    expected = ([0.25, 0.25, 0.25, 0.25], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_weights_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights='error', values=[2, 3])
    assert 'passed weights must be a list' in str(excinfo.value)


def test_process_weights_values_values_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.1, 0.9], values='error')
    assert 'passed values must be a list' in str(excinfo.value)


def test_process_weights_values_weights_must_sum_to_1_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.2, 0.9], values=[2, 3])
    assert 'weights don\'t sum to 1 - they sum to 1.1' in str(excinfo.value)


def test_process_weights_values_length_mismatch_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.1, 0.9], values=[2, 3, 4])
    assert 'weights and values not same length' in str(excinfo.value)


def test_process_weights_values_negative_weights():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[-0.1, 0.2, 0.9], values=[2, 3, 4])
    assert 'weight cannot be negative' in str(excinfo.value)


def test_process_weights_values_remove_zero_weights():
    test = _process_weights_values(weights=[0, 0.3, 0, 0.7, 0],
                                   values=[1, 2, 3, 4, 5])
    expected = ([0.3, 0.7], [2, 4])
    assert test == expected


def test_event_occurs():
    set_seed(42)
    assert event_occurs(0.9)
    set_seed(42)
    assert not event_occurs(0.1)


def test_event_occurs_can_handle_distributions():
    set_seed(42)
    assert event_occurs(bernoulli(0.9))


def test_event_occurs_can_handle_distributions2():
    set_seed(42)
    assert event_occurs(beta(10, 1))


def test_event_occurs_can_handle_distributions_callable():
    def get_p():
        return 0.9
    assert event_occurs(get_p)


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


def test_one_in():
    assert one_in(0.1) == "1 in 10"
    assert one_in(0.02) == "1 in 50"
    assert one_in(0.00002) == "1 in 50,000"


def test_one_in_w_rounding():
    assert one_in(0.1415) == "1 in 7"
    assert one_in(0.1415, digits=1) == "1 in 7.1"
    assert one_in(0.1415, digits=2) == "1 in 7.07"
    assert one_in(0.1415, digits=3) == "1 in 7.067"


def test_one_in_not_verbose():
    assert one_in(0.1415, digits=3, verbose=False) == 7.067


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


def test_get_percentiles_length_one():
    test = get_percentiles(range(1, 901), percentiles=[25], digits=1)
    assert test == 225.8
    test = get_percentiles(range(1, 901), percentiles=25, digits=1)
    assert test == 225.8


def test_get_percentiles_zero_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=0)
    expected = {25: 226, 75: 675}
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


def test_get_log_percentiles_length_one():
    test = get_log_percentiles([10 ** x for x in range(1, 10)],
                               percentiles=[20],
                               display=False,
                               digits=0)
    assert test == 3
    test = get_log_percentiles([10 ** x for x in range(1, 10)],
                               percentiles=20,
                               display=False,
                               digits=0)
    assert test == 3


def test_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.26


def test_geomean_numpy():
    assert round(geomean(np.array([0.1, 0.2, 0.3, 0.4, 0.5])), 2) == 0.26


def test_weighted_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5],
                         weights=[0.5, 0.1, 0.1, 0.1, 0.2]), 2) == 0.19


def test_weighted_geomean_alt_format():
    assert round(geomean([[0.5, 0.1],
                          [0.1, 0.2],
                          [0.1, 0.3],
                          [0.1, 0.4],
                          [0.2, 0.5]]), 2) == 0.19


def test_weighted_geomean_alt2_format():
    assert round(geomean({0.1: 0.5,
                          0.2: 0.1,
                          0.3: 0.1,
                          0.4: 0.1,
                          0.5: 0.2}), 2) == 0.19


def test_p_to_odds():
    assert round(p_to_odds(0.1), 2) == 0.11


def test_odds_to_p():
    assert round(odds_to_p(1/9), 2) == 0.1


def test_geomean_odds():
    assert round(geomean_odds([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.28


def test_geomean_odds_numpy():
    assert round(geomean_odds(np.array([0.1, 0.2, 0.3, 0.4, 0.5])), 2) == 0.28


def test_weighted_geomean_odds():
    assert round(geomean_odds([0.1, 0.2, 0.3, 0.4, 0.5],
                              weights=[0.5, 0.1, 0.1, 0.1, 0.2]), 2) == 0.2


def test_weighted_geomean_odds_alt_format():
    assert round(geomean_odds([[0.5, 0.1],
                               [0.1, 0.2],
                               [0.1, 0.3],
                               [0.1, 0.4],
                               [0.2, 0.5]]), 2) == 0.2


def test_weighted_geomean_odds_alt2_format():
    assert round(geomean_odds({0.1: 0.5,
                               0.2: 0.1,
                               0.3: 0.1,
                               0.4: 0.1,
                               0.5: 0.2}), 2) == 0.2


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


def test_roll_die_with_distribution():
    set_seed(42)
    assert (norm(2, 6) >> dist_round >> roll_die) == 2


def test_roll_one_sided_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(1)
    assert 'cannot roll less than a 2-sided die' in str(excinfo.value)


def test_roll_nonint_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(2.5)
    assert 'can only roll an integer number of sides' in str(excinfo.value)


def test_roll_nonint_n():
    with pytest.raises(ValueError) as excinfo:
        roll_die(6, 2.5)
    assert 'can only roll an integer number of times' in str(excinfo.value)


def test_roll_five_die():
    set_seed(42)
    assert list(roll_die(4, 4)) == [4, 2, 4, 3]


def test_flip_coin():
    set_seed(42)
    assert flip_coin() == 'heads'


def test_flip_five_coins():
    set_seed(42)
    assert flip_coin(5) == ['heads', 'tails', 'heads', 'heads', 'tails']


def test_kelly_market_price_error():
    for val in [0, 1, 2, -1]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=0.99, market_price=val)
        assert 'market_price must be >0 and <1' in str(execinfo.value)


def test_kelly_my_price_error():
    for val in [0, 1, 2, -1]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=val, market_price=0.99)
        assert 'my_price must be >0 and <1' in str(execinfo.value)


def test_kelly_deference_error():
    for val in [-1, 2]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=0.01, market_price=0.99, deference=val)
        assert 'deference must be >=0 and <=1' in str(execinfo.value)


def test_kelly_defaults():
    obj = kelly(my_price=0.99, market_price=0.01)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 0.99
    assert obj['current'] == 0
    assert obj['delta'] == 0.99
    assert obj['max_gain'] == 98.99
    assert obj['modeled_gain'] == 97.99
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_full_kelly():
    obj = full_kelly(my_price=0.99, market_price=0.01)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 0.99
    assert obj['current'] == 0
    assert obj['delta'] == 0.99
    assert obj['max_gain'] == 98.99
    assert obj['modeled_gain'] == 97.99
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_half_kelly():
    obj = half_kelly(my_price=0.99, market_price=0.01)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0.5
    assert obj['adj_price'] == 0.5
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.49
    assert obj['kelly'] == 0.495
    assert obj['target'] == 0.49
    assert obj['current'] == 0
    assert obj['delta'] == 0.49
    assert obj['max_gain'] == 49.49
    assert obj['modeled_gain'] == 24.5
    assert obj['expected_roi'] == 49
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_quarter_kelly():
    obj = quarter_kelly(my_price=0.99, market_price=0.01)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0.75
    assert obj['adj_price'] == 0.26
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.24
    assert obj['kelly'] == 0.247
    assert obj['target'] == 0.25
    assert obj['current'] == 0
    assert obj['delta'] == 0.25
    assert obj['max_gain'] == 24.75
    assert obj['modeled_gain'] == 6.13
    assert obj['expected_roi'] == 24.5
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_kelly_with_bankroll():
    obj = kelly(my_price=0.99, market_price=0.01, bankroll=1000)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 989.9
    assert obj['current'] == 0
    assert obj['delta'] == 989.9
    assert obj['max_gain'] == 98989.9
    assert obj['modeled_gain'] == 97990.1
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_kelly_with_current():
    obj = kelly(my_price=0.99, market_price=0.01, bankroll=1000, current=100)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 989.9
    assert obj['current'] == 100
    assert obj['delta'] == 889.9
    assert obj['max_gain'] == 98989.9
    assert obj['modeled_gain'] == 97990.1
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] is None
    assert obj['resolve_date'] is None


def test_kelly_with_resolve_date():
    one_year_from_today = datetime.now() + timedelta(days=365)
    one_year_from_today_str = one_year_from_today.strftime('%Y-%m-%d')
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=one_year_from_today_str)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 0.99
    assert obj['current'] == 0
    assert obj['delta'] == 0.99
    assert obj['max_gain'] == 98.99
    assert obj['modeled_gain'] == 97.99
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] == 99.258
    assert obj['resolve_date'] == datetime(one_year_from_today.year,
                                           one_year_from_today.month,
                                           one_year_from_today.day,
                                           0,
                                           0)


def test_kelly_with_resolve_date2():
    two_years_from_today = datetime.now() + timedelta(days=365*2)
    two_years_from_today_str = two_years_from_today.strftime('%Y-%m-%d')
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=two_years_from_today_str)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 0.99
    assert obj['current'] == 0
    assert obj['delta'] == 0.99
    assert obj['max_gain'] == 98.99
    assert obj['modeled_gain'] == 97.99
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] == 8.981
    assert obj['resolve_date'] == datetime(two_years_from_today.year,
                                           two_years_from_today.month,
                                           two_years_from_today.day,
                                           0,
                                           0)


def test_kelly_with_resolve_date0pt5():
    half_year_from_today = datetime.now() + timedelta(days=int(round(365*0.5)))
    half_year_from_today_str = half_year_from_today.strftime('%Y-%m-%d')
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=half_year_from_today_str)
    assert obj['my_price'] == 0.99
    assert obj['market_price'] == 0.01
    assert obj['deference'] == 0
    assert obj['adj_price'] == 0.99
    assert obj['delta_price'] == 0.98
    assert obj['adj_delta_price'] == 0.98
    assert obj['kelly'] == 0.99
    assert obj['target'] == 0.99
    assert obj['current'] == 0
    assert obj['delta'] == 0.99
    assert obj['max_gain'] == 98.99
    assert obj['modeled_gain'] == 97.99
    assert obj['expected_roi'] == 98
    assert obj['expected_arr'] == 10575.628
    assert obj['resolve_date'] == datetime(half_year_from_today.year,
                                           half_year_from_today.month,
                                           half_year_from_today.day,
                                           0,
                                           0)
