import time
import json
import numpy as np
import squigglepy as sq

from squigglepy.numbers import K, M
from squigglepy import bayes

from tqdm import tqdm


RUNS = 10*K


def _within(actual, expected, tolerance_ratio=None, abs_tolerance=None):
    if expected == 0 or actual == 0:
        ratio = None
    elif actual < expected:
        ratio = expected / actual
    else:
        ratio = actual / expected

    abs_diff = np.abs(actual - expected)

    if abs_tolerance is not None and abs_diff < abs_tolerance:
        return True
    elif tolerance_ratio is not None and ratio < tolerance_ratio:
        return True
    else:
        return False


def _mark_time(start, expected_sec, label, tolerance_ratio=1.05, tolerance_ms_threshold=5):
    end = time.time()
    delta_sec = end - start
    use_delta = delta_sec
    expected = expected_sec
    delta_label = 'sec'
    if delta_sec < 1:
        delta_ms = delta_sec * 1000
        expected = expected_sec * 1000
        use_delta = delta_ms
        delta_label = 'ms'
    use_delta = round(use_delta, 2)
    print('...{} in {}{} (expected ~{}{})'.format(label,
                                                  use_delta,
                                                  delta_label,
                                                  expected,
                                                  delta_label))
    if delta_label == 'ms':
        deviation = not _within(use_delta, expected, tolerance_ratio, tolerance_ms_threshold)
    else:
        deviation = not _within(use_delta, expected, tolerance_ratio)
    if deviation:
        print('!!! WARNING: Unexpected timing deviation')
    return {'timing(sec)': delta_sec, 'deviation': deviation}


def pct_of_pop_w_pianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01


def piano_tuners_per_piano():
    pianos_per_piano_tuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianos_per_piano_tuner)


def total_tuners_in_2022():
    return (sq.sample(pop_of_ny_2022) *
            pct_of_pop_w_pianos() *
            piano_tuners_per_piano())


def pop_at_time(t):
    avg_yearly_pct_change = sq.to(-0.01, 0.05)
    return sq.sample(pop_of_ny_2022) * ((sq.sample(avg_yearly_pct_change) + 1) ** t)


def total_tuners_at_time(t):
    return (pop_at_time(t) *
            pct_of_pop_w_pianos() *
            piano_tuners_per_piano())


def pop_at_time2(t):
    return pop_of_ny_2022 * ((sq.to(-0.01, 0.05) + 1) ** t)


def total_tuners_at_time2(t):
    piano_tuners_per_piano = 1 / sq.to(2*K, 50*K)
    pct_of_pop_w_pianos = sq.to(.2, 1) * 0.01
    return pop_at_time2(t) * pct_of_pop_w_pianos * piano_tuners_per_piano


def roll_die(sides, n=1):
    return sq.sample(sq.discrete(list(range(1, sides + 1))), n=n) if sides > 0 else None


def roll_die2(sides, n=1):
    return sq.discrete(list(range(1, sides + 1))) @ n if sides > 0 else None


def mammography(has_cancer):
    return sq.event(0.8 if has_cancer else 0.096)


def mammography_event():
    cancer = ~sq.bernoulli(0.01)
    return {'mammography': mammography(cancer),
            'cancer': cancer}


def p_alarm_goes_off(burglary, earthquake):
    if burglary and earthquake:
        return 0.95
    elif burglary and not earthquake:
        return 0.94
    elif not burglary and earthquake:
        return 0.29
    elif not burglary and not earthquake:
        return 0.001


def p_john_calls(alarm_goes_off):
    return 0.9 if alarm_goes_off else 0.05


def p_mary_calls(alarm_goes_off):
    return 0.7 if alarm_goes_off else 0.01


def alarm_net():
    burglary_happens = sq.event(p=0.001)
    earthquake_happens = sq.event(p=0.002)
    alarm_goes_off = sq.event(p_alarm_goes_off(burglary_happens,
                                               earthquake_happens))
    john_calls = sq.event(p_john_calls(alarm_goes_off))
    mary_calls = sq.event(p_mary_calls(alarm_goes_off))
    return {'burglary': burglary_happens,
            'earthquake': earthquake_happens,
            'alarm_goes_off': alarm_goes_off,
            'john_calls': john_calls,
            'mary_calls': mary_calls}


def monte_hall(door_picked, switch=False):
    doors = ['A', 'B', 'C']
    car_is_behind_door = ~sq.discrete(doors)
    reveal_door = [d for d in doors if d != door_picked and d != car_is_behind_door]
    reveal_door = ~sq.discrete(reveal_door)

    if switch:
        old_door_picked = door_picked
        door_picked = [d for d in doors if d != old_door_picked and d != reveal_door][0]

    won_car = (car_is_behind_door == door_picked)
    return won_car


def monte_hall_event():
    door = ~sq.discrete(['A', 'B', 'C'])
    switch = sq.event(0.5)
    return {'won': monte_hall(door_picked=door, switch=switch),
            'switched': switch}


def coins_and_dice():
    flip = sq.flip_coin()
    if flip == 'heads':
        dice_sides = 6
    else:
        dice_sides = ~sq.discrete([4, 6, 10, 20])
    return sq.roll_die(dice_sides)


def model():
    prior = sq.exponential(12)
    guess = sq.norm(10, 14)
    days = bayes.average(prior, guess, weights=[0.3, 0.7])

    def move_days(days):
        if days < 4 and sq.event(0.9):
            days = 4
        if days < 7 and sq.event(0.9):
            diff_days = 7 - days
            days = days + sq.norm(diff_days / 1.5, diff_days * 1.5)
        return days
        
    return sq.dist_fn(days, fn=move_days) >> sq.dist_round >> sq.lclip(3)


if __name__ == '__main__':
    print('Test 1 (PIANO TUNERS, NO TIME, LONG FORMAT)...')
    sq.set_seed(42)
    start1 = time.time()
    pop_of_ny_2022 = sq.to(8.1*M, 8.4*M)
    out = sq.get_percentiles(sq.sample(total_tuners_in_2022, n=100), digits=1)
    expected = {1: 0.6, 5: 0.9, 10: 1.1, 20: 2.0, 30: 2.6, 40: 3.1,
                50: 3.9, 60: 4.6, 70: 6.1, 80: 8.1, 90: 11.8,
                95: 19.6, 99: 36.8}
    if out != expected:
        print('ERROR 1')
        import pdb
        pdb.set_trace()
    _mark_time(start1, 0.033, 'Test 1 complete')


    print('Test 2 (PIANO TUNERS, TIME COMPONENT, LONG FORMAT)...')
    sq.set_seed(42)
    start2 = time.time()
    out = sq.get_percentiles(sq.sample(lambda: total_tuners_at_time(2030-2022), n=100), digits=1)
    expected = {1: 0.7, 5: 1.0, 10: 1.3, 20: 2.1, 30: 2.7, 40: 3.4, 50: 4.3, 60: 6.0,
                70: 7.4, 80: 9.4, 90: 14.1, 95: 19.6, 99: 24.4}

    if out != expected:
        print('ERROR 2')
        import pdb
        pdb.set_trace()
    _mark_time(start2, 0.046, 'Test 2 complete')


    print('Test 3 (PIANO TUNERS, TIME COMPONENT, SHORT FORMAT)...')
    sq.set_seed(42)
    start3 = time.time()
    out = sq.get_percentiles(total_tuners_at_time2(2030-2022) @ 100, digits=1)
    expected = {1: 0.5, 5: 0.6, 10: 1.1, 20: 1.5, 30: 1.8, 40: 2.4, 50: 3.1, 60: 4.4,
                70: 7.3, 80: 9.8, 90: 16.6, 95: 28.4, 99: 85.4}

    if out != expected:
        print('ERROR 3')
        import pdb
        pdb.set_trace()
    _mark_time(start3, 0.001, 'Test 3 complete')


    print('Test 4 (VARIOUS DISTRIBUTIONS, LONG FORMAT)...')
    sq.set_seed(42)
    start4 = time.time()
    sq.sample(sq.norm(1, 3))  # 90% interval from 1 to 3
    sq.sample(sq.norm(mean=0, sd=1))
    sq.sample(sq.norm(-1.67, 1.67))  # This is equivalent to mean=0, sd=1
    sq.sample(sq.norm(1, 3), n=100)
    sq.sample(sq.lognorm(1, 10))
    sq.sample(sq.tdist(1, 10, t=5))
    sq.sample(sq.triangular(1, 2, 3))
    sq.sample(sq.binomial(p=0.5, n=5))
    sq.sample(sq.beta(a=1, b=2))
    sq.sample(sq.bernoulli(p=0.5))
    sq.sample(sq.poisson(10))
    sq.sample(sq.chisquare(2))
    sq.sample(sq.gamma(3, 2))
    sq.sample(sq.exponential(scale=1))
    sq.sample(sq.discrete({'A': 0.1, 'B': 0.9}))
    sq.sample(sq.discrete({0: 0.1, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.15}))
    sq.sample(sq.discrete([[0.1,  0],
                           [0.3,  1],
                           [0.3,  2],
                           [0.15, 3],
                           [0.15, 4]]))
    sq.sample(sq.discrete([0, 1, 2]))
    sq.sample(sq.mixture([sq.norm(1, 3),
                          sq.norm(4, 10),
                          sq.lognorm(1, 10)],
                         [0.3, 0.3, 0.4]))
    sq.sample(sq.mixture([[0.3, sq.norm(1, 3)],
                          [0.3, sq.norm(4, 10)],
                          [0.4, sq.lognorm(1, 10)]]))
    sq.sample(lambda: sq.sample(sq.norm(1, 3)) + sq.sample(sq.norm(4, 5)), n=100)
    sq.sample(lambda: sq.sample(sq.norm(1, 3)) - sq.sample(sq.norm(4, 5)), n=100)
    sq.sample(lambda: sq.sample(sq.norm(1, 3)) * sq.sample(sq.norm(4, 5)), n=100)
    sq.sample(lambda: sq.sample(sq.norm(1, 3)) / sq.sample(sq.norm(4, 5)), n=100)
    sq.sample(sq.norm(1, 3, credibility=80))
    sq.sample(sq.norm(0, 3, lclip=0, rclip=5))
    sq.sample(sq.const(4))
    sq.sample(sq.zero_inflated(0.6, sq.norm(1, 2)))
    roll_die(sides=6, n=10)
    _mark_time(start4, 0.110, 'Test 4 complete')


    print('Test 5 (VARIOUS DISTRIBUTIONS, SHORT FORMAT)...')
    sq.set_seed(42)
    start5 = time.time()
    ~sq.norm(1, 3)
    ~sq.norm(mean=0, sd=1)
    ~sq.norm(-1.67, 1.67)
    sq.norm(1, 3) @ 100
    ~sq.lognorm(1, 10)
    ~sq.tdist(1, 10, t=5)
    ~sq.triangular(1, 2, 3)
    ~sq.binomial(p=0.5, n=5)
    ~sq.beta(a=1, b=2)
    ~sq.bernoulli(p=0.5)
    ~sq.poisson(10)
    ~sq.chisquare(2)
    ~sq.gamma(3, 2)
    ~sq.exponential(scale=1)
    ~sq.discrete({'A': 0.1, 'B': 0.9})
    ~sq.discrete({0: 0.1, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.15})
    ~sq.discrete([[0.1,  0],
                  [0.3,  1],
                  [0.3,  2],
                  [0.15, 3],
                  [0.15, 4]])
    ~sq.discrete([0, 1, 2])
    ~sq.mixture([sq.norm(1, 3),
                 sq.norm(4, 10),
                 sq.lognorm(1, 10)],
                [0.3, 0.3, 0.4])
    ~sq.mixture([[0.3, sq.norm(1, 3)],
                 [0.3, sq.norm(4, 10)],
                 [0.4, sq.lognorm(1, 10)]])
    ~sq.norm(1, 3) + ~sq.norm(4, 5)
    ~sq.norm(1, 3) - ~sq.norm(4, 5)
    ~sq.norm(1, 3) / ~sq.norm(4, 5)
    ~sq.norm(1, 3) * ~sq.norm(4, 5)
    ~(sq.norm(1, 3) + ~sq.norm(4, 5))
    ~(sq.norm(1, 3) - ~sq.norm(4, 5))
    ~(sq.norm(1, 3) / ~sq.norm(4, 5))
    ~(sq.norm(1, 3) * ~sq.norm(4, 5))
    (sq.norm(1, 3) + ~sq.norm(4, 5)) @ 100
    (sq.norm(1, 3) - ~sq.norm(4, 5)) @ 100
    (sq.norm(1, 3) / ~sq.norm(4, 5)) @ 100
    (sq.norm(1, 3) * ~sq.norm(4, 5)) @ 100
    ~sq.norm(1, 3, credibility=80)
    ~sq.norm(0, 3, lclip=0, rclip=5)
    ~sq.const(4)
    ~sq.zero_inflated(0.6, sq.norm(1, 2))
    roll_die2(sides=6, n=10)
    _mark_time(start5, 0.005, 'Test 5 complete')


    print('Test 6 (MAMMOGRAPHY BAYES)...')
    sq.set_seed(42)
    start6 = time.time()
    out = bayes.bayesnet(mammography_event,
                         find=lambda e: e['cancer'],
                         conditional_on=lambda e: e['mammography'],
                         memcache=False,
                         n=RUNS)
    expected = 0.09
    if round(out, 2) != expected:
        print('ERROR 6')
        import pdb
        pdb.set_trace()
    test_6_mark = _mark_time(start6, 0.186, 'Test 6 complete')


    print('Test 7 (SIMPLE BAYES)...')
    sq.set_seed(42)
    start7 = time.time()
    out = bayes.simple_bayes(prior=0.01, likelihood_h=0.8, likelihood_not_h=0.096)
    expected = None
    if round(out, 2) != 0.08:
        print('ERROR 7')
        import pdb
        pdb.set_trace()
    _mark_time(start7, 0.00001, 'Test 7 complete')


    print('Test 8 (BAYESIAN UPDATE)...')
    sq.set_seed(42)
    start8 = time.time()
    prior = sq.norm(1, 5)
    evidence = sq.norm(2, 3)
    posterior = bayes.update(prior, evidence)
    if round(posterior.mean, 2) != 2.53 and round(posterior.sd, 2) != 0.3:
        print('ERROR 8')
        import pdb
        pdb.set_trace()
    _mark_time(start8, 0.0004, 'Test 8 complete')


    print('Test 9 (BAYESIAN AVERAGE)...')
    sq.set_seed(42)
    start9 = time.time()
    average = bayes.average(prior, evidence)
    average_samples = sq.sample(average, n=K)
    out = (np.mean(average_samples), np.std(average_samples))
    if round(out[0], 2) != 2.75 and round(out[1], 2) != 0.94:
        print('ERROR 9')
        import pdb
        pdb.set_trace()
    _mark_time(start9, 0.019, 'Test 9 complete')


    print('Test 10 (ALARM NET)...')
    sq.set_seed(42)
    start10 = time.time()
    out = bayes.bayesnet(alarm_net,
                         n=RUNS * 3,
                         find=lambda e: (e['mary_calls'] and e['john_calls']),
                         conditional_on=lambda e: e['earthquake'])
    if round(out, 2) != 0.19:
        print('ERROR 10')
        import pdb
        pdb.set_trace()
    _mark_time(start10, 0.68, 'Test 10 complete')


    print('Test 11 (ALARM NET II)...')
    sq.set_seed(42)
    start11 = time.time()
    out = bayes.bayesnet(alarm_net,
                         n=RUNS * 3,
                         find=lambda e: e['burglary'],
                         conditional_on=lambda e: (e['mary_calls'] and e['john_calls']))
    if round(out, 2) != 0.35:
        print('ERROR 11')
        import pdb
        pdb.set_trace()
    _mark_time(start11, 0.0025, 'Test 11 complete')


    print('Test 12 (MONTE HALL)...')
    sq.set_seed(42)
    start12 = time.time()
    out = bayes.bayesnet(monte_hall_event,
                         find=lambda e: e['won'],
                         conditional_on=lambda e: e['switched'],
                         n=RUNS)
    if round(out, 2) != 0.67:
        print('ERROR 12')
        import pdb
        pdb.set_trace()
    _mark_time(start12, 1.26, 'Test 12 complete')


    print('Test 13 (MONTE HALL II)...')
    sq.set_seed(42)
    start13 = time.time()
    out = bayes.bayesnet(monte_hall_event,
                         find=lambda e: e['won'],
                         conditional_on=lambda e: not e['switched'],
                         n=RUNS)
    if round(out, 2) != 0.34:
        print('ERROR 13')
        import pdb
        pdb.set_trace()
    _mark_time(start13, 0.003, 'Test 13 complete')


    print('Test 14 (COINS AND DICE)...')
    sq.set_seed(42)
    start14 = time.time()
    out = bayes.bayesnet(coins_and_dice,
                         find=lambda e: e == 6,
                         n=RUNS)
    if round(out, 2) != 0.12:
        print('ERROR 14')
        import pdb
        pdb.set_trace()
    _mark_time(start14, 1.24, 'Test 14 complete')


    print('Test 15 (PIPES)...')
    sq.set_seed(42)
    start15 = time.time()
    samples = sq.sample(model, n=1000)
    if not all(isinstance(s, np.int64) for s in samples):
        print('ERROR 15')
        import pdb
        pdb.set_trace()
    _mark_time(start15, 0.247, 'Test 15 complete')


    print('Test 16 (T TEST)...')
    sq.set_seed(42)
    start16 = time.time()
# TODO: Accuracy with t<20
    ts = [20, 40, 50]
    vals = [[1, 10], [0, 3], [-4, 4], [5, 10], [100, 200]]
    credibilities = [80, 90]
    tqdm_ = tqdm(total=len(ts) * len(vals) * len(credibilities) * 2)
    for t in ts:
        for val in vals:
            for credibility in credibilities:
                for dist in [sq.tdist, sq.log_tdist]:
                    dist = dist(val[0], val[1], t, credibility=credibility)
                    if not (dist.type == 'log_tdist' and val[0] < 1):
                        pctiles = sq.get_percentiles(dist @ (20*K),
                                                     percentiles=[(100 - credibility) / 2,
                                                                  100 - ((100 - credibility) / 2)])
                        tol = 140 / t if dist.type == 'log_tdist' else 1.35
                        if (not _within(pctiles[(100 - credibility) / 2], val[0], tol, tol) or
                            not _within(pctiles[100 - ((100 - credibility) / 2)], val[1], tol, tol)):
                            print('ERROR 16 on {}'.format(str(dist)))
                            print(pctiles)
                            import pdb
                            pdb.set_trace()
                    tqdm_.update(1)
    tqdm_.close()
    _mark_time(start16, 0.082, 'Test 16 complete')


    print('Test 17 (SPEED TEST, 10M SAMPLES)...')
    start17 = time.time()
    samps = (sq.norm(1, 3) + sq.norm(4, 5)) @ (10*M)
    if len(samps) != (10*M):
        print('ERROR ON 17')
        import pdb
        pdb.set_trace()
    _mark_time(start17, 0.327, 'Test 17 complete')


    print('Test 18 (LCLIP FIDELITY, 1M SAMPLES)...')
    start18 = time.time()
    dist = sq.mixture([[0.1, 0],
                       [0.8, sq.norm(0,3)],
                       [0.1, sq.norm(7,11)]], lclip=0)
    samps = dist @ (1*M)
    if any(samps < 0):
        print('ERROR ON 18')
        import pdb
        pdb.set_trace()
    _mark_time(start18, 18.9, 'Test 18 complete')


    print('Test 19 (RCLIP FIDELITY, 1M SAMPLES)...')
    start19 = time.time()
    dist = sq.mixture([[0.1, 0],
                       [0.1, sq.norm(0,3)],
                       [0.8, sq.norm(7,11)]], rclip=3)
    samps = dist @ (1*M)
    if any(samps > 3):
        print('ERROR ON 19')
        import pdb
        pdb.set_trace()
    test_19_mark = _mark_time(start19, 18.9, 'Test 19 complete')


    print('Test 20 (MULTICORE SAMPLE, 10M SAMPLES)...')
    start20 = time.time()
    dist = sq.mixture([[0.1, 0],
                       [0.1, sq.norm(0,3)],
                       [0.8, sq.norm(7,11)]], rclip=3)
    samps = sq.sample(dist, cores=7, n=10*M, verbose=True)
    if len(samps) != (10*M) or any(samps > 3):
        print('ERROR ON 20')
        import pdb
        pdb.set_trace()
    test_20_mark = _mark_time(start20, 108.2, 'Test 20 complete')
    print('1 core 10M RUNS expected {}sec'.format(round(test_19_mark['timing(sec)'] * 10, 1)))
    print('7 core 10M RUNS ideal {}sec'.format(round(test_19_mark['timing(sec)'] * 10 / 7, 1)))
    print('7 core 10M RUNS actual {}sec'.format(round(test_20_mark['timing(sec)'], 1)))


    print('Test 21 (MAMMOGRAPHY BAYES MULTICORE)...')
    sq.set_seed(42)
    start21 = time.time()
    out = bayes.bayesnet(mammography_event,
                         find=lambda e: e['cancer'],
                         conditional_on=lambda e: e['mammography'],
                         n=10*M,
                         verbose=True,
                         memcache=False,
                         cores=7)
    expected = 0.08
    if round(out, 2) != expected:
        print('ERROR ON 21')
        import pdb
        pdb.set_trace()
    test_21_mark = _mark_time(start21, 90.6, 'Test 21 complete')
    print('1 core 10M RUNS expected {}sec'.format(round(test_6_mark['timing(sec)'] * K, 1)))
    print('7 core 10M RUNS ideal {}sec'.format(round(test_6_mark['timing(sec)'] * K / 7, 1)))
    print('7 core 10M RUNS actual {}sec'.format(round(test_21_mark['timing(sec)'], 1)))

# END
    _mark_time(start1, 218.4, 'Integration tests complete')
    print('DONE! INTEGRATION TEST SUCCESS!')
