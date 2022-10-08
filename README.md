## Squigglepy: Implementation of Squiggle in Python

[Squiggle](https://www.squiggle-language.com/) is a "simple programming language for intuitive probabilistic estimation". It serves as its own standalone programming language with its own syntax, but it is implemented in JavaScript. I like the features of Squiggle and intend to use it frequently, but I also sometimes want to use similar functionalities in Python, especially alongside other Python statistical programming packages like Numpy, Pandas, and Matplotlib. The **squigglepy** package here implements many Squiggle-like functionalities in Python.


## Installation

`pip3 install squigglepy`


## Usage

### Core Features

Here's the Squigglepy implementation of [the example from Squiggle Docs](https://www.squiggle-language.com/docs/Overview):

```Python
import squigglepy as sq
from squigglepy.numbers import K, M

pop_of_ny_2022 = sq.to(8.1*M, 8.4*M) # This means that you're 90% confident the value is between 8.1 and 8.4 Million.

def pct_of_pop_w_pianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01 # We assume there are almost no people with multiple pianos

def piano_tuners_per_piano():
    pianos_per_piano_tuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianos_per_piano_tuner)

def total_tuners_in_2022():
    return (sq.sample(pop_of_ny_2022) *
            pct_of_pop_w_pianos() *
            piano_tuners_per_piano())

sq.get_percentiles(sq.sample(total_tuners_in_2022, n=1000))
```

And the version from the Squiggle doc that incorporates time:

```Python
import squigglepy as sq
from squigglepy.numbers import K, M

pop_of_ny_2022 = sq.to(8.1*M, 8.4*M)

def pct_of_pop_w_pianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def pct_of_pop_w_pianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def piano_tuners_per_piano():
    pianos_per_piano_tuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianos_per_piano_tuner)

# Time in years after 2022
def pop_at_time(t):
    avg_yearly_pct_change = sq.to(-0.01, 0.05) # We're expecting NYC to continuously grow with an mean of roughly between -1% and +4% per year
    return sq.sample(pop_of_ny_2022) * ((sq.sample(avg_yearly_pct_change) + 1) ** t)

def total_tuners_at_time(t):
    return (pop_at_time(t) *
            pct_of_pop_w_pianos() *
            piano_tuners_per_piano())

# Get total piano tuners at 2030
sq.get_percentiles(sq.sample(lambda: total_tuners_at_time(2030-2022), n=1000))
```

**WARNING:** Be careful about dividing by `K`, `M`, etc. `1/2*K` = 500 in Python. Use `1/(2*K)` instead to get the expected outcome.

### Additional Features

```Python
import squigglepy as sq

# Normal distribution
sq.sample(sq.norm(1, 3))  # 90% interval from 1 to 3

# Distribution can be sampled with mean and sd too
sq.sample(sq.norm(mean=0, sd=1))
sq.sample(sq.norm(-1.67, 1.67))  # This is equivalent to mean=0, sd=1

# Get more than one sample
sq.sample(sq.norm(1, 3), n=100)

# Other distributions exist
sq.sample(sq.lognorm(1, 10))
sq.sample(sq.tdist(1, 10, t=5))
sq.sample(sq.triangular(1, 2, 3))
sq.sample(sq.binomial(p=0.5, n=5))
sq.sample(sq.beta(a=1, b=2))
sq.sample(sq.bernoulli(p=0.5))
sq.sample(sq.poisson(10))
sq.sample(sq.gamma(3, 2))
sq.sample(sq.exponential(scale=1))

# Discrete sampling
sq.sample(sq.discrete({'A': 0.1, 'B': 0.9}))

# Can return integers
sq.sample(sq.discrete({0: 0.1, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.15}))

# Alternate format (also can be used to return more complex objects)
sq.sample(sq.discrete([[0.1,  0],
                       [0.3,  1],
                       [0.3,  2],
                       [0.15, 3],
                       [0.15, 4]]))

sq.sample(sq.discrete([0, 1, 2])) # No weights assumes equal weights

# You can mix distributions together
sq.sample(sq.mixture([sq.norm(1, 3),
                      sq.norm(4, 10),
                      sq.lognorm(1, 10)],  # Distributions to mix
                     [0.3, 0.3, 0.4]))     # These are the weights on each distribution

# This is equivalent to the above, just a different way of doing the notation
sq.sample(sq.mixture([[0.3, sq.norm(1,3)],
                      [0.3, sq.norm(4,10)],
                      [0.4, sq.lognorm(1,10)]]))

# You can add and subtract distributions (a little less cool compared to native Squiggle unfortunately):
sq.sample(lambda: sq.sample(sq.norm(1,3)) + sq.sample(sq.norm(4,5)), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) - sq.sample(sq.norm(4,5)), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) * sq.sample(sq.norm(4,5)), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) / sq.sample(sq.norm(4,5)), n=100)

# You can change the CI from 90% (default) to 80%
sq.sample(sq.norm(1, 3, credibility=0.8))

# You can clip
sq.sample(sq.norm(0, 3, lclip=0, rclip=5)) # Sample norm with a 90% CI from 0-3, but anything lower than 0 gets clipped to 0 and anything higher than 5 gets clipped to 5.

# You can specify a constant (which can be useful for passing things into functions or mixtures)
sq.sample(sq.const(4)) # Always returns 4
```


### Rolling a Die

An example of how to use distributions to build tools:

```Python
import squigglepy as sq

def roll_die(sides, n=1):
    return sq.sample(sq.discrete(list(range(1, sides + 1))), n=n) if sides > 0 else None

roll_die(sides=6, n=10)
# [2, 6, 5, 2, 6, 2, 3, 1, 5, 2]
```

This is already included standard in the utils of this package. Use `sq.roll_die`.


### Bayesian inference

1% of women at age forty who participate in routine screening have breast cancer.
80% of women with breast cancer will get positive mammographies.
9.6% of women without breast cancer will also get positive mammographies.

A woman in this age group had a positive mammography in a routine screening.
What is the probability that she actually has breast cancer?

We can approximate the answer with a Bayesian network (uses rejection sampling):

```Python
import squigglepy as sq
from squigglepy import bayes
from squigglepy.numbers import M

def mammography(has_cancer):
    p = 0.8 if has_cancer else 0.096
    return bool(sq.sample(sq.bernoulli(p)))

def define_event():
    cancer = sq.sample(sq.bernoulli(0.01))    
    return({'mammography': mammography(cancer),
            'cancer': cancer})

bayes.bayesnet(define_event,
               find=lambda e: e['cancer'],
               conditional_on=lambda e: e['mammography'],
               n=1*M)
# 0.07723995880535531
```

Or if we have the information immediately on hand, we can directly calculate it. Though this doesn't work for very complex stuff.

```Python
from squigglepy import bayes
bayes.simple_bayes(prior=0.01, likelihood_h=0.8, likelihood_not_h=0.096)
# 0.07763975155279504
```

You can also make distributions and update them:

```Python
import matplotlib.pyplot as plt
import squigglepy as sq
from squigglepy import bayes
from squigglepy.numbers import K

print('Prior')
prior = sq.norm(1,5)
prior_samples = sq.sample(prior, n=10*K)
plt.hist(prior_samples, bins = 200)
plt.show()
print(sq.get_percentiles(prior_samples))
print('Prior Mean: {} SD: {}'.format(np.mean(prior_samples), np.std(prior_samples)))
print('-')

print('Evidence')
evidence = sq.norm(2,3)
evidence_samples = sq.sample(evidence, n=10*K)
plt.hist(evidence_samples, bins = 200)
plt.show()
print(sq.get_percentiles(evidence_samples))
print('Evidence Mean: {} SD: {}'.format(np.mean(evidence_samples), np.std(evidence_samples)))
print('-')

print('Posterior')
posterior = bayes.update(prior_samples, evidence_samples)
posterior_samples = sq.sample(posterior, n=10*K)
plt.hist(posterior_samples, bins = 200)
plt.show()
print(sq.get_percentiles(posterior_samples))
print('Posterior Mean: {} SD: {}'.format(np.mean(posterior_samples), np.std(posterior_samples)))

print('Average')
average = bayes.average(prior, evidence)
average_samples = sq.sample(average, n=10*K)
plt.hist(average_samples, bins = 200)
plt.show()
print(sq.get_percentiles(average_samples))
print('Average Mean: {} SD: {}'.format(np.mean(average_samples), np.std(average_samples)))
```


### Alarm net

This is the alarm network from [Bayesian Artificial Intelligence - Section 2.5.1](https://bayesian-intelligence.com/publications/bai/book/BAI_Chapter2.pdf):

> Assume your house has an alarm system against burglary.
>
> You live in the seismically active area and the alarm system can get occasionally set off by an earthquake.
>
> You have two neighbors, Mary and John, who do not know each other.
> If they hear the alarm they call you, but this is not guaranteed.
>
> The chance of a burglary on a particular day is 0.1%.
> The chance of an earthquake on a particular day is 0.2%.
>
> The alarm will go off 95% of the time with both a burglary and an earthquake, 94% of the time with just a burglary, 29% of the time with just an earthquake, and 0.1% of the time with nothing (total false alarm).
>
> John will call you 90% of the time when the alarm goes off. But on 5% of the days, John will just call to say "hi".
> Mary will call you 70% of the time when the alarm goes off. But on 1% of the days, Mary will just call to say "hi".


```Python
import squigglepy as sq
from squigglepy import bayes
from squigglepy.numbers import M

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

def define_event():
    burglary_happens = bool(sq.sample(sq.bernoulli(p=0.001)))
    earthquake_happens = bool(sq.sample(sq.bernoulli(p=0.002)))
    alarm_goes_off = bool(sq.sample(sq.bernoulli(p_alarm_goes_off(burglary_happens, earthquake_happens))))
    john_calls = bool(sq.sample(sq.bernoulli(p_john_calls(alarm_goes_off))))
    mary_calls = bool(sq.sample(sq.bernoulli(p_mary_calls(alarm_goes_off))))
    return {'burglary': burglary_happens,
            'earthquake': earthquake_happens,
            'alarm_goes_off': alarm_goes_off,
            'john_calls': john_calls,
            'mary_calls': mary_calls}

# What are the chances that both John and Mary call if an earthquake happens?
bayes.bayesnet(define_event,
               n=1*M,
               find=lambda e: (e['mary_calls'] and e['john_calls']),
               conditional_on=lambda e: e['earthquake'])
# Result will be ~0.19, though it varies because it is based on a random sample.
# This also may take a minute to run.

# If both John and Mary call, what is the chance there's been a burglary?
bayes.bayesnet(define_event,
               n=1*M,
               find=lambda e: e['burglary'],
               conditional_on=lambda e: (e['mary_calls'] and e['john_calls']))
# Result will be ~0.27, though it varies because it is based on a random sample.
# This will run quickly because there is a built-in cache.
# Use `cache=False` to not build a cache and `reload_cache=True` to recalculate the cache.
```

Note that the amount of Bayesian analysis that squigglepy can do is pretty limited. For more complex bayesian analysis, consider [sorobn](https://github.com/MaxHalford/sorobn), [pomegranate](https://github.com/jmschrei/pomegranate), [bnlearn](https://github.com/erdogant/bnlearn), or [pyMC](https://github.com/pymc-devs/pymc).


### A Demonstration of the Monte Hall Problem

```Python
import random
import squigglepy as sq
from squigglepy import bayes
from squigglepy.numbers import K, M, B, T


def monte_hall(door_picked, switch=False):
    doors = ['A', 'B', 'C']
    car_is_behind_door = random.choice(doors)
    reveal_door = random.choice([d for d in doors if d != door_picked and d != car_is_behind_door])
    
    if switch:
        old_door_picked = door_picked
        door_picked = [d for d in doors if d != old_door_picked and d != reveal_door][0]
        
    won_car = (car_is_behind_door == door_picked)
    return won_car 


def define_event():
    door = random.choice(['A', 'B', 'C'])
    switch = random.random() >= 0.5
    return {'won': monte_hall(door_picked=door, switch=switch),
            'switched': switch}

RUNS = 10*K
r = bayes.bayesnet(define_event,
                   find=lambda e: e['won'],
                   conditional_on=lambda e: e['switched'],
                   verbose=True,
                   n=RUNS)
print('Win {}% of the time when switching'.format(int(r * 100)))

r = bayes.bayesnet(define_event,
                   find=lambda e: e['won'],
                   conditional_on=lambda e: not e['switched'],
                   verbose=True,
                   n=RUNS)
print('Win {}% of the time when not switching'.format(int(r * 100)))

# Win 66% of the time when switching
# Win 34% of the time when not switching
```


### More complex coin/dice interactions

> Imagine that I flip a coin. If heads, I take a random die out of my blue bag. If tails, I take a random die out of my red bag.
> The blue bag contains only 6-sided dice. The red bag contains a 4-sided die, a 6-sided die, a 10-sided die, and a 20-sided die.
> I then roll the random die I took. What is the chance that I roll a 6?

```Python
import squigglepy as sq
from squigglepy.numbers import K, M, B, T
from squigglepy import bayes

def define_event():
    flip = sq.flip_coin()
    if flip == 'heads': # Blue bag
        dice_sides = 6
    else: # Red bag
        dice_sides = sq.sample(sq.discrete([4, 6, 10, 20]))
    return sq.roll_die(dice_sides)


bayes.bayesnet(define_event,
               find=lambda e: e == 6,
               verbose=True,
               n=100*K)
# This run for me returned 0.12306 which is pretty close to the correct answer of 0.12292
```

## Run tests

`rm -rf build; flake8; pytest; python3 tests/integration.py`

