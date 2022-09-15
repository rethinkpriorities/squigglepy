## v0.6

#### New distributions

* Add `binomial(n, p)` to calculate a binomial distribution.
* Add `beta(a, b)` to calculate a beta distribution.
* Add `bernoulli(p)` to calculate a bernoulli distribution.
* Add `exponential(scale)` to calculate an exponential distribution.

#### New Bayesian library

* Add `bayes.update` to get a posterior distribution from a prior distribution and an evidence distribution.

#### New utility functions

* Add `laplace` to calculate Laplace's Law of Succession. If `s` and `n` are passed, it will calculate `(s+1)/(n+2)`. If `s`, `time_passed`, and `time_remaining` are passed, it will use the [time invariant version](https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule). Use `time_fixed=True` for fixed time periods and `time_fixed=False` (default) otherwise.
* Add `geomean` to calculate the geometric mean.
* Add `p_to_odds` to convert probability to odds. Also `odds_to_p` to convert odds to probability.
* Add `geomean_odds` to calculate the geometric mean of odds, converted to and from probabilities.

#### Other

* If a distribution is defined with `sd` but not `mean`, `mean` will be inferred to be 0.

## v0.5

* Fix critical bug to `tdist` and `log_tdist` introduced in v0.3.

## v0.4

* Fix critical bug introduced in v0.3.

## v0.3

* Be able to define distributions using `mean` and `sd` instead of defining the interval.

## v0.2

* Change `distributed_log` to `mixture` (to follow Squiggle) and allow it to implement any sub-distribution.
* Changed library to single import.
* Remove `weighted_log` as a distribution.


## v0.1

* Initial library
