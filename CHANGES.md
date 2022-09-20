## v0.7

#### Bugfixes

* Fixes an issue with sampling from the `bernoulli` distribution.

#### New distributions

* Add `discrete` to calculate a discrete distribution. Example: `discrete({'A': 0.3, 'B': 0.3, 'C': 0.4})` will return A 30% of the time, B 30% of the time, and C 40% of the time.

#### Bayesian library updates

* Adds `bayes.bayesnet`.
* `bayes.update` now can take an `evidence_weight` parameter. Typically this would be equal to the number of samples.
* `bayes.bayes` has been renamed `bayes.simple_bayes`.


## v0.6

#### New distributions

* Add `triangular(left, mode, right)` to calculate a triangular distribution.
* Add `binomial(n, p)` to calculate a binomial distribution.
* Add `beta(a, b)` to calculate a beta distribution.
* Add `bernoulli(p)` to calculate a bernoulli distribution.
* Add `exponential(scale)` to calculate an exponential distribution.

#### New Bayesian library

* Add `bayes.update` to get a posterior distribution from a prior distribution and an evidence distribution.
* Add `bayes.average` to average distributions (via a mixture).

#### New utility functions

* Add `laplace` to calculate Laplace's Law of Succession. If `s` and `n` are passed, it will calculate `(s+1)/(n+2)`. If `s`, `time_passed`, and `time_remaining` are passed, it will use the [time invariant version](https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule). Use `time_fixed=True` for fixed time periods and `time_fixed=False` (default) otherwise.
* Add `geomean` to calculate the geometric mean.
* Add `p_to_odds` to convert probability to odds. Also `odds_to_p` to convert odds to probability.
* Add `geomean_odds` to calculate the geometric mean of odds, converted to and from probabilities.

#### Other

* If a distribution is defined with `sd` but not `mean`, `mean` will be inferred to be 0.
* `sample` can now take `lclip` and `rclip` directly, in addition to defining `lclip` and `rclip` on the distribution itself. If both are defined, the most restrictive of the two bounds will be used.


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

