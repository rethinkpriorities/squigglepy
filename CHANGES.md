## v0.14

* Nested sampling now works as intended.
* You can now use `>>` for pipes for distributions. For example, `sq.norm(1, 2) >> dist_ceil`
* Distributions can now be compared with `>`, `<`, `>=`, and `<=`.
* `dist_max` can be used to get the maximum value between two distributions. This family of functions are not evaluated until the distribution is sampled and they work with pipes.
* `dist_min` can be used to get the minimum value between two distributions.
* `dist_round` can be used to round the final output of a distribution. This makes the distribution discrete.
* `dist_ceil` can be used to ceiling round the final output of a distribution. This makes the distribution discrete.
* `dist_floor` can be used to floor round the final output of a distribution. This makes the distribution discrete.
* `lclip` can be used to clip a distribution to a lower bound. This is the same functionality that is available within the distribution and the `sample` method.
* `rclip` can be used to clip a distribution to an upper bound. This is the same functionality that is available within the distribution and the `sample` method.
* `clip` can be used to clip a distribution to both an upper bound and a lower bound. This is the same functionality that is available within the distribution and the `sample` method.
* `sample` can now be used directly on numbers. This makes `const` functionally obsolete, but `const` is maintained for backwards compatibility and in case it is useful.
* `sample(None)` now returns `None` instead of an error.

## v0.13

* Sample shorthand notation can go in either order. That is, `100 @ sq.norm(1, 2)` now works and is the same as `sq.norm(1, 2) @ 100`, which is the same as `sq.sample(sq.norm(1, 2), n=100)`.

## v0.12

* Distributions now implement math directly. That is, you can do things like `sq.norm(2, 3) + sq.norm(4, 5)`, whereas previously this would not work. Thanks to Dawn Drescher for helping me implement this.
* `~sq.norm(1, 2)` is now a shorthand for `sq.sample(sq.norm(1, 2))`. Thanks to Dawn Drescher for helping me implement this shorthand.
* `sq.norm(1, 2) @ 100` is now a shorthand for `sq.sample(sq.norm(1, 2), n=100)`

## v0.11

#### Distributions

* **[Breaking change]** `tdist` and `log_tdist` have been modified to better approximate the desired credible intervals.
* `tdist` now can be defined by just `t`, producing a classic t-distribution.
* `tdist` now has a default value for `t`: 1.
* Added `chisquare` distribution.
* `lognormal` now returns an error if it is defined with a zero or negative value.

#### Other

* All functions now have docstrings.
* Added `kelly` to calculate Kelly criterion for bet sizing with probabilities.
* Added `full_kelly`, `half_kelly`, `quarter_kelly` as helpful aliases.

## v0.10

* **[Breaking change]** `credibility` is now defined using a number out of 100 (e.g., `credibility=80` to define an 80% CI) rather than a decimal out of 1 (e.g., `credibility=0.8` to define an 80% CI).
* Distribution objects now print their parameters.


## v0.9

* `goemean` and `geomean_odds` now can take the nested-list-based and dictionary-based formats for passing weights.


## v0.8

#### Non-visible backend changes

* Distributions are now implemented as classes (rather than lists).

#### Bayesian library updates
* **[Breaking change]** `bayes.update` now updates normal distributions from the distribution rather than from samples.
* **[Breaking change]** `bayes.update` no longer takes a `type` parameter but can now infer the type from the passed distribution.
* **[Breaking change]** Corrected a bug in how `bayes.update` implemented `evidence_weight` when updating normal distributions.


## v0.7

#### Bugfixes

* Fixes an issue with sampling from the `bernoulli` distribution.
* Fixes a bug with the implementation of `lclip` and `rclip`.

#### New distributions

* Adds `discrete` to calculate a discrete distribution. Example: `discrete({'A': 0.3, 'B': 0.3, 'C': 0.4})` will return A 30% of the time, B 30% of the time, and C 40% of the time.
* Adds `poisson(lam)` to calculate a poisson distribution.
* Adds `gamma(size, scale)` to calculate a gamma distribution.

#### Bayesian library updates

* Adds `bayes.bayesnet` to do bayesian inferece (see README).
* `bayes.update` now can take an `evidence_weight` parameter. Typically this would be equal to the number of samples.
* **[Breaking change]** `bayes.bayes` has been renamed `bayes.simple_bayes`.

#### Other

* **[Breaking change]** `credibility`, which defines the size of the interval (e.g., `credibility=0.8` for an 80% CI), is now a property of the distribution rather than the sampler. That is, you should now call `sample(norm(1, 3, credibility=0.8))` whereas previously it was `sample(norm(1, 3), credibility=0.8)`. This will allow mixing of distributions with different credible ranges.
* **[Breaking change]** Numbers have been changed from functions to global variables. Use `thousand` or `K` instead of `thousand()` (old/deprecated).
* `sample` now has a nice progress reporter if `verbose=True`.
* The `exponential` distribution now implements `lclip` and `rclip`.
* The `mixture` distribution can infer equal weights if no weights are given.
* The `mixture` distribution can infer the last weight if the last weight is not given.
* `geomean` and `geomean_odds` can infer the last weight if the last weight is not given.
* You can use `flip_coin` and `roll_die(sides)` to flip a coin or roll a die.
* `event_happens` and `event` are aliases for `event_occurs`.
* `get_percentiles` will now cast output to `int` if `digits=0`.
* `get_log_percentiles` now has a default value for `percentiles`.
* You can now set the seed for the RNG using `sq.set_seed`.

#### Non-visible backend changes

* Now has tests via pytest.
* The random numbers now come from a numpy generator as opposed to the previous deprecated `np.random` methods.
* The `sample` module (containing the `sample` function) has been renamed `samplers`.


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

* **[Breaking change]** Change `distributed_log` to `mixture` (to follow Squiggle) and allow it to implement any sub-distribution.
* **[Breaking change]** Changed library to single import.
* **[Breaking change]** Remove `weighted_log` as a distribution.


## v0.1

* Initial library

