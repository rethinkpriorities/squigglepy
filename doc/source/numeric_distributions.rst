Numeric Distributions
=====================

A ``NumericDistribution`` representats a probability distribution as a histogram
of values along with the probability mass near each value.

A ``NumericDistribution`` is functionally equivalent to a Monte Carlo
simulation where you generate infinitely many samples and then group the
samples into finitely many bins, keeping track of the proportion of samples
in each bin (a.k.a. the probability mass) and the average value for each
bin.

Compared to a Monte Carlo simulation, ``NumericDistribution`` can represent
information much more densely by grouping together nearby values (although
some information is lost in the grouping). The benefit of this is most
obvious in fat-tailed distributions. In a Monte Carlo simulation, perhaps 1
in 1000 samples account for 10% of the expected value, but a
``NumericDistribution`` (with the right bin sizing method, see
:any:`BinSizing`) can easily track the probability mass of large values.

Accuracy
--------

The construction of ``NumericDistribution`` ensures that its expected value
is always close to 100% accurate. The higher moments (standard deviation,
skewness, etc.) and percentiles are less accurate, but still almost always
more accurate than Monte Carlo in practice.

We are probably most interested in the accuracy of percentiles. Consider a
simulation that applies binary operations to combine ``m`` different
``NumericDistribution`` s, each with ``n`` bins. The relative error of
estimated percentiles grows with :math:`O(m / n^2)`. That is, the error is
proportional to the number of operations and inversely proportional to the
square of the number of bins.

Compare this to the relative error of percentiles for a Monte Carlo (MC)
simulation over a log-normal distribution. MC relative error grows with
:math:`O(\sqrt{m} / n)` [1], given the assumption that if our
``NumericDistribution`` has ``n`` bins, then our MC simulation runs ``n^2``
samples (because both have a runtime of approximately :math:`O(n^2)`). So
MC scales worse with ``n``, but better with ``m``.

I tested accuracy across a range of percentiles for a variety of values of
``m`` and ``n``. Although MC scales better with ``m`` than
``NumericDistribution``, MC does not achieve lower error rates until ``m =
500`` or so (using ``n = 200``). Few simulations will involve combining 500
separate variables, so ``NumericDistribution`` should nearly always perform
better in practice.

Similarly, the error on ``NumericDistribution``'s estimated standard
deviation scales with :math:`O(m / n^2)`. I don't know the formula for the relative error of MC standard deviation, but empirically, it appears to scale with :math:`O(\sqrt{m} / n)`.

[1] Goodman (1983). Accuracy and Efficiency of Monte Carlo Method.
https://inis.iaea.org/collection/NCLCollectionStore/_Public/19/047/19047359.pdf

Runtime performance
-------------------

Bottom line: On the example models that I tested, simulating the model
using ``NumericDistribution``s with ``n`` bins ran about 3x faster than
using Monte Carlo with ``n^2`` bins, and the ``NumericDistribution``
results were more accurate.

Where ``n`` is the number of bins, constructing a ``NumericDistribution``
or performing a unary operation has runtime :math:`O(n)`. A binary
operation (such as addition or multiplication) has a runtime close to
:math:`O(n^2)`. To be precise, the runtime is :math:`O(n^2 \log(n))`
because the :math:`n^2` results of a binary operation must be partitioned
into :math:`n` ordered bins. In practice, this partitioning operation takes
up a fairly small portion of the runtime for ``n = 200`` (the default bin
count), and only takes up ~half the runtime for ``n > 1000``.

For ``n = 200``, a binary operation takes about twice as long as
constructing a ``NumericDistribution``.

Accuracy is linear in the number of bins but runtime is quadratic, so you
typically don't want to use bin counts larger than the default unless
you're particularly concerned about accuracy.

On setting values within bins
-----------------------------
Whenever possible, NumericDistribution assigns the value of each bin as the
average value between the two edges (weighted by mass). You can think of
this as the result you'd get if you generated infinitely many Monte Carlo
samples and grouped them into bins, setting the value of each bin as the
average of the samples. You might call this the expected value (EV) method,
in contrast to two methods described below.

The EV method guarantees that, whenever the histogram width covers the full
support of the distribution, the histogram's expected value exactly equals
the expected value of the true distribution (modulo floating point rounding
errors).

There are some other methods we could use, which are generally worse:

1. Set the value of each bin to the average of the two edges (the
"trapezoid rule"). The purpose of using the trapezoid rule is that we don't
know the probability mass within a bin (perhaps the CDF is too hard to
evaluate) so we have to estimate it. But whenever we *do* know the CDF, we
can calculate the probability mass exactly, so we don't need to use the
trapezoid rule.

2. Set the value of each bin to the center of the probability mass (the
"mass method"). This is equivalent to generating infinitely many Monte
Carlo samples and grouping them into bins, setting the value of each bin as
the **median** of the samples. This approach does not particularly help us
because we don't care about the median of every bin. We might care about
the median of the distribution, but we can calculate that near-exactly
regardless of what value-setting method we use by looking at the value in
the bin where the probability mass crosses 0.5. And the mass method will
systematically underestimate (the absolute value of) EV because the
definition of expected value places larger weight on larger (absolute)
values, and the mass method does not.

Although the EV method perfectly measures the expected value of a distribution,
it systematically underestimates the variance. To see this, consider that it is
possible to define the variance of a random variable X as :math:`E[X^2] -
E[X]^2`. The EV method correctly estimates :math:`E[X]`, so it also correctly
estimates :math:`E[X]^2`. However, it systematically underestimates
:math:`E[X^2]` because :math:`E[X^2]` places more weight on larger values. But
an alternative method that accurately estimated variance would necessarily
overestimate :math:`E[X]`.

On bin sizing for two-sided distributions
-----------------------------------------
The interpretation of the EV bin-sizing method is slightly non-obvious
for two-sided distributions because we must decide how to interpret bins
with negative expected value.

The EV method arranges values into bins such that:
    * The negative side has the correct negative contribution to EV and the
      positive side has the correct positive contribution to EV.
    * Every negative bin has equal contribution to EV and every positive bin
      has equal contribution to EV.
    * If a side has nonzero probability mass, then it has at least one bin,
      regardless of how small its probability mass.
    * The number of negative and positive bins are chosen such that the
      absolute contribution to EV for negative bins is as close as possible
      to the absolute contribution to EV for positive bins given the above
      constraints.

This binning method means that the distribution EV is exactly preserved
and there is no bin that contains the value zero. However, the positive
and negative bins do not necessarily have equal contribution to EV, and
the magnitude of the error is at most ``1 / num_bins / 2``.
