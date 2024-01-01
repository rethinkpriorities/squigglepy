from functools import reduce
from hypothesis import assume, example, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, optimize, stats
import sys
from unittest.mock import patch, Mock
import warnings

from ..squigglepy.distributions import *
from ..squigglepy.numeric_distribution import numeric, NumericDistribution
from ..squigglepy import samplers, utils


RUN_PRINT_ONLY_TESTS = False
"""Some tests print information but don't assert anything. This flag determines
whether to run those tests."""


def relative_error(x, y):
    if x == 0 and y == 0:
        return 0
    if x == 0:
        return abs(y)
    if y == 0:
        return abs(x)
    return max(x / y, y / x) - 1


def print_accuracy_ratio(x, y, extra_message=None):
    ratio = relative_error(x, y)
    if extra_message is not None:
        extra_message += " "
    else:
        extra_message = ""
    direction_off = "small" if x < y else "large"
    if ratio > 1:
        print(f"{extra_message}Ratio: {direction_off} by a factor of {ratio + 1:.1f}")
    else:
        print(f"{extra_message}Ratio: {direction_off} by {100 * ratio:.4f}%")


def fmt(x):
    return f"{(100*x):.4f}%"


def get_mc_accuracy(exact_sd, num_samples, dists, operation):
    # Run multiple trials because NumericDistribution should usually beat MC,
    # but sometimes MC wins by luck. Even though NumericDistribution wins a
    # large percentage of the time, this test suite does a lot of runs, so the
    # chance of MC winning at least once is fairly high.
    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(operation, mcs)
        mc_abs_error.append(abs(np.std(mc) - exact_sd))

    mc_abs_error.sort()

    # Small numbers are good. A smaller index in mc_abs_error has a better
    # accuracy
    return mc_abs_error[-5]


def test_norm_sd_bin_sizing_accuracy():
    # Accuracy order is ev > uniform > mass
    dist = NormalDistribution(mean=0, sd=1)
    ev_hist = numeric(dist, bin_sizing="ev", warn=False)
    mass_hist = numeric(dist, bin_sizing="mass", warn=False)
    uniform_hist = numeric(dist, bin_sizing="uniform", warn=False)

    sd_errors = [
        relative_error(uniform_hist.est_sd(), dist.sd),
        relative_error(ev_hist.est_sd(), dist.sd),
        relative_error(mass_hist.est_sd(), dist.sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_norm_product_bin_sizing_accuracy():
    dist = NormalDistribution(mean=2, sd=1)
    uniform_hist = numeric(dist, bin_sizing="uniform", warn=False)
    uniform_hist = uniform_hist * uniform_hist
    ev_hist = numeric(dist, bin_sizing="ev", warn=False)
    ev_hist = ev_hist * ev_hist
    mass_hist = numeric(dist, bin_sizing="mass", warn=False)
    mass_hist = mass_hist * mass_hist

    # uniform and log-uniform should have small errors and the others should be
    # pretty much perfect
    mean_errors = np.array([
        relative_error(mass_hist.est_mean(), ev_hist.exact_mean),
        relative_error(ev_hist.est_mean(), ev_hist.exact_mean),
        relative_error(uniform_hist.est_mean(), ev_hist.exact_mean),
    ])
    assert all(mean_errors <= 1e-6)

    sd_errors = [
        relative_error(ev_hist.est_sd(), ev_hist.exact_sd),
        relative_error(mass_hist.est_sd(), ev_hist.exact_sd),
        relative_error(uniform_hist.est_sd(), ev_hist.exact_sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_lognorm_product_bin_sizing_accuracy():
    dist = LognormalDistribution(norm_mean=np.log(1e6), norm_sd=1)
    uniform_hist = numeric(dist, bin_sizing="uniform", warn=False)
    uniform_hist = uniform_hist * uniform_hist
    log_uniform_hist = numeric(dist, bin_sizing="log-uniform", warn=False)
    log_uniform_hist = log_uniform_hist * log_uniform_hist
    ev_hist = numeric(dist, bin_sizing="ev", warn=False)
    ev_hist = ev_hist * ev_hist
    mass_hist = numeric(dist, bin_sizing="mass", warn=False)
    mass_hist = mass_hist * mass_hist
    fat_hybrid_hist = numeric(dist, bin_sizing="fat-hybrid", warn=False)
    fat_hybrid_hist = fat_hybrid_hist * fat_hybrid_hist
    dist_prod = LognormalDistribution(
        norm_mean=2 * dist.norm_mean, norm_sd=np.sqrt(2) * dist.norm_sd
    )

    mean_errors = np.array([
        relative_error(mass_hist.est_mean(), dist_prod.lognorm_mean),
        relative_error(ev_hist.est_mean(), dist_prod.lognorm_mean),
        relative_error(fat_hybrid_hist.est_mean(), dist_prod.lognorm_mean),
        relative_error(uniform_hist.est_mean(), dist_prod.lognorm_mean),
        relative_error(log_uniform_hist.est_mean(), dist_prod.lognorm_mean),
    ])
    assert all(mean_errors <= 1e-6)

    sd_errors = [
        relative_error(fat_hybrid_hist.est_sd(), dist_prod.lognorm_sd),
        relative_error(log_uniform_hist.est_sd(), dist_prod.lognorm_sd),
        relative_error(ev_hist.est_sd(), dist_prod.lognorm_sd),
        relative_error(mass_hist.est_sd(), dist_prod.lognorm_sd),
        relative_error(uniform_hist.est_sd(), dist_prod.lognorm_sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_lognorm_clip_center_bin_sizing_accuracy():
    dist1 = LognormalDistribution(norm_mean=-1, norm_sd=0.5, lclip=0, rclip=1)
    dist2 = LognormalDistribution(norm_mean=0, norm_sd=1, lclip=0, rclip=2 * np.e)
    true_mean1 = stats.lognorm.expect(
        lambda x: x,
        args=(dist1.norm_sd,),
        scale=np.exp(dist1.norm_mean),
        lb=dist1.lclip,
        ub=dist1.rclip,
        conditional=True,
    )
    true_sd1 = np.sqrt(
        stats.lognorm.expect(
            lambda x: (x - true_mean1) ** 2,
            args=(dist1.norm_sd,),
            scale=np.exp(dist1.norm_mean),
            lb=dist1.lclip,
            ub=dist1.rclip,
            conditional=True,
        )
    )
    true_mean2 = stats.lognorm.expect(
        lambda x: x,
        args=(dist2.norm_sd,),
        scale=np.exp(dist2.norm_mean),
        lb=dist2.lclip,
        ub=dist2.rclip,
        conditional=True,
    )
    true_sd2 = np.sqrt(
        stats.lognorm.expect(
            lambda x: (x - true_mean2) ** 2,
            args=(dist2.norm_sd,),
            scale=np.exp(dist2.norm_mean),
            lb=dist2.lclip,
            ub=dist2.rclip,
            conditional=True,
        )
    )
    true_mean = true_mean1 * true_mean2
    true_sd = np.sqrt(
        true_sd1**2 * true_mean2**2
        + true_mean1**2 * true_sd2**2
        + true_sd1**2 * true_sd2**2
    )

    uniform_hist = numeric(dist1, bin_sizing="uniform", warn=False) * numeric(
        dist2, bin_sizing="uniform", warn=False
    )
    log_uniform_hist = numeric(dist1, bin_sizing="log-uniform", warn=False) * numeric(
        dist2, bin_sizing="log-uniform", warn=False
    )
    ev_hist = numeric(dist1, bin_sizing="ev", warn=False) * numeric(
        dist2, bin_sizing="ev", warn=False
    )
    mass_hist = numeric(dist1, bin_sizing="mass", warn=False) * numeric(
        dist2, bin_sizing="mass", warn=False
    )
    fat_hybrid_hist = numeric(dist1, bin_sizing="fat-hybrid", warn=False) * numeric(
        dist2, bin_sizing="fat-hybrid", warn=False
    )

    mean_errors = np.array([
        relative_error(ev_hist.est_mean(), true_mean),
        relative_error(mass_hist.est_mean(), true_mean),
        relative_error(uniform_hist.est_mean(), true_mean),
        relative_error(fat_hybrid_hist.est_mean(), true_mean),
        relative_error(log_uniform_hist.est_mean(), true_mean),
    ])
    assert all(mean_errors <= 1e-6)

    # Uniform does poorly in general with fat-tailed dists, but it does well
    # with a center clip because most of the mass is in the center
    sd_errors = [
        relative_error(mass_hist.est_mean(), true_mean),
        relative_error(uniform_hist.est_sd(), true_sd),
        relative_error(ev_hist.est_sd(), true_sd),
        relative_error(fat_hybrid_hist.est_sd(), true_sd),
        relative_error(log_uniform_hist.est_sd(), true_sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_lognorm_clip_tail_bin_sizing_accuracy():
    # cut off 99% of mass and 95% of mass, respectively
    dist1 = LognormalDistribution(norm_mean=0, norm_sd=1, lclip=10)
    dist2 = LognormalDistribution(norm_mean=0, norm_sd=2, rclip=27)
    true_mean1 = stats.lognorm.expect(
        lambda x: x,
        args=(dist1.norm_sd,),
        scale=np.exp(dist1.norm_mean),
        lb=dist1.lclip,
        ub=dist1.rclip,
        conditional=True,
    )
    true_sd1 = np.sqrt(
        stats.lognorm.expect(
            lambda x: (x - true_mean1) ** 2,
            args=(dist1.norm_sd,),
            scale=np.exp(dist1.norm_mean),
            lb=dist1.lclip,
            ub=dist1.rclip,
            conditional=True,
        )
    )
    true_mean2 = stats.lognorm.expect(
        lambda x: x,
        args=(dist2.norm_sd,),
        scale=np.exp(dist2.norm_mean),
        lb=dist2.lclip,
        ub=dist2.rclip,
        conditional=True,
    )
    true_sd2 = np.sqrt(
        stats.lognorm.expect(
            lambda x: (x - true_mean2) ** 2,
            args=(dist2.norm_sd,),
            scale=np.exp(dist2.norm_mean),
            lb=dist2.lclip,
            ub=dist2.rclip,
            conditional=True,
        )
    )
    true_mean = true_mean1 * true_mean2
    true_sd = np.sqrt(
        true_sd1**2 * true_mean2**2
        + true_mean1**2 * true_sd2**2
        + true_sd1**2 * true_sd2**2
    )

    uniform_hist = numeric(dist1, bin_sizing="uniform", warn=False) * numeric(
        dist2, bin_sizing="uniform", warn=False
    )
    log_uniform_hist = numeric(dist1, bin_sizing="log-uniform", warn=False) * numeric(
        dist2, bin_sizing="log-uniform", warn=False
    )
    ev_hist = numeric(dist1, bin_sizing="ev", warn=False) * numeric(
        dist2, bin_sizing="ev", warn=False
    )
    mass_hist = numeric(dist1, bin_sizing="mass", warn=False) * numeric(
        dist2, bin_sizing="mass", warn=False
    )
    fat_hybrid_hist = numeric(dist1, bin_sizing="fat-hybrid", warn=False) * numeric(
        dist2, bin_sizing="fat-hybrid", warn=False
    )

    mean_errors = np.array([
        relative_error(mass_hist.est_mean(), true_mean),
        relative_error(uniform_hist.est_mean(), true_mean),
        relative_error(ev_hist.est_mean(), true_mean),
        relative_error(fat_hybrid_hist.est_mean(), true_mean),
        relative_error(log_uniform_hist.est_mean(), true_mean),
    ])
    assert all(mean_errors <= 1e-5)

    sd_errors = [
        relative_error(fat_hybrid_hist.est_sd(), true_sd),
        relative_error(log_uniform_hist.est_sd(), true_sd),
        relative_error(ev_hist.est_sd(), true_sd),
        relative_error(uniform_hist.est_sd(), true_sd),
        relative_error(mass_hist.est_sd(), true_sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_gamma_bin_sizing_accuracy():
    dist1 = GammaDistribution(shape=1, scale=5)
    dist2 = GammaDistribution(shape=10, scale=1)

    uniform_hist = numeric(dist1, bin_sizing="uniform") * numeric(dist2, bin_sizing="uniform")
    log_uniform_hist = numeric(dist1, bin_sizing="log-uniform") * numeric(
        dist2, bin_sizing="log-uniform"
    )
    ev_hist = numeric(dist1, bin_sizing="ev") * numeric(dist2, bin_sizing="ev")
    mass_hist = numeric(dist1, bin_sizing="mass") * numeric(dist2, bin_sizing="mass")
    fat_hybrid_hist = numeric(dist1, bin_sizing="fat-hybrid") * numeric(
        dist2, bin_sizing="fat-hybrid"
    )

    true_mean = uniform_hist.exact_mean
    true_sd = uniform_hist.exact_sd

    mean_errors = np.array([
        relative_error(mass_hist.est_mean(), true_mean),
        relative_error(uniform_hist.est_mean(), true_mean),
        relative_error(ev_hist.est_mean(), true_mean),
        relative_error(log_uniform_hist.est_mean(), true_mean),
        relative_error(fat_hybrid_hist.est_mean(), true_mean),
    ])
    assert all(mean_errors <= 1e-6)

    sd_errors = [
        relative_error(ev_hist.est_sd(), true_sd),
        relative_error(uniform_hist.est_sd(), true_sd),
        relative_error(fat_hybrid_hist.est_sd(), true_sd),
        relative_error(log_uniform_hist.est_sd(), true_sd),
        relative_error(mass_hist.est_sd(), true_sd),
    ]
    assert all(np.diff(sd_errors) >= 0)


def test_norm_product_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
    # Time complexity for binary operations is roughly O(n^2) for PMH and O(n)
    # for MC, so let MC have num_bins^2 samples.
    num_bins = 100
    num_samples = 100**2
    dists = [NormalDistribution(mean=i, sd=0.5 + i / 4) for i in range(9)]
    hists = [numeric(dist, num_bins=num_bins, warn=False) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.est_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc * mc)
    assert dist_abs_error < mc_abs_error


def test_lognorm_product_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(9)]
    hists = [numeric(dist, num_bins=num_bins, warn=False) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.est_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc * mc)
    assert dist_abs_error < mc_abs_error


def test_norm_sum_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
    num_bins = 1000
    num_samples = num_bins**2
    dists = [NormalDistribution(mean=i, sd=0.5 + i / 4) for i in range(9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hists = [numeric(dist, num_bins=num_bins, bin_sizing="uniform") for dist in dists]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.est_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc + mc)
    assert dist_abs_error < mc_abs_error


def test_lognorm_sum_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(17)]
    hists = [numeric(dist, num_bins=num_bins, warn=False) for dist in dists]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.est_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc + mc)
    assert dist_abs_error < mc_abs_error


def test_quantile_accuracy():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    props = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    # props = np.array([0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999])
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    true_quantiles = stats.lognorm.ppf(props, dist.norm_sd, scale=np.exp(dist.norm_mean))
    # dist = NormalDistribution(mean=0, sd=1)
    # true_quantiles = stats.norm.ppf(props, dist.mean, dist.sd)
    num_bins = 100
    num_mc_samples = num_bins**2

    # Formula from Goodman, "Accuracy and Efficiency of Monte Carlo Method."
    # https://inis.iaea.org/collection/NCLCollectionStore/_Public/19/047/19047359.pdf
    # Figure 20 on page 434.
    mc_error = np.sqrt(props * (1 - props)) * np.sqrt(2 * np.pi) * dist.norm_sd * np.exp(0.5 * (np.log(true_quantiles) - dist.norm_mean)**2 / dist.norm_sd**2) / np.sqrt(num_mc_samples)
    # mc_error = np.sqrt(props * (1 - props)) * np.sqrt(2 * np.pi) * np.exp(0.5 * (true_quantiles - dist.mean)**2) / abs(true_quantiles) / np.sqrt(num_mc_samples)

    print("\n")
    print(f"MC error: average {fmt(np.mean(mc_error))}, median {fmt(np.median(mc_error))}, max {fmt(np.max(mc_error))}")

    for bin_sizing in ["log-uniform", "mass", "ev", "fat-hybrid"]:
    # for bin_sizing in ["uniform", "mass", "ev"]:
        hist = numeric(dist, bin_sizing=bin_sizing, warn=False, num_bins=num_bins)
        linear_quantiles = np.interp(props, np.cumsum(hist.masses) - 0.5 * hist.masses, hist.values)
        hist_quantiles = hist.quantile(props)
        linear_error = abs(true_quantiles - linear_quantiles) / abs(true_quantiles)
        hist_error = abs(true_quantiles - hist_quantiles) / abs(true_quantiles)
        print(f"\n{bin_sizing}")
        print(f"\tLinear error: average {fmt(np.mean(linear_error))}, median {fmt(np.median(linear_error))}, max {fmt(np.max(linear_error))}")
        print(f"\tHist error  : average {fmt(np.mean(hist_error))}, median {fmt(np.median(hist_error))}, max {fmt(np.max(hist_error))}")
        print(f"\tHist / MC   : average {fmt(np.mean(hist_error / mc_error))}, median {fmt(np.median(hist_error / mc_error))}, max {fmt(np.max(hist_error / mc_error))}")


def test_quantile_product_accuracy():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    props = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 0.999])  # EV
    # props = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]) # lognorm
    # props = np.array([0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999])  # norm
    num_bins = 200
    print("\n")

    bin_sizing = "log-uniform"
    hists = []
    # for bin_sizing in ["log-uniform", "mass", "ev", "fat-hybrid"]:
    for num_products in [2, 8, 32, 128, 512]:
        dist1 = LognormalDistribution(norm_mean=0, norm_sd=1 / np.sqrt(num_products))
        dist = LognormalDistribution(norm_mean=dist1.norm_mean * num_products, norm_sd=dist1.norm_sd * np.sqrt(num_products))
        true_quantiles = stats.lognorm.ppf(props, dist.norm_sd, scale=np.exp(dist.norm_mean))
        num_mc_samples = num_bins**2

        # I'm not sure how to prove this, but empirically, it looks like the error
        # for MC(x) * MC(y) is the same as the error for MC(x * y).
        mc_error = np.sqrt(props * (1 - props)) * np.sqrt(2 * np.pi) * dist.norm_sd * np.exp(0.5 * (np.log(true_quantiles) - dist.norm_mean)**2 / dist.norm_sd**2) / np.sqrt(num_mc_samples)

        hist1 = numeric(dist1, bin_sizing=bin_sizing, warn=False, num_bins=num_bins)
        hist = reduce(lambda acc, x: acc * x, [hist1] * num_products)
        oneshot = numeric(dist, bin_sizing=bin_sizing, warn=False, num_bins=num_bins)
        linear_quantiles = np.interp(props, np.cumsum(hist.masses) - 0.5 * hist.masses, hist.values)
        hist_quantiles = hist.quantile(props)
        linear_error = abs(true_quantiles - linear_quantiles) / abs(true_quantiles)
        hist_error = abs(true_quantiles - hist_quantiles) / abs(true_quantiles)
        oneshot_error = abs(true_quantiles - oneshot.quantile(props)) / abs(true_quantiles)
        hists.append(hist)

        # print(f"\n{bin_sizing}")
        # print(f"\tLinear error: average {fmt(np.mean(linear_error))}, median {fmt(np.median(linear_error))}, max {fmt(np.max(linear_error))}")
        print(f"{num_products}")
        print(f"\tHist error  : average {fmt(np.mean(hist_error))}, median {fmt(np.median(hist_error))}, max {fmt(np.max(hist_error))}")
        print(f"\tHist / MC   : average {fmt(np.mean(hist_error / mc_error))}, median {fmt(np.median(hist_error / mc_error))}, max {fmt(np.max(hist_error / mc_error))}")
        print(f"\tHist / 1shot: average {fmt(np.mean(hist_error / oneshot_error))}, median {fmt(np.median(hist_error / oneshot_error))}, max {fmt(np.max(hist_error / oneshot_error))}")

    indexes = [10, 20, 50, 80, 90]
    selected = np.array([x.values[indexes] for x in hists])
    diffs = np.diff(selected, axis=0)


def test_individual_bin_accuracy():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    num_bins = 200
    bin_sizing = "ev"
    print("")
    bin_errs = []
    num_products = 16
    bin_sizes = 40 * np.arange(1, 11)
    for num_bins in bin_sizes:
        operation = "mul"
        if operation == "mul":
            true_dist_type = 'lognorm'
            true_dist = LognormalDistribution(norm_mean=0, norm_sd=1)
            dist1 = LognormalDistribution(norm_mean=0, norm_sd=1 / np.sqrt(num_products))
            true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            hist = reduce(lambda acc, x: acc * x, [hist1] * num_products)
        elif operation == "add":
            true_dist_type = 'norm'
            true_dist = NormalDistribution(mean=0, sd=1)
            dist1 = NormalDistribution(mean=0, sd=1 / np.sqrt(num_products))
            true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            hist = reduce(lambda acc, x: acc + x, [hist1] * num_products)
        elif operation == "exp":
            true_dist_type = 'lognorm'
            true_dist = LognormalDistribution(norm_mean=0, norm_sd=1)
            true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            dist1 = NormalDistribution(mean=0, sd=1)
            hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
            hist = hist1.exp()

        cum_mass = np.cumsum(hist.masses)
        cum_cev = np.cumsum(hist.masses * abs(hist.values))
        cum_cev_frac = cum_cev / cum_cev[-1]
        if true_dist_type == 'lognorm':
            expected_cum_mass = stats.lognorm.cdf(true_dist.inv_contribution_to_ev(cum_cev_frac), true_dist.norm_sd, scale=np.exp(true_dist.norm_mean))
        elif true_dist_type == 'norm':
            expected_cum_mass = stats.norm.cdf(true_dist.inv_contribution_to_ev(cum_cev_frac), true_dist.mean, true_dist.sd)

        # Take only every nth value where n = num_bins/40
        cum_mass = cum_mass[::num_bins // 40]
        expected_cum_mass = expected_cum_mass[::num_bins // 40]
        bin_errs.append(abs(cum_mass - expected_cum_mass) / expected_cum_mass)

    bin_errs = np.array(bin_errs)

    best_fits = []
    for i in range(40):
        try:
            best_fit = optimize.curve_fit(lambda x, a, r: a*x**r, bin_sizes, bin_errs[:, i], p0=[1, 2])[0]
            best_fits.append(best_fit)
            print(f"{i:2d} {best_fit[0]:9.3f} * x ^ {best_fit[1]:.3f}")
        except RuntimeError:
            # optimal parameters not found
            print(f"{i:2d} ? ?")

    print("")
    print(f"Average: {np.mean(best_fits, axis=0)}\nMedian: {np.median(best_fits, axis=0)}")

    meta_fit = np.polynomial.polynomial.Polynomial.fit(np.array(range(len(best_fits))) / len(best_fits), np.array(best_fits)[:, 1], 2)
    print(f"\nMeta fit: {meta_fit}")


def test_richardson_product():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    print("")
    num_bins = 200
    num_products = 2
    bin_sizing = "ev"
    # mixture_ratio = [0.035, 0.965]
    mixture_ratio = [0, 1]
    # mixture_ratio = [0.3, 0.7]
    bin_sizes = 40 * np.arange(1, 11)
    product_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    err_rates = []
    for num_products in product_nums:
    # for num_bins in bin_sizes:
        true_mixture_ratio = reduce(lambda acc, x: (acc[0] * x[1] + acc[1] * x[0], acc[0] * x[0] + acc[1] * x[1]), [(mixture_ratio) for _ in range(num_products)])
        one_sided_dist = LognormalDistribution(norm_mean=0, norm_sd=1)
        true_dist = mixture([-one_sided_dist, one_sided_dist], true_mixture_ratio)
        true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        one_sided_dist1 = LognormalDistribution(norm_mean=0, norm_sd=1 / np.sqrt(num_products))
        dist1 = mixture([-one_sided_dist1, one_sided_dist1], mixture_ratio)
        hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        hist = reduce(lambda acc, x: acc * x, [hist1] * num_products)

        test_mode = 'ppf'
        if test_mode == 'cev':
            true_answer = one_sided_dist.contribution_to_ev(stats.lognorm.ppf(2 * hist.masses[50:100].sum(), one_sided_dist.norm_sd, scale=np.exp(one_sided_dist.norm_mean)), False) / 2
            est_answer = (hist.masses * abs(hist.values))[50:100].sum()
            print_accuracy_ratio(est_answer, true_answer, f"CEV({num_products:3d})")
        elif test_mode == 'sd':
            mcs = [samplers.sample(dist, num_bins**2) for dist in [dist1] * num_products]
            mc = reduce(lambda acc, x: acc * x, mcs)
            true_answer = true_hist.exact_sd
            est_answer = hist.est_sd()
            mc_answer = np.std(mc)
            print_accuracy_ratio(est_answer, true_answer, f"SD({num_products:3d}, {num_bins:3d})")
            # print_accuracy_ratio(mc_answer, true_answer, f"MC({num_products:3d}, {num_bins:3d})")
            err_rates.append(abs(est_answer - true_answer))
        elif test_mode == 'ppf':
            fracs = [0.5, 0.75, 0.9, 0.97, 0.99]
            frac_errs = []
            # mc_errs = []
            for frac in fracs:
                true_answer = stats.lognorm.ppf((frac - true_mixture_ratio[0]) / true_mixture_ratio[1], one_sided_dist.norm_sd, scale=np.exp(one_sided_dist.norm_mean))
                oneshot_answer = true_hist.ppf(frac)
                est_answer = hist.ppf(frac)
                frac_errs.append(abs(est_answer - true_answer) / true_answer)
            median_err = np.median(frac_errs)
            print(f"ppf ({num_products:3d}, {num_bins:3d}): {median_err * 100:.3f}%")
            err_rates.append(median_err)

    if num_bins == bin_sizes[-1]:
        best_fit = optimize.curve_fit(lambda x, a, r: a*x**r, bin_sizes, err_rates, p0=[1, 2])[0]
        print(f"\nBest fit: {best_fit}")
    else:
        best_fit = optimize.curve_fit(lambda x, a, r: a*x**r, product_nums, err_rates, p0=[1, 2])[0]
        print(f"\nBest fit: {best_fit}")


def test_richardson_sum():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    print("")
    num_bins = 200
    num_sums = 2
    bin_sizing = "ev"
    bin_sizes = 40 * np.arange(1, 11)
    err_rates = []
    # for num_sums in [2, 4, 8, 16, 32, 64]:
    for num_bins in bin_sizes:
        true_dist = NormalDistribution(mean=0, sd=1)
        true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        dist1 = NormalDistribution(mean=0, sd=1 / np.sqrt(num_sums))
        hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        hist = reduce(lambda acc, x: acc + x, [hist1] * num_sums)

        test_mode = 'ppf'
        if test_mode == 'cev':
            true_answer = one_sided_dist.contribution_to_ev(stats.lognorm.ppf(2 * hist.masses[50:100].sum(), one_sided_dist.norm_sd, scale=np.exp(one_sided_dist.norm_mean)), False) / 2
            est_answer = (hist.masses * abs(hist.values))[50:100].sum()
            print_accuracy_ratio(est_answer, true_answer, f"CEV({num_sums:3d})")
        elif test_mode == 'sd':
            true_answer = true_hist.exact_sd
            est_answer = hist.est_sd()
            print_accuracy_ratio(est_answer, true_answer, f"SD({num_sums}, {num_bins:3d})")
            err_rates.append(abs(est_answer - true_answer))
        elif test_mode == 'ppf':
            fracs = [0.75, 0.9, 0.95, 0.98, 0.99]
            frac_errs = []
            for frac in fracs:
                true_answer = stats.norm.ppf(frac, true_dist.mean, true_dist.sd)
                est_answer = hist.ppf(frac)
                frac_errs.append(abs(est_answer - true_answer) / true_answer)
            median_err = np.median(frac_errs)
            print(f"ppf ({num_sums:3d}, {num_bins:3d}): {median_err * 100:.3f}%")
            err_rates.append(median_err)

    if len(err_rates) == len(bin_sizes):
        best_fit = optimize.curve_fit(lambda x, a, r: a*x**r, bin_sizes, err_rates, p0=[1, 2])[0]
        print(f"\nBest fit: {best_fit}")


def test_richardson_exp():
    if not RUN_PRINT_ONLY_TESTS:
        return None
    print("")
    bin_sizing = "ev"
    bin_sizes = 200 * np.arange(1, 11)
    err_rates = []
    for num_bins in bin_sizes:
        true_dist = LognormalDistribution(norm_mean=0, norm_sd=1)
        true_hist = numeric(true_dist, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        dist1 = NormalDistribution(mean=0, sd=1)
        hist1 = numeric(dist1, bin_sizing=bin_sizing, num_bins=num_bins, warn=False)
        hist = hist1.exp()

        test_mode = 'sd'
        if test_mode == 'sd':
            true_answer = true_hist.exact_sd
            est_answer = hist.est_sd()
            print_accuracy_ratio(est_answer, true_answer, f"SD({num_bins:3d})")
            err_rates.append(abs(est_answer - true_answer))
        elif test_mode == 'ppf':
            fracs = [0.5, 0.75, 0.9, 0.97, 0.99]
            frac_errs = []
            for frac in fracs:
                true_answer = stats.lognorm.ppf(frac, true_dist.norm_sd, scale=np.exp(true_dist.norm_mean))
                est_answer = hist.ppf(frac)
                frac_errs.append(abs(est_answer - true_answer) / true_answer)
            median_err = np.median(frac_errs)
            print(f"ppf ({num_bins:4d}): {median_err * 100:.5f}%")
            err_rates.append(median_err)

    if len(err_rates) == len(bin_sizes):
        best_fit = optimize.curve_fit(lambda x, a, r: a*x**r, bin_sizes, err_rates, p0=[1, 2])[0]
        print(f"\nBest fit: {best_fit}")
