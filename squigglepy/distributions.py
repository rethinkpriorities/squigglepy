from __future__ import annotations
from functools import partial
import functools

import math
import operator
from typing import Any, Callable, Optional, Self, TypeVar, Union

import numpy as np
from scipy import stats

from .utils import (
    Integer,
    Weights,
    _is_numpy,
    _process_weights_values,
    _round,
    is_dist,
    Number,
    Float,
)
from .version import __version__

from typing import overload

from numpy import ufunc, vectorize


class BaseDistribution:
    def __init__(self) -> None:
        self.lclip: Optional[Number] = None
        self.rclip: Optional[Number] = None
        self._version: str = __version__

    def __str__(self):
        return "<Distribution> base"

    def __repr__(self) -> str:
        return str(self)


class OperableDistribution(BaseDistribution):
    def __init__(self) -> None:
        super().__init__()

    def __invert__(self):
        from .samplers import sample

        return sample(self)

    def __matmul__(self, n: int):
        try:
            n = int(n)
        except ValueError:
            raise ValueError("number of samples must be an integer")
        from .samplers import sample

        return sample(self, n=n)

    def __rshift__(
        self, fn: Union[Callable[[Self], ComplexDistribution], ComplexDistribution]
    ) -> Union[ComplexDistribution, int]:
        if callable(fn):
            return fn(self)
        elif isinstance(fn, ComplexDistribution):
            return ComplexDistribution(self, fn.left, fn.fn, fn.fn_str, infix=False)
        else:
            raise ValueError

    def __rmatmul__(self, n: int):
        return self.__matmul__(n)

    def __gt__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.gt, ">")

    def __ge__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.ge, ">=")

    def __lt__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.lt, "<")

    def __le__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.le, "<=")

    def __eq__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.le, "==")

    def __ne__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.le, "!=")

    def __neg__(self) -> "ComplexDistribution":
        return ComplexDistribution(self, None, operator.neg, "-")

    def __add__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.add, "+")

    def __radd__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.add, "+")

    def __sub__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.sub, "-")

    def __rsub__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.sub, "-")

    def __mul__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.mul, "*")

    def __rmul__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.mul, "*")

    def __truediv__(self, dist: BaseDistribution) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.truediv, "/")

    def __rtruediv__(self, dist: Number) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.truediv, "/")

    def __floordiv__(self, dist: int) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.floordiv, "//")

    def __rfloordiv__(self, dist: int) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.floordiv, "//")

    def __pow__(self, dist: int) -> "ComplexDistribution":
        return ComplexDistribution(self, dist, operator.pow, "**")

    def __rpow__(self, dist: int) -> "ComplexDistribution":
        return ComplexDistribution(dist, self, operator.pow, "**")

    def plot(self, num_samples: int = 1000, bins: int = 200) -> None:
        """
        Plot a histogram of the samples.

        Parameters
        ----------
        num_samples : int
            The number of samples to draw for plotting. Defaults to 1000 if not set.
        bins : int
            The number of bins to plot. Defaults to 200 if not set.

        Examples
        --------
        >>> sq.norm(5, 10).plot()
        """
        from matplotlib import pyplot as plt

        num_samples = 1000 if num_samples is None else num_samples
        bins = 200 if bins is None else bins

        samples = self @ num_samples

        plt.hist(samples, bins=bins)
        plt.show()


class ComplexDistribution(OperableDistribution):
    def __init__(
        self,
        left: Any,
        right: Optional[Any] = None,
        fn: UFunction = operator.add,
        fn_str: str = "+",
        infix: bool = True,
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.fn = fn
        self.fn_str = fn_str
        self.infix = infix

    def __str__(self) -> str:
        left, right, infix = self.left, self.right, self.infix
        if isinstance(self.fn, functools.partial) and right is None and not infix:
            # This prints the arguments when a partial function is being used
            # this is useful for things like round, lclip, rclip, etc.
            right = "".join(str(arg) for arg in self.fn.args)
            for i, (k, v) in enumerate(self.fn.keywords.items()):
                if i == 0 and right:
                    right += ", "
                right += "{}={}".format(k, v)

        if right is None and infix:
            if self.fn_str == "-":
                out = "<Distribution> {}{}"
            else:
                out = "<Distribution> {} {}"
            out = out.format(self.fn_str, str(left).replace("<Distribution> ", ""))
        elif right is None and not infix:
            out = "<Distribution> {}({})".format(
                self.fn_str, str(left).replace("<Distribution> ", "")
            )
        elif right is not None and infix:
            out = "<Distribution> {} {} {}".format(
                str(left).replace("<Distribution> ", ""),
                self.fn_str,
                str(right).replace("<Distribution> ", ""),
            )
        elif right is not None and not infix:
            out = "<Distribution> {}({}, {})"
            out = out.format(
                self.fn_str,
                str(left).replace("<Distribution> ", ""),
                str(right).replace("<Distribution> ", ""),
            )
        else:
            raise ValueError("The complex distribution is not properly defined")
        return out


def _get_fname(f: UFunction, name: Optional[str]) -> str:
    if name is None:
        if isinstance(f, np.vectorize):
            name = f.pyfunc.__name__
        else:
            name = f.__name__
    return name


# These are the types of functions that can be used in the dist_fn function
# We allow booleans for the case of the comparison operators
UScalar = TypeVar("UScalar", bound=Union[Number, bool])
UFunction = Union[
    Callable[[UScalar], UScalar],
    Callable[[UScalar, UScalar], UScalar],
    ufunc,
    vectorize,
]


@overload
def dist_fn(
    dist: OperableDistribution, fn: Union[UFunction, list[UFunction]], name: Optional[str] = None
) -> ComplexDistribution:
    """
    Initialize a distribution that has a custom function applied to the result.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist : Distribution or function or list
        The distribution to apply the function to.
    fn : function or list of functions
        The function(s) to apply to the distribution.
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    ComplexDistribution
        This distribution performs a lazy evaluation of the desired function that
        will be calculated when it is sampled.

    Examples
    --------
    >>> def double(x):
    >>>     return x * 2
    >>> dist_fn(norm(0, 1), double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    """
    # A unary function (or functions) was passed in
    # The function is applied to each sample
    ...


@overload
def dist_fn(
    dist1: OperableDistribution,
    dist2: OperableDistribution,
    fn: Union[UFunction, list[UFunction]],
    name: Optional[str] = None,
) -> ComplexDistribution:
    """
    Apply a binary function to a pair of distributions.

    The function won't be applied until the returned distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution to apply the function to.
    dist2 : Distribution
        The second distribution to apply the function to.
    fn : function or list of functions
        The function(s) to apply to the distributions. Must take two arguments and
        return a single value.
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    ComplexDistribution
        This distribution performs a lazy evaluation of the desired function that
        will be calculated when it is sampled.

    Examples
    --------
    >>> def add(x, y):
    >>>     return x + y
    >>> dist_fn(norm(0, 1), norm(1, 2), add)
    <Distribution> add(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    # A binary function (or function) passed was passed in
    # The function is applied to pairs of samples, and returns the combined sample
    ...


@overload
def dist_fn(
    fn: Union[UFunction, list[UFunction]],
    name: Optional[str] = None,
) -> Callable[[OperableDistribution], ComplexDistribution]:
    """
    Lazily applies the given function to the previous distribution in a pipe. (Or, equivalently,
    produces a partial of the function with the previous distribution as the first argument.)

    Parameters
    ----------
    fn : function or list of functions
        The function(s) to apply to the distributions. Each must take one argument and
        return a single value.
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    function
        A function that takes a distribution and returns a distribution. The returned distribution
        performs a lazy evaluation of the desired function whenever it is sampled.

    Examples
    --------
    >>> def double(x):
    >>>     return x * 2
    >>> norm(0, 1) >> dist_fn(double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    """
    # Unary case of the pipe version
    ...


@overload
def dist_fn(
    dist: OperableDistribution,
    fn: Union[UFunction, list[UFunction]],
    name: Optional[str] = None,
) -> Callable[[OperableDistribution], ComplexDistribution]:
    """
    Lazily applies the given binary function to the previous distribution in a pipe, with the given
    distribution as the second argument. (Or, equivalently, produces a partial of the function with
    the previous distribution passed as the second argument.)

    Parameters
    ----------
    dist : Distribution
        The distribution to apply the function to.
    fn : function or list of functions
        The function(s) to apply to the distributions. Each must take two arguments and
        return a single value.
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    function
        A function that takes a distribution and returns a distribution. The returned distribution
        performs a lazy evaluation of the desired function whenever it is sampled.

    Examples
    --------
    >>> def add(x, y):
    >>>     return x + y
    >>> norm(0, 1) >> dist_fn(norm(1, 2), add)
    <Distribution> add(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    # Binary case of the pipe version
    ...


def dist_fn(
    *args,
    **kwargs,
) -> Union[ComplexDistribution, Callable[[OperableDistribution], ComplexDistribution]]:
    """
    This is the dispatcher for the `dist_fn` function. It handles the different cases
    of the function, which are then passed to the `_dist_fn` function to actually
    create the respective distribution or partial function.

    This handles both unary and binary functions, as well as the pipe versions of each.
    The pipe version simply creates a partial of the inner `_dist_fn` function, and returns
    that partial.
    """
    # Get the arguments
    p1: Union[OperableDistribution, Union[UFunction, list[UFunction]]] = (
        args[0] if len(args) > 0 else kwargs.get("dist1", None)
    )
    p2: Optional[Union[OperableDistribution, Union[UFunction, list[UFunction]]]] = (
        args[1] if len(args) > 1 else kwargs.get("dist2", None)
    )
    p3: Union[UFunction, list[UFunction]] = args[2] if len(args) > 2 else kwargs.get("fn", None)
    name: str = kwargs.get("name", None)

    # Dispatch to the proper form of the function
    if (
        (isinstance(p1, OperableDistribution))
        and (not isinstance(p2, OperableDistribution) and p2 is not None)
        and (p3 is None)
    ):
        # Simple case, unary function
        p2 = p2 if isinstance(p2, list) else [p2]
        return _dist_fn(dist1=p1, fn_list=p2, name=name)
    elif (
        isinstance(p1, OperableDistribution)
        and isinstance(p2, OperableDistribution)
        and (p3 is not None)
    ):
        # Simple case, binary function
        p3 = p3 if isinstance(p3, list) else [p3]
        return _dist_fn(dist1=p1, dist2=p2, fn_list=p3, name=name)
    elif p1 is not None and not isinstance(p1, OperableDistribution) and p2 is None and p3 is None:
        # Pipe case, unary function
        p1 = p1 if isinstance(p1, list) else [p1]
        return lambda d: _dist_fn(dist1=d, fn_list=p1, name=name)
    elif (
        p1 is not None
        and not isinstance(p1, OperableDistribution)
        and isinstance(p2, OperableDistribution)
        and p3 is None
    ):
        p1 = p1 if isinstance(p1, list) else [p1]
        # Pipe case, binary function
        return lambda d: _dist_fn(dist1=d, dist2=p2, fn_list=p1, name=name)
    else:
        raise ValueError("Invalid arguments to dist_fn")


def _dist_fn(
    dist1: Optional[OperableDistribution] = None,
    dist2: Optional[OperableDistribution] = None,
    fn_list: Optional[list[UFunction]] = None,
    name: Optional[str] = None,
) -> ComplexDistribution:
    """
    This is the actual function that creates the complex distribution
    whenever `dist_fn` is used. It handles the simple one function case,
    as well as the case where multiple functions are being composed together.
    """
    assert dist1 is not None and fn_list is not None
    assert all(callable(f) for f in fn_list), "All functions provided must be callable"
    assert len(fn_list) > 0, "Must provide at least one function to compose"

    if len(fn_list) == 1:
        return ComplexDistribution(
            dist1, dist2, fn=fn_list[0], fn_str=_get_fname(fn_list[0], name), infix=False
        )
    else:
        assert (
            dist2 is None
        ), "Cannot provide a second distribution when composing multiple functions"

        out = ComplexDistribution(
            dist1, None, fn=fn_list[0], fn_str=_get_fname(fn_list[0], name), infix=False
        )
        for f in fn_list[1:]:
            out = ComplexDistribution(out, None, fn=f, fn_str=_get_fname(f, name), infix=False)

        return out


# def _dist_fn(
#     dist1: Union[OperableDistribution, Callable, list[Callable]],
#     dist2: Optional[Union[OperableDistribution, Callable, list[Callable]]] = None,
#     fn: Optional[UFunction] = None,
#     name: Optional[str] = None,
# ) -> Union[ComplexDistribution, Callable]:
#     """
#     Initialize a distribution that has a custom function applied to the result.

#     The function won't be applied until the distribution is sampled.

#     Parameters
#     ----------
#     dist1 : Distribution or function or list
#         Typically, the distribution to apply the function to. Could also be a function
#         or list of functions if ``dist_fn`` is being used in a pipe.
#     dist2 : Distribution or function or list or None
#         Typically, the second distribution to apply the function to if the function takes
#         two arguments. Could also be a function or list of functions if ``dist_fn`` is
#         being used in a pipe.
#     fn : function or None
#         The function to apply to the distribution(s).
#     name : str or None
#         By default, ``fn.__name__`` will be used to name the function. But you can pass
#         a custom name.

#     Returns
#     -------
#     ComplexDistribution or function
#         This will be a lazy evaluation of the desired function that will then be calculated
#         when it is sampled.

#     Examples
#     --------
#     >>> def double(x):
#     >>>     return x * 2
#     >>> dist_fn(norm(0, 1), double)
#     <Distribution> double(norm(mean=0.5, sd=0.3))
#     >>> norm(0, 1) >> dist_fn(double)
#     <Distribution> double(norm(mean=0.5, sd=0.3))
#     """

#     p1 = dist1 if isinstance(dist1, list) else [dist1]
#     assert (p1 is None) or all(callable(f) for f in p1)
#     p2 = dist2 if isinstance(dist2, list) else [dist2]
#     assert (p2 is None) or all(callable(f) for f in p2)
#     p3 = fn if isinstance(fn, list) else [fn]
#     assert (p3 is None) or all(callable(f) for f in p3)


#     if (p1 is not None and p2 is None and p3 is None) and (not ):
#         # This is a simple pipe,
#         # we compose the functions and return a partial of the composition

#         def out_fn(d):
#             out = d
#             for fn in p1:
#                 out = ComplexDistribution(
#                     out, None, fn=fn, fn_str=_get_fname(fn, name), infix=False
#                 )
#             return out

#         return out_fn


#     if callable(dist1) and dist2 is None and fn is None:
#         return lambda d: dist_fn(d, fn=dist1)

#     # These are cases where we're just applying function(s) to a distribution
#     if isinstance(dist2, list) and callable(dist2[0]) and fn is None:
#         # The user provided the functions through dist2
#         assert all(callable(f) for f in dist2)
#         fn_list = dist2

#     if callable(dist2) and fn is None:
#         fn_list = [dist2]

#     if fn is not None and callable(fn):
#         fn_list = [fn]

#     out = dist1
#     for f in fn_list:
#         out = ComplexDistribution(out, dist2, fn=f, fn_str=_get_fname(f, name), infix=False)

#     return out


def dist_max(
    dist1: OperableDistribution, dist2: Optional[OperableDistribution] = None
) -> Union[Callable, ComplexDistribution]:
    """
    Initialize the calculation of the maximum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the max of.
    dist2 : Distribution
        The second distribution to sample and determine the max of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated
        when it is sampled.

    Examples
    --------
    >>> dist_max(norm(0, 1), norm(1, 2))
    <Distribution> max(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    if is_dist(dist1) and dist2 is None:
        return lambda d: dist_fn(d, dist1, np.maximum, name="max")
    elif dist2 is not None:
        return dist_fn(dist1, dist2, np.maximum, name="max")
    else:
        raise ValueError("Invalid arguments to dist_max")


def dist_min(
    dist1: OperableDistribution,
    dist2: Optional[OperableDistribution] = None,
) -> Union[Callable, ComplexDistribution]:
    """
    Initialize the calculation of the minimum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the min of.
    dist2 : Distribution
        The second distribution to sample and determine the min of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_min(norm(0, 1), norm(1, 2))
    <Distribution> min(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    if is_dist(dist1) and dist2 is None:
        return lambda d: dist_fn(d, dist1, np.minimum, name="min")
    elif dist2 is not None:
        return dist_fn(dist1, dist2, np.minimum, name="min")
    else:
        raise ValueError("Invalid arguments to dist_min")


def dist_round(
    dist: OperableDistribution, digits: int = 0
) -> Union[Callable, ComplexDistribution]:
    """
    Initialize the rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then round.
    digits : int
        The number of digits to round to.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_round(norm(0, 1))
    <Distribution> round(norm(mean=0.5, sd=0.3), 0)
    """
    if isinstance(dist, int) and digits == 0:
        return lambda d: dist_round(d, digits=dist)
    else:
        return dist_fn(dist, partial(_round, digits=digits), name="round")


def dist_ceil(
    dist1: OperableDistribution,
) -> Union[ComplexDistribution, Callable]:
    """
    Initialize the ceiling rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then ceiling round.

    Returns
    -------
    ComplexDistribution
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_ceil(norm(0, 1))
    <Distribution> ceil(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, np.ceil)


def dist_floor(dist1: OperableDistribution):
    """
    Initialize the floor rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then floor round.

    Returns
    -------
    ComplexDistribution
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_floor(norm(0, 1))
    <Distribution> floor(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, np.floor)


def dist_log(
    dist1: OperableDistribution, base: Number = math.e
) -> Union[ComplexDistribution, Callable]:
    """
    Initialize the log of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then take the log of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_log(norm(0, 1), 10)
    <Distribution> log(norm(mean=0.5, sd=0.3), const(10))
    """
    return dist_fn(dist1, const(base), math.log)


def dist_exp(dist1: OperableDistribution):
    """
    Initialize the exp of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then take the exp of.

    Returns
    -------
    ComplexDistribution
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_exp(norm(0, 1))
    <Distribution> exp(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, math.exp)


@np.vectorize
def _lclip(n: Number, val: Number) -> Number:
    if val is None:
        return n
    else:
        return val if n < val else n


def lclip(
    dist1: Union[OperableDistribution, Callable],
    val: Optional[Number] = None,
) -> Union[ComplexDistribution, Callable]:
    """
    Initialize the clipping/bounding of the output of the distribution by the lower value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the lower bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> lclip(norm(0, 1), 0.5)
    <Distribution> lclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: lclip(d, dist1)
    elif isinstance(dist1, OperableDistribution):
        return dist_fn(dist1, partial(_lclip, val), name="lclip")
    else:
        return _lclip(dist1, val)


@np.vectorize
def _rclip(n: Number, val: Number) -> Number:
    if val is None:
        return n
    else:
        return val if n > val else n


def rclip(
    dist1: Union[OperableDistribution, Callable],
    val: Optional[Number] = None,
) -> Union[ComplexDistribution, Callable]:
    """
    Initialize the clipping/bounding of the output of the distribution by the upper value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> rclip(norm(0, 1), 0.5)
    <Distribution> rclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: rclip(d, dist1)
    elif isinstance(dist1, OperableDistribution):
        return dist_fn(dist1, partial(_rclip, val), name="rclip")
    else:
        return _rclip(dist1, val)


def clip(
    dist1: Union[OperableDistribution, Callable],
    left: Optional[Number],
    right: Optional[Number] = None,
) -> Union[ComplexDistribution, Callable]:
    """
    Initialize the clipping/bounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    left : int or float or None
        The value to use as the lower bound for clipping.
    right : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> clip(norm(0, 1), 0.5, 0.9)
    <Distribution> rclip(lclip(norm(mean=0.5, sd=0.3), 0.5), 0.9)
    """
    if (
        (isinstance(dist1, int) or isinstance(dist1, float))
        and (isinstance(left, int) or isinstance(left, float))
        and right is None
    ):
        return lambda d: rclip(lclip(d, dist1), left)
    else:
        return rclip(lclip(dist1, left), right)


class ConstantDistribution(OperableDistribution):
    def __init__(self, x: Any) -> None:
        super().__init__()
        self.x = x

    def __str__(self) -> str:
        return "<Distribution> const({})".format(self.x)


def const(x: Any) -> ConstantDistribution:
    """
    Initialize a constant distribution.

    Constant distributions always return the same value no matter what.

    Parameters
    ----------
    x : anything
        The value the constant distribution should always return.

    Returns
    -------
    ConstantDistribution

    Examples
    --------
    >>> const(1)
    <Distribution> const(1)
    """
    return ConstantDistribution(x)


class UniformDistribution(OperableDistribution):
    def __init__(self, x: Number, y: Number) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return "<Distribution> uniform({}, {})".format(self.x, self.y)


def uniform(x: Number, y: Number) -> UniformDistribution:
    """
    Initialize a uniform random distribution.

    Parameters
    ----------
    x : float
        The smallest value the uniform distribution will return.
    y : float
        The largest value the uniform distribution will return.

    Returns
    -------
    UniformDistribution

    Examples
    --------
    >>> uniform(0, 1)
    <Distribution> uniform(0, 1)
    """
    return UniformDistribution(x=x, y=y)


class NormalDistribution(OperableDistribution):
    def __init__(
        self,
        x: Optional[Number] = None,
        y: Optional[Number] = None,
        mean: Optional[Number] = None,
        sd: Optional[Number] = None,
        credibility: Number = 90,
        lclip: Optional[Number] = None,
        rclip: Optional[Number] = None,
    ) -> None:
        super().__init__()
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip

        self.x: Optional[Number]
        self.y: Optional[Number]
        self.mean: Optional[Number]
        self.sd: Optional[Number]

        # Define the complementary set of parameters
        # x/y => mean/sd, mean/sd => x/y
        if mean is None and sd is None and x is not None and y is not None:
            if x > y:
                raise ValueError("`high value` (y) cannot be lower than `low value` (x)")
            self.x, self.y = x, y
            self.mean = (self.x + self.y) / 2
            cdf_value: Float = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma: np.float64 = stats.norm.ppf(cdf_value)  # type: ignore
            assert self.mean is not None
            self.sd = (self.y - self.mean) / normed_sigma
        elif sd is not None and x is None and y is None:
            self.sd = sd
            self.mean = 0 if mean is None else mean
            self.x = None
            self.y = None
        else:
            raise ValueError("you must define either x/y or mean/sd")

    def __str__(self) -> str:
        assert self.mean is not None and self.sd is not None
        out = "<Distribution> norm(mean={}, sd={}".format(round(self.mean, 2), round(self.sd, 2))
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def norm(
    x: Optional[Number] = None,
    y: Optional[Number] = None,
    credibility: Integer = 90,
    mean: Optional[Number] = None,
    sd: Optional[Number] = None,
    lclip: Optional[Number] = None,
    rclip: Optional[Number] = None,
) -> NormalDistribution:
    """
    Initialize a normal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    mean : float or None
        The mean of the normal distribution. If not defined, defaults to 0.
    sd : float
        The standard deviation of the normal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    NormalDistribution

    Examples
    --------
    >>> norm(0, 1)
    <Distribution> norm(mean=0.5, sd=0.3)
    >>> norm(mean=1, sd=2)
    <Distribution> norm(mean=1, sd=2)
    """
    return NormalDistribution(
        x=x, y=y, credibility=credibility, mean=mean, sd=sd, lclip=lclip, rclip=rclip
    )


class LognormalDistribution(OperableDistribution):
    def __init__(
        self,
        x: Optional[Number] = None,
        y: Optional[Number] = None,
        norm_mean: Optional[Number] = None,
        norm_sd: Optional[Number] = None,
        lognorm_mean: Optional[Number] = None,
        lognorm_sd: Optional[Number] = None,
        credibility: Integer = 90,
        lclip: Optional[Number] = None,
        rclip: Optional[Number] = None,
    ) -> None:
        super().__init__()
        self.x: Optional[Number] = x
        self.y: Optional[Number] = y
        self.credibility: Integer = credibility
        self.norm_mean: Optional[Number] = norm_mean
        self.norm_sd: Optional[Number] = norm_sd
        self.lognorm_mean: Optional[Number] = lognorm_mean
        self.lognorm_sd: Optional[Number] = lognorm_sd
        self.lclip: Optional[Number] = lclip
        self.rclip: Optional[Number] = rclip

        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")
        if self.x is not None and self.x <= 0:
            raise ValueError("lognormal distribution must have values > 0")

        if (self.x is None or self.y is None) and self.norm_sd is None and self.lognorm_sd is None:
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif (self.x is not None or self.y is not None) and (
            self.norm_sd is not None or self.lognorm_sd is not None
        ):
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif (self.norm_sd is not None or self.norm_mean is not None) and (
            self.lognorm_sd is not None or self.lognorm_mean is not None
        ):
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif self.norm_sd is not None and self.norm_mean is None:
            self.norm_mean = 0
        elif self.lognorm_sd is not None and self.lognorm_mean is None:
            self.lognorm_mean = 1

        if self.x is not None and self.y is not None:
            self.norm_mean = (np.log(self.x) + np.log(self.y)) / 2
            cdf_value = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma = stats.norm.ppf(cdf_value)
            self.norm_sd = (np.log(self.y) - self.norm_mean) / normed_sigma

        if self.lognorm_sd is None:
            assert self.norm_sd is not None and self.norm_mean is not None
            self.lognorm_mean = np.exp(self.norm_mean + self.norm_sd**2 / 2)
            self.lognorm_sd = (
                (np.exp(self.norm_sd**2) - 1) * np.exp(2 * self.norm_mean + self.norm_sd**2)
            ) ** 0.5
        elif self.norm_sd is None:
            assert self.lognorm_sd is not None and self.lognorm_mean is not None
            self.norm_mean = np.log(
                (self.lognorm_mean**2 / np.sqrt(self.lognorm_sd**2 + self.lognorm_mean**2))
            )
            self.norm_sd = np.sqrt(np.log(1 + self.lognorm_sd**2 / self.lognorm_mean**2))

    def __str__(self) -> str:
        assert self.lognorm_mean is not None and self.lognorm_sd is not None
        assert self.norm_mean is not None and self.norm_sd is not None
        out = "<Distribution> lognorm(lognorm_mean={}, lognorm_sd={}, norm_mean={}, norm_sd={}"
        out = out.format(
            round(self.lognorm_mean, 2),
            round(self.lognorm_sd, 2),
            round(self.norm_mean, 2),
            round(self.norm_sd, 2),
        )
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def lognorm(
    x: Optional[Number] = None,
    y: Optional[Number] = None,
    credibility: Integer = 90,
    norm_mean: Optional[Number] = None,
    norm_sd: Optional[Number] = None,
    lognorm_mean: Optional[Number] = None,
    lognorm_sd: Optional[Number] = None,
    lclip: Optional[Number] = None,
    rclip: Optional[Number] = None,
) -> LognormalDistribution:
    """
    Initialize a lognormal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    norm_mean : float or None
        The mean of the underlying normal distribution. If not defined, defaults to 0.
    norm_sd : float
        The standard deviation of the underlying normal distribution.
    lognorm_mean : float or None
        The mean of the lognormal distribution. If not defined, defaults to 1.
    lognorm_sd : float
        The standard deviation of the lognormal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LognormalDistribution

    Examples
    --------
    >>> lognorm(1, 10)
    <Distribution> lognorm(lognorm_mean=4.04, lognorm_sd=3.21, norm_mean=1.15, norm_sd=0.7)
    >>> lognorm(norm_mean=1, norm_sd=2)
    <Distribution> lognorm(lognorm_mean=20.09, lognorm_sd=147.05, norm_mean=1, norm_sd=2)
    >>> lognorm(lognorm_mean=1, lognorm_sd=2)
    <Distribution> lognorm(lognorm_mean=1, lognorm_sd=2, norm_mean=-0.8, norm_sd=1.27)
    """
    return LognormalDistribution(
        x=x,
        y=y,
        credibility=credibility,
        norm_mean=norm_mean,
        norm_sd=norm_sd,
        lognorm_mean=lognorm_mean,
        lognorm_sd=lognorm_sd,
        lclip=lclip,
        rclip=rclip,
    )


def to(
    x: Number,
    y: Number,
    credibility: Integer = 90,
    lclip: Optional[Number] = None,
    rclip: Optional[Number] = None,
) -> Union[LognormalDistribution, NormalDistribution]:
    """
    Initialize a distribution from ``x`` to ``y``.

    The distribution will be lognormal by default, unless ``x`` is less than or equal to 0,
    in which case it will become a normal distribution.

    The distribution will default to be a 90% credible interval between ``x`` and ``y`` unless
    ``credibility`` is passed.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ``LognormalDistribution`` if ``x`` > 0, otherwise a ``NormalDistribution``

    Examples
    --------
    >>> to(1, 10)
    <Distribution> lognorm(mean=1.15, sd=0.7)
    >>> to(-10, 10)
    <Distribution> norm(mean=0.0, sd=6.08)
    """
    if x > 0:
        return lognorm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)
    else:
        return norm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)


class BinomialDistribution(OperableDistribution):
    def __init__(self, n: int, p: Number) -> None:
        super().__init__()
        self.n = n
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be between 0 and 1")

    def __str__(self) -> str:
        return "<Distribution> binomial(n={}, p={})".format(self.n, self.p)


def binomial(n: int, p: Number) -> BinomialDistribution:
    """
    Initialize a binomial distribution.

    Parameters
    ----------
    n : int
        The number of trials.
    p : float
        The probability of success for each trial. Must be between 0 and 1.

    Returns
    -------
    BinomialDistribution

    Examples
    --------
    >>> binomial(1, 0.1)
    <Distribution> binomial(1, 0.1)
    """
    return BinomialDistribution(n=n, p=p)


class BetaDistribution(OperableDistribution):
    def __init__(self, a: Number, b: Number) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return "<Distribution> beta(a={}, b={})".format(self.a, self.b)


def beta(a: Number, b: Number) -> BetaDistribution:
    """
    Initialize a beta distribution.

    Parameters
    ----------
    a : float
        The alpha shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a success.
    b : float
        The beta shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a failure.

    Returns
    -------
    BetaDistribution

    Examples
    --------
    >>> beta(1, 2)
    <Distribution> beta(1, 2)
    """
    return BetaDistribution(a, b)


class BernoulliDistribution(OperableDistribution):
    def __init__(self, p: Number) -> None:
        super().__init__()
        if not isinstance(p, float) or isinstance(p, int):
            raise ValueError("bernoulli p must be a float or int")
        if p < 0 or p > 1:
            raise ValueError("bernoulli p must be 0-1")
        self.p = p

    def __str__(self) -> str:
        return "<Distribution> bernoulli(p={})".format(self.p)


def bernoulli(p: Number) -> BernoulliDistribution:
    """
    Initialize a Bernoulli distribution.

    Parameters
    ----------
    p : float
        The probability of the binary event. Must be between 0 and 1.

    Returns
    -------
    BernoulliDistribution

    Examples
    --------
    >>> bernoulli(0.1)
    <Distribution> bernoulli(p=0.1)
    """
    return BernoulliDistribution(p)


class DiscreteDistribution(OperableDistribution):
    def __init__(self, items: Any) -> None:
        super().__init__()
        if not isinstance(items, dict) and not isinstance(items, list) and not _is_numpy(items):
            raise ValueError("inputs to discrete must be a dict or list")
        self.items = list(items) if _is_numpy(items) else items

    def __str__(self) -> str:
        return "<Distribution> discrete({})".format(self.items)


def discrete(items: Union[dict[Any, Number], list[list[Number]]]) -> DiscreteDistribution:
    """
    Initialize a discrete distribution (aka categorical distribution).

    Parameters
    ----------
    items : list or dict
        The values that the discrete distribution will return and their associated
        weights (or likelihoods of being returned when sampled).

    Returns
    -------
    DiscreteDistribution

    Examples
    --------
    >>> discrete({0: 0.1, 1: 0.9})  # 10% chance of returning 0, 90% chance of returning 1
    <Distribution> discrete({0: 0.1, 1: 0.9})
    >>> discrete([[0.1, 0], [0.9, 1]])  # Different notation for the same thing.
    <Distribution> discrete([[0.1, 0], [0.9, 1]])
    >>> discrete([0, 1, 2])  # When no weights are given, all have equal chance of happening.
    <Distribution> discrete([0, 1, 2])
    >>> discrete({'a': 0.1, 'b': 0.9})  # Values do not have to be numbers.
    <Distribution> discrete({'a': 0.1, 'b': 0.9})
    """
    return DiscreteDistribution(items)


class TDistribution(OperableDistribution):
    def __init__(
        self,
        x: Optional[Number] = None,
        y: Optional[Number] = None,
        t: Integer = 20,
        credibility: Integer = 90,
        lclip: Optional[Number] = None,
        rclip: Optional[Number] = None,
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError("must define either both `x` and `y` or neither.")
        elif self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")

        if self.x is None:
            self.credibility = None

    def __str__(self) -> str:
        if self.x is not None:
            out = "<Distribution> tdist(x={}, y={}, t={}".format(self.x, self.y, self.t)
        else:
            out = "<Distribution> tdist(t={}".format(self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ", credibility={}".format(self.credibility)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def tdist(
    x: Optional[Number] = None,
    y: Optional[Number] = None,
    t: Integer = 20,
    credibility: Integer = 90,
    lclip: Optional[Number] = None,
    rclip: Optional[Number] = None,
) -> TDistribution:
    """
    Initialize a t-distribution.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : integer
        The number of degrees of freedom of the t-distribution. Defaults to 20.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    TDistribution

    Examples
    --------
    >>> tdist(0, 1, 2)
    <Distribution> tdist(x=0, y=1, t=2)
    >>> tdist()
    <Distribution> tdist(t=1)
    """
    return TDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class LogTDistribution(OperableDistribution):
    def __init__(
        self,
        x: Optional[Integer] = None,
        y: Optional[Integer] = None,
        t: int = 1,
        credibility: int = 90,
        lclip: Optional[Integer] = None,
        rclip: Optional[Integer] = None,
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError("must define either both `x` and `y` or neither.")
        elif self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")

        if self.x is None:
            self.credibility = None

    def __str__(self) -> str:
        if self.x is not None:
            out = "<Distribution> log_tdist(x={}, y={}, t={}".format(self.x, self.y, self.t)
        else:
            out = "<Distribution> log_tdist(t={}".format(self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ", credibility={}".format(self.credibility)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def log_tdist(
    x: Optional[Integer] = None,
    y: Optional[Integer] = None,
    t: int = 1,
    credibility: int = 90,
    lclip: Optional[Integer] = None,
    rclip: Optional[Integer] = None,
) -> LogTDistribution:
    """
    Initialize a log t-distribution, which is a t-distribution in log-space.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom, but in log-space.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 1.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LogTDistribution

    Examples
    --------
    >>> log_tdist(0, 1, 2)
    <Distribution> log_tdist(x=0, y=1, t=2)
    >>> log_tdist()
    <Distribution> log_tdist(t=1)
    """
    return LogTDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class TriangularDistribution(OperableDistribution):
    def __init__(
        self,
        left: int,
        mode: int,
        right: int,
        lclip: Optional[Integer] = None,
        rclip: Optional[Integer] = None,
    ) -> None:
        super().__init__()
        self.left = left
        self.mode = mode
        self.right = right
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self) -> str:
        out = "<Distribution> triangular({}, {}, {}".format(self.left, self.mode, self.right)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def triangular(
    left: int,
    mode: int,
    right: int,
    lclip: Optional[Integer] = None,
    rclip: Optional[Integer] = None,
) -> TriangularDistribution:
    """
    Initialize a triangular distribution.

    Parameters
    ----------
    left : float
        The smallest value of the triangular distribution.
    mode : float
        The most common value of the triangular distribution.
    right : float
        The largest value of the triangular distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    TriangularDistribution

    Examples
    --------
    >>> triangular(1, 2, 3)
    <Distribution> triangular(1, 2, 3)
    """
    return TriangularDistribution(left=left, mode=mode, right=right, lclip=lclip, rclip=rclip)


class PoissonDistribution(OperableDistribution):
    def __init__(
        self, lam: Number, lclip: Optional[Integer] = None, rclip: Optional[Integer] = None
    ) -> None:
        super().__init__()
        self.lam = lam
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self) -> str:
        out = "<Distribution> poisson({}".format(self.lam)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def poisson(
    lam: Number, lclip: Optional[Integer] = None, rclip: Optional[Integer] = None
) -> PoissonDistribution:
    """
    Initialize a poisson distribution.

    Parameters
    ----------
    lam : float
        The lambda value of the poisson distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    PoissonDistribution

    Examples
    --------
    >>> poisson(1)
    <Distribution> poisson(1)
    """
    return PoissonDistribution(lam=lam, lclip=lclip, rclip=rclip)


class ChiSquareDistribution(OperableDistribution):
    def __init__(self, df: int) -> None:
        super().__init__()
        self.df = df
        if self.df <= 0:
            raise ValueError("df must be positive")

    def __str__(self) -> str:
        return "<Distribution> chisquare({})".format(self.df)


def chisquare(df: int) -> ChiSquareDistribution:
    """
    Initialize a chi-square distribution.

    Parameters
    ----------
    df : float
        The degrees of freedom. Must be positive.

    Returns
    -------
    ChiSquareDistribution

    Examples
    --------
    >>> chisquare(2)
    <Distribution> chiaquare(2)
    """
    return ChiSquareDistribution(df=df)


class ExponentialDistribution(OperableDistribution):
    def __init__(
        self, scale: int, lclip: Optional[Integer] = None, rclip: Optional[Integer] = None
    ) -> None:
        super().__init__()
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self) -> str:
        out = "<Distribution> exponential({}".format(self.scale)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def exponential(
    scale: int, lclip: Optional[Integer] = None, rclip: Optional[Integer] = None
) -> ExponentialDistribution:
    """
    Initialize an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale value of the exponential distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ExponentialDistribution

    Examples
    --------
    >>> exponential(1)
    <Distribution> exponential(1)
    """
    return ExponentialDistribution(scale=scale, lclip=lclip, rclip=rclip)


class GammaDistribution(OperableDistribution):
    def __init__(
        self,
        shape: int,
        scale: int = 1,
        lclip: Optional[Integer] = None,
        rclip: Optional[Integer] = None,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self) -> str:
        out = "<Distribution> gamma(shape={}, scale={}".format(self.shape, self.scale)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def gamma(
    shape: int, scale: int = 1, lclip: Optional[Integer] = None, rclip: Optional[Integer] = None
) -> GammaDistribution:
    """
    Initialize a gamma distribution.

    Parameters
    ----------
    shape : float
        The shape value of the gamma distribution.
    scale : float
        The scale value of the gamma distribution. Defaults to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    GammaDistribution

    Examples
    --------
    >>> gamma(10, 1)
    <Distribution> gamma(shape=10, scale=1)
    """
    return GammaDistribution(shape=shape, scale=scale, lclip=lclip, rclip=rclip)


class ParetoDistribution(OperableDistribution):
    def __init__(self, shape: int) -> None:
        super().__init__()
        self.shape = shape

    def __str__(self) -> str:
        return "<Distribution> pareto({})".format(self.shape)


def pareto(shape: int) -> ParetoDistribution:
    """
    Initialize a pareto distribution.

    Parameters
    ----------
    shape : float
        The shape value of the pareto distribution.

    Returns
    -------
    ParetoDistribution

    Examples
    --------
    >>> pareto(1)
    <Distribution> pareto(1)
    """
    return ParetoDistribution(shape=shape)


class MixtureDistribution(OperableDistribution):
    def __init__(
        self,
        dists: Any,
        weights: Optional[Weights] = None,
        relative_weights: Optional[Weights] = None,
        lclip: Optional[Number] = None,
        rclip: Optional[Number] = None,
    ) -> None:
        super().__init__()
        weights, dists = _process_weights_values(weights, relative_weights, dists)
        self.dists = dists
        self.weights = weights
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self) -> str:
        out = "<Distribution> mixture"
        assert self.weights is not None
        for i in range(len(self.dists)):
            out += "\n - {} weight on {}".format(self.weights[i], self.dists[i])
        return out


def mixture(
    dists: Any,
    weights: Optional[Weights] = None,
    relative_weights: Optional[Weights] = None,
    lclip: Optional[Integer] = None,
    rclip: Optional[Integer] = None,
) -> MixtureDistribution:
    """
    Initialize a mixture distribution, which is a combination of different distributions.

    Parameters
    ----------
    dists : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> mixture([norm(1, 2), norm(3, 4)], weights=[0.1, 0.9])
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    >>> mixture([[0.1, norm(1, 2)], [0.9, norm(3, 4)]])  # Different notation for the same thing.
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    >>> mixture([norm(1, 2), norm(3, 4)])  # When no weights are given, all have equal chance
    >>>                                    # of happening.
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    """
    return MixtureDistribution(
        dists=dists,
        weights=weights,
        relative_weights=relative_weights,
        lclip=lclip,
        rclip=rclip,
    )


def zero_inflated(p_zero: float, dist: NormalDistribution) -> MixtureDistribution:
    """
    Initialize an arbitrary zero-inflated distribution.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> zero_inflated(0.6, norm(1, 2))
    <Distribution> mixture
     - 0
     - <Distribution> norm(mean=1.5, sd=0.3)
    """
    if p_zero > 1 or p_zero < 0 or not isinstance(p_zero, float):
        raise ValueError("`p_zero` must be between 0 and 1")
    return MixtureDistribution(dists=[0, dist], weights=[p_zero])


def inf0(p_zero: float, dist: NormalDistribution) -> MixtureDistribution:
    """
    Initialize an arbitrary zero-inflated distribution.

    Alias for ``zero_inflated``.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> inf0(0.6, norm(1, 2))
    <Distribution> mixture
     - 0
     - <Distribution> norm(mean=1.5, sd=0.3)
    """
    return zero_inflated(p_zero=p_zero, dist=dist)
