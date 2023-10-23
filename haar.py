"""implementation of haar's estimation"""
from typing import Iterator, TypeAlias
from random import choice

import matplotlib.pyplot as plt  # type: ignore # missing stubs

import numpy as np
from numpy.typing import NDArray
import sympy as sp

x = sp.Symbol('x')  # x axis
j = sp.Symbol('j')  # slice index
d = sp.Symbol('d')  # depth index
f = sp.Function('f')(x)  # f(x)  # pylint:disable = not-callable


__LOWER_BOUND: sp.Expr = j / (2 ** (d - 1))  # slice lower bound
__MIDDLE_POINT: sp.Expr = (j + (1 / 2)) / (2 ** (d - 1))  # slice middle
__UPPER_BOUND: sp.Expr = (j + 1) / (2 ** (d - 1))  # slice upper bound

# used for calculating amplitude
AMP_EXP: sp.Expr = (
    (2 ** (d - 1)) * (
        sp.integrate(f, (x, __LOWER_BOUND, __MIDDLE_POINT)) -
        sp.integrate(f, (x, __MIDDLE_POINT, __UPPER_BOUND))
    )
)
AMP_EXP_AT_0: sp.Expr = sp.integrate(f, (x, 0, 1))


def linspace_from_resolution(res: int):
    """initialize linspace with resolution (number of elements) of 2^res"""
    return np.linspace(0, 1, 2 ** res)


def plot(haar_: 'Haar'):
    """shorthand plot function"""
    lin_space = linspace_from_resolution(haar_.max_depth)
    plt.step(lin_space, haar_.array, where='mid')
    plt.show()


class Haar:
    """1 dimensional haar approximation"""

    # shorthand types
    TData: TypeAlias = NDArray[np.float64]
    PosNegTuple: TypeAlias = tuple[TData, TData]

    def __init__(self, max_depth: int):
        # initialize array with resolution (number of elements) of 2^res
        self.max_depth = max_depth
        self.array = np.zeros(2 ** max_depth)
        self.slices = self.__init_slice_cache(max_depth)

    @classmethod
    def from_amplitudes(cls, amps: dict[tuple[int, int], np.float_], max_depth: int):
        """initialize using known slices->amplitude map"""
        out = cls(max_depth)
        for (depth, index), amp in amps.items():
            amp = np.float64(amp)
            if depth == 0:
                # amplitude is uniform bias at depth = 0
                out.array += amp
                continue
            pos, neg = out.slices[depth][index]
            pos += amp
            neg -= amp
        return out

    @classmethod
    def from_expression(cls, exp: sp.Expr, max_depth: int):
        """initialize using max depth (for resolution), and expression to approximate"""
        amps: dict[tuple[int, int], np.float_] = {
            (depth, index): get_amplitude(exp, depth, index)
            for depth, index in iter_depth(max_depth)
        }
        return cls.from_amplitudes(amps, max_depth)

    @classmethod
    def random(cls, max_depth: int):
        """generates random haar array from amplitudes taken from {-1, 1}"""
        amps = {
            (depth, index): np.float64(choice([-1, 1]))
            for depth, index in iter_depth(max_depth)
        }
        return cls.from_amplitudes(amps, max_depth)

    def __init_slice_cache(self, max_depth: int):
        """
        depth -> slice index -> (pos, neg)
        mapping initializer
        """
        slices: dict[int, dict[int, Haar.PosNegTuple]] = {}
        for depth in range(1, max_depth + 1):
            slices[depth] = {}
            if depth == 1:
                pos, neg, *_ = np.split(self.array, 2)
                slices[depth][0] = (pos, neg)
                continue
            parts = 2 ** (depth - 1)
            partitions = np.split(self.array, parts)
            for index, partition in enumerate(partitions):
                pos, neg, *_ = np.split(partition, 2)
                slices[depth][index] = (pos, neg)
        return slices


def get_amplitude(func: sp.Expr, depth: int, index: int):
    """get amplitude for slice at given depth and index, for a given expression"""
    if depth < 0:
        raise ValueError(f"depth can't be below 0: {depth=}")
    if not 0 <= index <= (2 ** depth):
        raise ValueError(
            f"incorrect range: 0 <= ({index=}) <= ({2 ** depth=})"
        )
    exp = (
        AMP_EXP.subs({d: depth, j: index, f: func, })
        if depth else AMP_EXP_AT_0.subs({f: func})
    )
    return exp.evalf()


def iter_depth(max_depth: int) -> Iterator[tuple[int, int]]:
    """iterates through range of depth and indeces, for a given maximal depth"""
    if max_depth < 0:
        raise ValueError(f"depth cant be below 0: {max_depth=}")
    yield (0, 0)
    if max_depth == 0:
        return
    yield (1, 0)
    if max_depth == 1:
        return
    for depth in range(2, max_depth + 1):
        for index in range(0, 2 ** (depth - 1)):
            yield (depth, index)
