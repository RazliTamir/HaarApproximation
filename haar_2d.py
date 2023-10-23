"""implementation of haar's estimation, 2-dimensional"""
from typing import TypeAlias
from itertools import product
from random import choice

import matplotlib.pyplot as plt  # type: ignore # missing stubs

import numpy as np


Coords: TypeAlias = tuple[int, int]
Edges: TypeAlias = tuple[int, int]


def plot(haar_: 'Haar2D'):
    """shorthand plot function"""
    _, axes = plt.subplots()
    axes.imshow(haar_.array)
    plt.show()


class Haar2D:
    """2 dimensional haar approximation"""
    # pylint: disable=R0903

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.__res = _res_from_max_depth(max_depth)
        self.array = np.zeros((self.__res, self.__res))
        self._sub_rectangles = _possible_rectangles(max_depth)

    @classmethod
    def random(cls, max_depth: int):
        """generates random haar array from amplitudes taken from {-1, 1}"""
        # pylint: disable=R0914
        out = cls(max_depth)
        out.array += choice([-1, 1])
        for _, edge_coords_map in out._sub_rectangles.items():
            for edges, coords_list in edge_coords_map.items():
                for (x_start, y_start) in coords_list:
                    # get partition
                    x_end = x_start + edges[0]
                    y_end = y_start + edges[1]
                    sub_rect = out.array[x_start:x_end, y_start:y_end]
                    # split partition to 4 quarters
                    # and apply amplitude add and substract
                    sub_x, sub_y = sub_rect.shape
                    sub_x_mid, sub_y_mid = sub_x // 2, sub_y // 2
                    amp = choice([-1, 1])
                    sub_rect[:sub_x_mid, :sub_y_mid] += amp
                    sub_rect[sub_x_mid:, sub_y_mid:] += amp
                    sub_rect[:sub_x_mid, sub_y_mid:] -= amp
                    sub_rect[sub_x_mid:, :sub_y_mid] -= amp
        return out


def _res_from_max_depth(max_depth: int) -> int:
    """
    amount of units (resolution) required
    for an array used with Haar2D with a certain maximal depth
    """
    return 2 ** (max_depth + 1)


def _rectsizes(max_depth: int) -> dict[int, list[Edges]]:
    """
    finds all rectangles' edge lengths who's area corresponds to
    `1 / (2 ** d)` of the 2d array used in Haar2D.
    edge length are adjusted to the 2d array's `resolution` (see _res_from_max_depth).

    therefore actual formula for areas of squares in given depth `d`,
    used with array of resolution `r` is `(r^2) / (2 ** d)`

    return type is mapping of depth and list of Edges consistent with the formula
    """
    res = _res_from_max_depth(max_depth)
    return {
        depth: [
            (
                res // (2 ** k),
                res // (2 ** (depth - k))
            )
            for k in range(depth + 1)
        ]
        for depth in range(1, max_depth + 1)
    }


def _find_possible_coords(max_depth: int, depth: int, edges: Edges) -> list[Coords]:
    """
    finds possible coordinates for a rectangle of size `edges`
    in a 2d array with resolution corresponding to `max_depth`
    """
    res = _res_from_max_depth(max_depth)
    step = 2 ** (1 + max_depth - depth)
    edge_x, edge_y = edges
    return list(product(
        range(0, res - (edge_x - 1), step),
        range(0, res - (edge_y - 1), step)
    ))


def _possible_rectangles(max_depth: int) -> dict[int, dict[Edges, list[Coords]]]:
    rect_sizes = _rectsizes(max_depth)
    out = {
        depth: {
            edges: _find_possible_coords(max_depth, depth, edges)
            for edges in edges_list
        }
        for depth, edges_list in rect_sizes.items()
    }
    return out
