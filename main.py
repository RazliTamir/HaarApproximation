"""main script"""

import sympy as sp

from haar import Haar, plot as plot_1d
from haar_2d import Haar2D, plot as plot_2d


def main():
    """entrypoint"""
    max_depth = 8
    # Example 1: Expression
    f: sp.Expr = sp.Symbol('x')
    haar_1d = Haar.from_expression(f, max_depth)
    plot_1d(haar_1d)
    # Example 2: random 1D
    haar_1d = Haar.random(max_depth)
    plot_1d(haar_1d)
    # Example 3: Random 2D
    haar_2d = Haar2D.random(max_depth)
    plot_2d(haar_2d)


if __name__ == "__main__":
    main()
