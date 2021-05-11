"""Marching Cubes"""
import numpy as np

# This table contains the triangle configuration for each of the 256 possible cube configurations.
# Each configuration contains at most 15 entries which corresponds to 5 triangles, each with 3 edge indices
# -1 is used as a place holder and should be discarded in your implementation
triangle_table = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 8, 9, 3, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [10, 2, 1, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 0, 10, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 9, 10, 8, 10, 2, 3, 8, 2, -1, -1, -1, -1, -1, -1],
    [2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [0, 11, 8, 2, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 3, 2, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [11, 8, 9, 11, 9, 1, 2, 11, 1, -1, -1, -1, -1, -1, -1],
    [3, 10, 11, 1, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1], [10, 11, 8, 10, 8, 0, 1, 10, 0, -1, -1, -1, -1, -1, -1],
    [9, 10, 11, 9, 11, 3, 0, 9, 3, -1, -1, -1, -1, -1, -1], [11, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 3, 7, 0, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 4, 8, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 3, 7, 1, 7, 4, 9, 1, 4, -1, -1, -1, -1, -1, -1],
    [7, 4, 8, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [10, 2, 1, 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1],
    [7, 4, 8, 2, 0, 9, 10, 2, 9, -1, -1, -1, -1, -1, -1], [4, 9, 7, 3, 7, 2, 7, 9, 2, 9, 10, 2, -1, -1, -1],
    [2, 11, 3, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 0, 2, 4, 2, 11, 7, 4, 11, -1, -1, -1, -1, -1, -1],
    [11, 3, 2, 7, 4, 8, 1, 0, 9, -1, -1, -1, -1, -1, -1], [1, 2, 9, 2, 11, 9, 11, 4, 9, 11, 7, 4, -1, -1, -1],
    [4, 8, 7, 10, 11, 3, 1, 10, 3, -1, -1, -1, -1, -1, -1], [4, 11, 7, 4, 0, 1, 11, 4, 1, 10, 11, 1, -1, -1, -1],
    [3, 0, 11, 10, 11, 9, 11, 0, 9, 8, 7, 4, -1, -1, -1], [10, 11, 9, 9, 11, 4, 11, 7, 4, -1, -1, -1, -1, -1, -1],
    [4, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 8, 0, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 1, 4, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 1, 3, 5, 3, 8, 4, 5, 8, -1, -1, -1, -1, -1, -1],
    [4, 5, 9, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 9, 4, 10, 2, 1, 8, 0, 3, -1, -1, -1, -1, -1, -1],
    [2, 0, 4, 2, 4, 5, 10, 2, 5, -1, -1, -1, -1, -1, -1], [8, 4, 3, 4, 5, 3, 5, 2, 3, 5, 10, 2, -1, -1, -1],
    [11, 3, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 9, 4, 11, 8, 0, 2, 11, 0, -1, -1, -1, -1, -1, -1],
    [11, 3, 2, 5, 1, 0, 4, 5, 0, -1, -1, -1, -1, -1, -1], [5, 8, 4, 11, 8, 2, 8, 5, 2, 5, 1, 2, -1, -1, -1],
    [4, 5, 9, 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1], [10, 11, 8, 1, 10, 8, 1, 8, 0, 5, 9, 4, -1, -1, -1],
    [3, 0, 11, 10, 11, 5, 11, 0, 5, 0, 4, 5, -1, -1, -1], [11, 8, 10, 10, 8, 5, 8, 4, 5, -1, -1, -1, -1, -1, -1],
    [9, 7, 5, 8, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 7, 5, 3, 5, 9, 0, 3, 9, -1, -1, -1, -1, -1, -1],
    [7, 5, 1, 7, 1, 0, 8, 7, 0, -1, -1, -1, -1, -1, -1], [7, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 10, 7, 5, 9, 8, 7, 9, -1, -1, -1, -1, -1, -1], [3, 7, 5, 0, 3, 5, 0, 5, 9, 2, 1, 10, -1, -1, -1],
    [2, 5, 10, 7, 5, 8, 5, 2, 8, 2, 0, 8, -1, -1, -1], [7, 5, 3, 3, 5, 2, 5, 10, 2, -1, -1, -1, -1, -1, -1],
    [2, 11, 3, 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1], [11, 7, 2, 0, 2, 9, 2, 7, 9, 7, 5, 9, -1, -1, -1],
    [7, 5, 1, 8, 7, 1, 8, 1, 0, 11, 3, 2, -1, -1, -1], [5, 1, 7, 7, 1, 11, 1, 2, 11, -1, -1, -1, -1, -1, -1],
    [11, 3, 10, 3, 1, 10, 7, 5, 8, 8, 5, 9, -1, -1, -1], [0, 10, 11, 10, 0, 1, 0, 11, 7, 9, 0, 5, 0, 7, 5],
    [0, 7, 5, 7, 0, 8, 0, 5, 10, 3, 0, 11, 0, 10, 11], [5, 11, 7, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 10, 5, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 10, 5, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 10, 5, 8, 9, 1, 3, 8, 1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 5, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 0, 3, 6, 2, 1, 5, 6, 1, -1, -1, -1, -1, -1, -1],
    [6, 2, 0, 6, 0, 9, 5, 6, 9, -1, -1, -1, -1, -1, -1], [8, 2, 3, 6, 2, 5, 2, 8, 5, 8, 9, 5, -1, -1, -1],
    [5, 6, 10, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 6, 10, 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1],
    [6, 10, 5, 11, 3, 2, 9, 1, 0, -1, -1, -1, -1, -1, -1], [11, 8, 9, 2, 11, 9, 2, 9, 1, 6, 10, 5, -1, -1, -1],
    [3, 1, 5, 3, 5, 6, 11, 3, 6, -1, -1, -1, -1, -1, -1], [6, 11, 5, 1, 5, 0, 5, 11, 0, 11, 8, 0, -1, -1, -1],
    [9, 5, 0, 5, 6, 0, 6, 3, 0, 6, 11, 3, -1, -1, -1], [8, 9, 11, 11, 9, 6, 9, 5, 6, -1, -1, -1, -1, -1, -1],
    [8, 7, 4, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1], [10, 5, 6, 3, 7, 4, 0, 3, 4, -1, -1, -1, -1, -1, -1],
    [7, 4, 8, 6, 10, 5, 0, 9, 1, -1, -1, -1, -1, -1, -1], [4, 9, 7, 3, 7, 1, 7, 9, 1, 5, 6, 10, -1, -1, -1],
    [8, 7, 4, 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1], [7, 4, 3, 4, 0, 3, 6, 2, 5, 5, 2, 1, -1, -1, -1],
    [6, 2, 0, 5, 6, 0, 5, 0, 9, 7, 4, 8, -1, -1, -1], [9, 6, 2, 6, 9, 5, 9, 2, 3, 4, 9, 7, 9, 3, 7],
    [5, 6, 10, 4, 8, 7, 2, 11, 3, -1, -1, -1, -1, -1, -1], [11, 7, 2, 0, 2, 4, 2, 7, 4, 6, 10, 5, -1, -1, -1],
    [6, 10, 5, 11, 3, 2, 8, 7, 4, 9, 1, 0, -1, -1, -1], [6, 10, 5, 4, 11, 7, 11, 4, 9, 2, 11, 9, 1, 2, 9],
    [6, 11, 5, 1, 5, 3, 5, 11, 3, 7, 4, 8, -1, -1, -1], [11, 4, 0, 4, 11, 7, 11, 0, 1, 6, 11, 5, 11, 1, 5],
    [7, 4, 8, 3, 6, 11, 6, 3, 0, 5, 6, 0, 9, 5, 0], [9, 11, 7, 9, 7, 4, 11, 9, 6, 9, 5, 6, -1, -1, -1],
    [10, 4, 6, 9, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 8, 0, 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1],
    [0, 4, 6, 0, 6, 10, 1, 0, 10, -1, -1, -1, -1, -1, -1], [10, 1, 6, 4, 6, 8, 6, 1, 8, 1, 3, 8, -1, -1, -1],
    [4, 6, 2, 4, 2, 1, 9, 4, 1, -1, -1, -1, -1, -1, -1], [4, 6, 2, 9, 4, 2, 9, 2, 1, 8, 0, 3, -1, -1, -1],
    [6, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 2, 4, 4, 2, 8, 2, 3, 8, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, 4, 6, 10, 9, 4, 10, -1, -1, -1, -1, -1, -1], [6, 10, 4, 10, 9, 4, 11, 8, 2, 2, 8, 0, -1, -1, -1],
    [10, 1, 6, 4, 6, 0, 6, 1, 0, 2, 11, 3, -1, -1, -1], [1, 11, 8, 11, 1, 2, 1, 8, 4, 10, 1, 6, 1, 4, 6],
    [3, 6, 11, 3, 1, 9, 6, 3, 9, 4, 6, 9, -1, -1, -1], [1, 4, 6, 4, 1, 9, 1, 6, 11, 0, 1, 8, 1, 11, 8],
    [4, 6, 0, 0, 6, 3, 6, 11, 3, -1, -1, -1, -1, -1, -1], [8, 6, 11, 8, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 9, 8, 10, 8, 7, 6, 10, 7, -1, -1, -1, -1, -1, -1], [10, 7, 6, 10, 9, 0, 7, 10, 0, 3, 7, 0, -1, -1, -1],
    [0, 8, 1, 8, 7, 1, 7, 10, 1, 7, 6, 10, -1, -1, -1], [3, 7, 1, 1, 7, 10, 7, 6, 10, -1, -1, -1, -1, -1, -1],
    [7, 6, 8, 9, 8, 1, 8, 6, 1, 6, 2, 1, -1, -1, -1], [9, 3, 7, 3, 9, 0, 9, 7, 6, 1, 9, 2, 9, 6, 2],
    [2, 0, 6, 6, 0, 7, 0, 8, 7, -1, -1, -1, -1, -1, -1], [2, 7, 6, 2, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 8, 9, 8, 10, 8, 6, 10, 11, 3, 2, -1, -1, -1], [7, 10, 9, 10, 7, 6, 7, 9, 0, 11, 7, 2, 7, 0, 2],
    [11, 3, 2, 10, 7, 6, 7, 10, 1, 8, 7, 1, 0, 8, 1], [1, 7, 6, 1, 6, 10, 7, 1, 11, 1, 2, 11, -1, -1, -1],
    [6, 3, 1, 3, 6, 11, 6, 1, 9, 7, 6, 8, 6, 9, 8], [7, 6, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 6, 11, 0, 11, 3, 6, 0, 7, 0, 8, 7, -1, -1, -1], [6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 7, 11, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 11, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 7, 11, 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1],
    [7, 11, 6, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1], [7, 11, 6, 8, 0, 3, 10, 2, 1, -1, -1, -1, -1, -1, -1],
    [7, 11, 6, 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1], [8, 9, 10, 3, 8, 10, 3, 10, 2, 7, 11, 6, -1, -1, -1],
    [7, 2, 6, 3, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1], [0, 2, 6, 0, 6, 7, 8, 0, 7, -1, -1, -1, -1, -1, -1],
    [9, 1, 0, 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1], [6, 7, 8, 8, 9, 1, 6, 8, 1, 2, 6, 1, -1, -1, -1],
    [7, 3, 1, 7, 1, 10, 6, 7, 10, -1, -1, -1, -1, -1, -1], [8, 0, 1, 7, 8, 1, 10, 7, 1, 6, 7, 10, -1, -1, -1],
    [7, 10, 6, 9, 10, 0, 10, 7, 0, 7, 3, 0, -1, -1, -1], [9, 10, 8, 8, 10, 7, 10, 6, 7, -1, -1, -1, -1, -1, -1],
    [6, 8, 11, 4, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 4, 0, 6, 0, 3, 11, 6, 3, -1, -1, -1, -1, -1, -1],
    [1, 0, 9, 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1], [6, 3, 11, 1, 3, 9, 3, 6, 9, 6, 4, 9, -1, -1, -1],
    [1, 10, 2, 8, 11, 6, 4, 8, 6, -1, -1, -1, -1, -1, -1], [6, 4, 0, 11, 6, 0, 11, 0, 3, 10, 2, 1, -1, -1, -1],
    [9, 10, 2, 9, 2, 0, 11, 6, 4, 8, 11, 4, -1, -1, -1], [3, 6, 4, 6, 3, 11, 3, 4, 9, 2, 3, 10, 3, 9, 10],
    [2, 6, 4, 2, 4, 8, 3, 2, 8, -1, -1, -1, -1, -1, -1], [2, 6, 4, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 4, 6, 4, 2, 4, 3, 2, 0, 9, 1, -1, -1, -1], [6, 4, 2, 2, 4, 1, 4, 9, 1, -1, -1, -1, -1, -1, -1],
    [1, 10, 6, 6, 4, 8, 1, 6, 8, 3, 1, 8, -1, -1, -1], [4, 0, 6, 6, 0, 10, 0, 1, 10, -1, -1, -1, -1, -1, -1],
    [3, 9, 10, 9, 3, 0, 3, 10, 6, 8, 3, 4, 3, 6, 4], [4, 10, 6, 4, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 7, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 7, 11, 5, 9, 4, 3, 8, 0, -1, -1, -1, -1, -1, -1],
    [11, 6, 7, 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1], [5, 1, 3, 4, 5, 3, 4, 3, 8, 6, 7, 11, -1, -1, -1],
    [11, 6, 7, 2, 1, 10, 4, 5, 9, -1, -1, -1, -1, -1, -1], [5, 9, 4, 3, 8, 0, 10, 2, 1, 7, 11, 6, -1, -1, -1],
    [2, 0, 4, 10, 2, 4, 10, 4, 5, 11, 6, 7, -1, -1, -1], [6, 7, 11, 2, 5, 10, 5, 2, 3, 4, 5, 3, 8, 4, 3],
    [9, 4, 5, 2, 6, 7, 3, 2, 7, -1, -1, -1, -1, -1, -1], [7, 8, 6, 2, 6, 0, 6, 8, 0, 4, 5, 9, -1, -1, -1],
    [0, 4, 5, 0, 5, 1, 6, 7, 3, 2, 6, 3, -1, -1, -1], [8, 5, 1, 5, 8, 4, 8, 1, 2, 7, 8, 6, 8, 2, 6],
    [7, 3, 1, 6, 7, 1, 6, 1, 10, 4, 5, 9, -1, -1, -1], [4, 5, 9, 0, 7, 8, 7, 0, 1, 6, 7, 1, 10, 6, 1],
    [10, 7, 3, 7, 10, 6, 10, 3, 0, 5, 10, 4, 10, 0, 4], [10, 8, 4, 10, 4, 5, 8, 10, 7, 10, 6, 7, -1, -1, -1],
    [9, 8, 11, 9, 11, 6, 5, 9, 6, -1, -1, -1, -1, -1, -1], [5, 9, 0, 6, 5, 0, 3, 6, 0, 11, 6, 3, -1, -1, -1],
    [11, 6, 5, 5, 1, 0, 11, 5, 0, 8, 11, 0, -1, -1, -1], [1, 3, 5, 5, 3, 6, 3, 11, 6, -1, -1, -1, -1, -1, -1],
    [6, 5, 11, 8, 11, 9, 11, 5, 9, 10, 2, 1, -1, -1, -1], [10, 2, 1, 9, 6, 5, 6, 9, 0, 11, 6, 0, 3, 11, 0],
    [5, 2, 0, 2, 5, 10, 5, 0, 8, 6, 5, 11, 5, 8, 11], [3, 5, 10, 3, 10, 2, 5, 3, 6, 3, 11, 6, -1, -1, -1],
    [2, 8, 3, 2, 6, 5, 8, 2, 5, 9, 8, 5, -1, -1, -1], [2, 6, 0, 0, 6, 9, 6, 5, 9, -1, -1, -1, -1, -1, -1],
    [8, 2, 6, 2, 8, 3, 8, 6, 5, 0, 8, 1, 8, 5, 1], [6, 1, 2, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 9, 8, 9, 6, 5, 6, 8, 3, 10, 6, 1, 6, 3, 1], [0, 6, 5, 0, 5, 9, 6, 0, 10, 0, 1, 10, -1, -1, -1],
    [10, 6, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 7, 10, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1], [0, 3, 8, 5, 7, 11, 10, 5, 11, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1], [1, 3, 8, 1, 8, 9, 7, 11, 10, 5, 7, 10, -1, -1, -1],
    [1, 5, 7, 1, 7, 11, 2, 1, 11, -1, -1, -1, -1, -1, -1], [11, 2, 7, 5, 7, 1, 7, 2, 1, 3, 8, 0, -1, -1, -1],
    [7, 11, 2, 2, 0, 9, 7, 2, 9, 5, 7, 9, -1, -1, -1], [2, 8, 9, 8, 2, 3, 2, 9, 5, 11, 2, 7, 2, 5, 7],
    [5, 7, 3, 5, 3, 2, 10, 5, 2, -1, -1, -1, -1, -1, -1], [5, 2, 10, 5, 7, 8, 2, 5, 8, 0, 2, 8, -1, -1, -1],
    [2, 10, 3, 7, 3, 5, 3, 10, 5, 1, 0, 9, -1, -1, -1], [2, 5, 7, 5, 2, 10, 2, 7, 8, 1, 2, 9, 2, 8, 9],
    [5, 7, 3, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 7, 1, 1, 7, 0, 7, 8, 0, -1, -1, -1, -1, -1, -1],
    [7, 3, 5, 5, 3, 9, 3, 0, 9, -1, -1, -1, -1, -1, -1], [7, 9, 5, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 11, 10, 8, 10, 5, 4, 8, 5, -1, -1, -1, -1, -1, -1], [0, 3, 11, 11, 10, 5, 0, 11, 5, 4, 0, 5, -1, -1, -1],
    [5, 4, 10, 11, 10, 8, 10, 4, 8, 9, 1, 0, -1, -1, -1], [4, 1, 3, 1, 4, 9, 4, 3, 11, 5, 4, 10, 4, 11, 10],
    [8, 5, 4, 8, 11, 2, 5, 8, 2, 1, 5, 2, -1, -1, -1], [11, 1, 5, 1, 11, 2, 11, 5, 4, 3, 11, 0, 11, 4, 0],
    [5, 8, 11, 8, 5, 4, 5, 11, 2, 9, 5, 0, 5, 2, 0], [3, 11, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 3, 5, 4, 3, 2, 5, 3, 10, 5, 2, -1, -1, -1], [0, 2, 4, 4, 2, 5, 2, 10, 5, -1, -1, -1, -1, -1, -1],
    [9, 1, 0, 8, 5, 4, 5, 8, 3, 10, 5, 3, 2, 10, 3], [2, 4, 9, 2, 9, 1, 4, 2, 5, 2, 10, 5, -1, -1, -1],
    [1, 5, 3, 3, 5, 8, 5, 4, 8, -1, -1, -1, -1, -1, -1], [5, 0, 1, 5, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 3, 0, 5, 0, 9, 3, 5, 8, 5, 4, 8, -1, -1, -1], [5, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 10, 9, 11, 9, 4, 7, 11, 4, -1, -1, -1, -1, -1, -1], [11, 10, 9, 7, 11, 9, 7, 9, 4, 3, 8, 0, -1, -1, -1],
    [11, 4, 7, 0, 4, 1, 4, 11, 1, 11, 10, 1, -1, -1, -1], [4, 11, 10, 11, 4, 7, 4, 10, 1, 8, 4, 3, 4, 1, 3],
    [2, 1, 9, 11, 2, 9, 4, 11, 9, 7, 11, 4, -1, -1, -1], [3, 8, 0, 1, 11, 2, 11, 1, 9, 7, 11, 9, 4, 7, 9],
    [0, 4, 2, 2, 4, 11, 4, 7, 11, -1, -1, -1, -1, -1, -1], [4, 2, 3, 4, 3, 8, 2, 4, 11, 4, 7, 11, -1, -1, -1],
    [9, 4, 7, 7, 3, 2, 9, 7, 2, 10, 9, 2, -1, -1, -1], [7, 0, 2, 0, 7, 8, 7, 2, 10, 4, 7, 9, 7, 10, 9],
    [10, 0, 4, 0, 10, 1, 10, 4, 7, 2, 10, 3, 10, 7, 3], [4, 7, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 7, 7, 1, 4, 1, 9, 4, -1, -1, -1, -1, -1, -1], [1, 7, 8, 1, 8, 0, 7, 1, 4, 1, 9, 4, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1], [7, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 11, 10, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1], [10, 9, 11, 11, 9, 3, 9, 0, 3, -1, -1, -1, -1, -1, -1],
    [11, 10, 8, 8, 10, 0, 10, 1, 0, -1, -1, -1, -1, -1, -1], [10, 3, 11, 10, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 11, 9, 9, 11, 1, 11, 2, 1, -1, -1, -1, -1, -1, -1], [9, 11, 2, 9, 2, 1, 11, 9, 3, 9, 0, 3, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 2, 8, 3, 2, -1, -1, -1, -1, -1, -1], [2, 9, 0, 2, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 10, 1, 8, 1, 0, 10, 8, 2, 8, 3, 2, -1, -1, -1], [2, 10, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
]


def compute_cube_index(cube: np.array, isolevel=0.) -> int:
    """
    Takes a cube and returns its Marching Cubes index.
    An index is an 8-bit integer value where each bit is defined to be 1 if that corner of the cube is < isolevel and 0 otherwise.
    The corners are defined counter-clockwise, bottom plane first. The exercise notebook contains a visualization of the exact placements.
    Example: A cube intersects the surface such that its top half lies within the shape (= top 4 corner voxels hold negative SDF values) and the bottom half outside (= bottom 4 corner voxels hold positive SDF values).
    Its index should then be 11110000 = 240.
    :param cube: The cube, as array of 8 corner voxels, each holding an sdf value
    :param isolevel: The surface isolevel. In our case, this is always 0.
    :return: The cube index as integer value
    """

    # ###############
    for i,e in enumerate(cube):
        if e < isolevel:
            cube[i] = 1
        else:
            cube[i] = 0
    return int(cube.dot(1 << np.arange(cube.size)[::-1]))
    # ###############


def marching_cubes(sdf: np.array) -> tuple:
    """
    Implements Marching Cubes. Using the incoming sdf grid, do the following for each cube:
    1. Compute cube index
    2. Compute vertex locations for each vertex defined in the triangle_table entry corresponding to the current cube index. Use vertex_interpolation for that
    3. Add these together with the triangles to a global vertex and triangle list and return them
    :param sdf: A cubic, regular grid containing SDF values
    :return: A tuple with (1) a list of vertices (nx3) and (2) a list of faces (mx3)
    """

    global_vertices = []
    global_triangles = []
    # ###############
    vertices_index = 0
    edges_to_corners = np.array([[0,1],[1,2],[3,2],[0,3],[4,5],[5,6],[7,6],[4,7],[0,4],[1,5],[2,6],[3,7]])
    for i,x in enumerate(sdf[:-1]):
        for j,y in enumerate(x[:-1]):
            for k, z in enumerate(y[:-1]):
                base = [i,j,k]
                # Offsets that needs to be added to the corners for exact location
                corner_location_offsets = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
                corners_sdf_values = np.array([sdf[i,j+1,k+1],sdf[i+1,j+1,k+1],sdf[i+1,j,k+1],sdf[i,j,k+1],sdf[i,j+1,k],sdf[i+1,j+1,k],sdf[i+1,j,k],sdf[i,j,k]])
                cube_index = compute_cube_index(corners_sdf_values)
                
                # from the triangle table
                edges_list = np.reshape(triangle_table[cube_index],(-1,3))
                # Remove -1 rows because they are irrelevant.
                edges_list = edges_list[~np.all(edges_list == -1, axis=1)]
                # All faces include 3 points
                for three_edges in edges_list:
                    face = []
                    for p in range(3): # Each face must have consecutive triangles.
                        p1_corner_index,p2_corner_index = edges_to_corners[three_edges[p]]                    
                        p_1 = base + corner_location_offsets[p1_corner_index]
                        p_2 = base + corner_location_offsets[p2_corner_index]
                        vertex = vertex_interpolation(p_1, p_2, v_1=0, v_2=0)
                        global_vertices.append(vertex)
                        face.append(vertices_index)
                        vertices_index+=1
                    global_triangles.append(face)
    # ###############

    return global_vertices, global_triangles


def vertex_interpolation(p_1, p_2, v_1, v_2, isovalue=0.):
    """
    Calculate the vertex location between corner points p_1 and p_2
    :param p_1: Corner point 1
    :param p_2: Corner point 2
    :param v_1: SDF value at corner point 1
    :param v_2: SDF value at corner point 2
    :param isovalue: The iso value, always 0 in our case
    :return: A single point
    """

    return p_1 + (p_2 - p_1) / 2.
