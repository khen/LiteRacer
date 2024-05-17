"""Utility functions."""

from numpy import sqrt, round, min, argwhere
import random
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate


def random_points_in_polygon(polygon, k):
    """Return list of k points chosen uniformly at random inside the polygon.

    See explanation here: https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    """

    areas = []
    transforms = []
    for t in triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)
        points.append(affine_transform(p, transform))
    return points


def distance_from_curve(X, Y, x, y):
    """Return distance between point and a curve."""

    d_x = X - x
    d_y = Y - y
    dis = sqrt( d_x**2 + d_y**2 )
    return dis


def min_distance_from_curve(X, Y, x, y, precision=5):
    """Compute minimum/a distance/s between a point and a curve rounded at `precision`.
        
    Returns min indexes and distances array.
    """

    # compute distance
    distances = distance_from_curve(X, Y, x, y)
    distances = round(distances, precision)
    # find the minima
    min_idxs = argwhere(distances==min(distances)).ravel()
    return min_idxs, distances