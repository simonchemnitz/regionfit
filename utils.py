import numpy as np


def ellipse(data: np.ndarray, alpha=0.05):
    """
    Return height and width of an ellipse
    that encircles data

    Parameters:
    -----------
    data: np.ndarray
        Rotated and centered data
        to fit ellipse to
    """
    # effect radius
    er = np.sqrt(-2 * np.log(alpha))

    x = data[:, 0]
    y = data[:, 1]

    cov = np.cov(x, y)

    e1, e2 = np.linalg.eigvals(cov)

    width = np.sqrt(e1) * er
    height = np.sqrt(e2) * er

    return height, width


def ellipse_points(height: float, width: float):
    """
    Return (x,y) points of an ellipse
    for plotting
    """
    angles = np.linspace(0, 2 * np.pi, 1000)

    x = width * np.cos(angles)
    y = height * np.sin(angles)

    points = np.column_stack((x, y))

    return points


def within(height: float, width: float, data: np.ndarray) -> np.ndarray[bool]:
    """
    Given an ellipse and a set of points
    determine which points are inside the ellipse
    """

    x = data[:, 0]
    y = data[:, 1]

    condition = (x / width) ** 2 + (y / height) ** 2

    indices = np.where(condition > 1)

    return indices
