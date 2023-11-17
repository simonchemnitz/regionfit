import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from datatransform import RCTransform
from regionfitter import EllipseRegion


def get_correlated_dataset(n: int, dependency: np.ndarray, mu: list, scale: list):
    """
    Simulate a 2D dataset with correlation
    """
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset, scaled_with_offset[:, 0], scaled_with_offset[:, 1]


if __name__ == "__main__":
    # Data parameters
    dependency_nstd = np.array([[0.8, 0.75], [-0.2, 0.35]])
    N = 200

    # Simulate data
    data, x, y = get_correlated_dataset(
        n=N, dependency=dependency_nstd, mu=[10, 20], scale=[10, 15]
    )

    ###Transformation example

    # Data transformer
    rct = RCTransform()
    rct.fit(data)

    # Transform Data
    t_data = rct.transform(data)
    it_data = rct.inv_transform(t_data)

    # Plot data
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(data[:, 0], data[:, 1], label="original", c="r")
    ax[0].scatter(it_data[:, 0], it_data[:, 1], c="b", alpha=0.5)

    ax[1].scatter(data[:, 0], data[:, 1], label="original", c="r", alpha=0.2)
    ax[1].scatter(t_data[:, 0], t_data[:, 1], label="transformed", c="b", alpha=0.7)

    ax[0].axis("equal")
    ax[1].axis("equal")

    plt.suptitle("Data Transformer")

    ### Ellipse Example
    alpha = 0.1
    er = EllipseRegion(alpha=alpha)
    er.fit(data=data)
    points = er.ellipse_points()

    # Plot data
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(data[:, 0], data[:, 1], label="original", c="r")

    ax[1].scatter(data[:, 0], data[:, 1], label="transformed", c="b", alpha=0.7)
    ax[1].plot(points[:, 0], points[:, 1], label="Ellipse", c="r")

    ax[0].axis("equal")
    ax[1].axis("equal")

    plt.suptitle("Ellipse Fitter")

    plt.show()

    # Verify number of points outside
    outside = er.in_ellipse(data)[0]
    print("outside, all points", len(outside), N)
    print((len(outside) / N) * 100, "%")
    print(alpha * 100)
