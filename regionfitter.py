import numpy as np

from datatransform import RCTransform
import utils as ut


class EllipseRegion:
    """
    Fit an ellipse region to 2D data
    """

    def __init__(self, alpha=0.05) -> None:
        self.alpha = alpha
        self.transformer = RCTransform()

    def fit(self, data: np.ndarray, alpha=None):
        """
        Fit the ellipse to data
        """
        # Set alpha value
        if alpha is not None:
            self.alpha = alpha

        # Fit data transformer
        self.transformer.fit(data)

        # Transform data
        rc_data = self.transformer.transform(data)

        # Fit ellipse on centered data
        self.height, self.width = ut.ellipse(rc_data, alpha=self.alpha)

    def ellipse_points(self) -> np.ndarray:
        """
        Return a list of (x,y) points
        of the ellipse boundary, used
        for plotting of the region
        """
        # Get points of the centered rotated ellipse
        standard_points = ut.ellipse_points(height=self.height, width=self.width)

        # Transform to the original ellipse
        points = self.transformer.inv_transform(standard_points)

        return points

    def in_ellipse(self, data: np.ndarray) -> np.ndarray[bool]:
        """
        Given an array of data
        return boolean if the point is inside the
        fitted ellipse
        """
        # Transform data
        t_data = self.transformer.transform(data)

        # indices of points that are inside the ellipse
        within_ellipse = ut.within(data=t_data, height=self.height, width=self.width)

        return within_ellipse
