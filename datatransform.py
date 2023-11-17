import numpy as np


class RCTransform:
    """
    Class Object to center data and rotate along principle components
    """

    def __init__(self) -> None:
        pass

    def _center_data(self) -> None:
        self.mu = np.mean(self.data, axis=0)
        self.c_data = self.data - self.mu

    def _rotate_data(self) -> None:
        """
        Rotate centered data
        """
        # calculate the covariance matrix
        covariance_matrix = np.cov(self.c_data.T)

        # find the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # sort the eigenvectors by eigenvalue
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # take the first two eigenvectors as principal components
        self.principal_components = sorted_eigenvectors[:, :2]

        # rotate the data
        self.rc_data = np.dot(self.c_data, self.principal_components)

    def _params(self) -> list:
        """
        Return a list of the parameters
        used to transform data
        """
        return [self.mu, self.principal_components]

    def param_fit(self, param_list: list[float, np.ndarray]) -> None:
        """
        Fit transformer given a parameter list
        """
        self.mu = param_list[0]
        self.principal_components = param_list[1]

    def fit(self, data: np.ndarray):
        """
        Fit transformer given a dataset
        """
        self.data = data

        self._center_data()
        self._rotate_data()

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data by centering and rotation
        """
        # Center the data
        t_data = data - self.mu

        # Rotate the data
        t_data = np.dot(t_data, self.principal_components)

        return t_data

    def inv_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data by rotating and translating
        """
        # Rotate data
        it_data = np.dot(data, self.principal_components.T)

        # Reposition data
        it_data = it_data + self.mu

        return it_data
