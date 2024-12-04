import numpy as np
from ProcessOptimizer.space import Space

from .suggestor import IncompatibleNumberAsked


class LHSSuggestor():
    def __init__(
            self,
            space: Space,
            rng: np.random.Generator,
            n_points: int = 5
    ):
        self.space = space
        self.rng = rng
        self.n_points = n_points
        self.cache = self.find_lhs_points()

    def find_lhs_points(self):
        # Create a list of evenly distributed points in the range [0, 1] to sample from
        sample_indices = (np.arange(self.n_points) + 0.5) / self.n_points
        samples = []
        for i in range(self.space.n_dims):
            # Sample the points in the ith dimension
            lhs_aranged = self.space.dimensions[i].sample(sample_indices)
            # Shuffle the points in the ith dimension
            samples.append(
                [lhs_aranged[p] for p in self.rng.permutation(self.n_points)]
            )
        # Now we have a list of lists where each inner list is all the points in one
        # dimension in random order. We need to transpose this so that we get a list of
        # points in the space, where each point is a list of values from each
        # dimension.
        transposed_samples = []
        for i in range(self.n_points):
            row = [samples[j][i] for j in range(self.space.n_dims)]
            transposed_samples.append(row)
        return transposed_samples

    def suggest(self, Xi: list[list], Yi: list, n_asked: int = 1) -> list[list]:
        if n_asked + len(Xi) > self.n_points:
            raise IncompatibleNumberAsked(
                "The number of points requested is greater than the number of points "
                "in the LHS cache."
            )
        return self.cache[len(Xi):len(Xi) + n_asked]
