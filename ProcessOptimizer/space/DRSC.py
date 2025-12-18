from typing import List, Tuple, Callable, Optional

import numpy as np

from scipy.optimize import linprog


class DRSCGenerator:
    """
    Dirichlet-Rescale-Constraints (DRSC) Algorithm
    Generates n-dimensional vectors with fixed sum satisfying linear and nonlinear constraints.
    
    This implementation is based on the paper "Generating Random Vectors 
    satisfying Linear and Nonlinear Constraints" by Rick S. H. Willemsen, 
    Wilco van den Heuvel and Michel van de Velden. It can be found here:
    https://arxiv.org/abs/2501.16936
    """
    
    def __init__(
        self, 
        n: int,
        bounds: np.ndarray,
        linear_constraints: Optional[List[Tuple[np.ndarray, float]]] = None,
        nonlinear_constraints: Optional[List[Callable]] = None,
        epsilon_ineq: float = 1e-3,
        epsilon_eq: float = 1e-2,
        max_restarts: int = 10000
    ):
        """
        Initialize the DRSC generator.
        
        Parameters
        ----------
        * `n` [int]: 
            Dimension of vectors to be generated
            
        * `bounds` [np.ndarray]: 
            Array of shape (n, 2) with [lower, upper] bounds for each dimension
            (normalized to sum = 1 simplex)
        
        * `linear_constraints`: 
            List of (a, b) tuples representing a^T x <= b
            
        * `nonlinear_constraints`: 
            List of functions f(x) that should be <= 0
            
        * `epsilon_ineq`: 
            Feasibility tolerance for inequality constraints
            
        * `epsilon_eq`:
            Feasibility tolerance for equality constraints
        
        * `max_restarts`:
            Maximum number of restarts before giving up
        """
        self.n = n
        self.original_bounds = np.array(bounds)
        self.linear_constraints = linear_constraints or []
        self.nonlinear_constraints = nonlinear_constraints or []
        self.epsilon_ineq = epsilon_ineq
        self.epsilon_eq = epsilon_eq
        self.max_restarts = max_restarts
        
        # Compute optimal dimension ordering based on constrained dimension width
        self._sort_order, self._inverse_order = self._compute_optimal_order()
        
        # Reorder bounds and constraints according to sort order
        self._sorted_bounds = self.original_bounds[self._sort_order]
        self._sorted_linear_constraints = self._reorder_linear_constraints()
        self._sorted_nonlinear_constraints = self._reorder_nonlinear_constraints()
        
        # Find induced simplices using sorted dimensions (Algorithm 3, line 1)
        self._thetas = self._find_induced_simplices()

    
    def _compute_optimal_order(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal dimension ordering based on width.
        
        Dimensions with smaller widths (less flexible) are processed first.
        This maximizes sampling efficiency by leaving flexible dimensions
        to absorb variation from random sampling.
        
        Returns
        -------
            `sort_order` [np.ndarray]: 
                Indices to sort dimensions by width (ascending)
            `inverse_order`[np.ndarray]: 
                Indices to restore original dimension order
        """
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        sort_order = np.argsort(widths)
        inverse_order = np.argsort(sort_order)
        
        return sort_order, inverse_order
    
    
    def _reorder_linear_constraints(self) -> List[Tuple[np.ndarray, float]]:
        """Reorder linear constraint coefficients according to sort order."""
        reordered = []
        for a, b in self.linear_constraints:
            a_reordered = a[self._sort_order]
            reordered.append((a_reordered, b))
        return reordered
    
    
    def _reorder_nonlinear_constraints(self) -> List[Callable]:
        """Wrap nonlinear constraints to work with sorted dimensions."""
        wrapped = []
        for f in self.nonlinear_constraints:
            def wrapped_f(x_sorted, func=f):
                # Convert from sorted order to original order before evaluating
                x_original = x_sorted[self._inverse_order]
                return func(x_original)
            wrapped.append(wrapped_f)
        return wrapped
    
    
    def _find_induced_simplices(self) -> np.ndarray:
        """
        Find induced simplices Si for each dimension using LP formulation (1)-(7).
        Uses sorted dimension order for optimal efficiency.
        
        For each dimension i, find theta_i such that the induced simplex
        S_i = {x in S : x_i >= theta_i} covers part of the infeasible region
        while not overlapping with previous induced simplices.
        
        The key insight from the paper is that induced simplices should cover
        the INFEASIBLE region, so that when a point lands there, we can 
        transform it back to the standard simplex.
        
        Returns
        -------
            `thetas` [np.ndarray]:
                Array of theta values for each dimension
        """
        thetas = np.zeros(self.n)
        
        for i in range(self.n):
            theta_i = self._solve_lp_for_dimension(i, thetas)
            thetas[i] = theta_i
        
        return thetas
    
    
    def _solve_lp_for_dimension(
            self,
            dim: int,
            previous_thetas: np.ndarray
        ) -> float:
        """
        Solve LP formulation (1)-(7) for dimension dim (in sorted order).
        
        The goal is to find the largest theta_i such that the induced simplex
        S_i = {x : x_i >= theta_i, sum(x) = 1, x >= 0} lies within the 
        infeasible region AND doesn't overlap with previous induced simplices.
        
        According to the paper, we maximize x_i subject to:
        - sum(x) = 1 (on the simplex)
        - 0 <= x <= 1 (standard simplex bounds)
        - All linear constraints in J_lin are satisfied
        - x_k < theta_k for all k < i (non-overlapping with previous simplices)
        
        Parameters
        ----------
            `dim` [int]:
                Current dimension index (in sorted order)
            `previous_thetas` [np.ndarray]:
                Array of theta values from previous dimensions
            
        Returns
        -------
            `theta_i` [float]:
                theta_i value for the induced simplex
        """
        # Objective: maximize x_i (we'll negate for minimization)
        c = np.zeros(self.n)
        c[dim] = -1  # Negative because linprog minimizes
        
        # Equality constraint: sum(x) = 1
        A_eq = np.ones((1, self.n))
        b_eq = np.array([1.0])
        
        # Inequality constraints
        A_ub = []
        b_ub = []
        
        # Box constraints: l <= x <= u (from the sorted bounds)
        for i in range(self.n):
            # Lower bound: x[i] >= lower  →  -x[i] <= -lower
            a_lower = np.zeros(self.n)
            a_lower[i] = -1.0
            A_ub.append(a_lower)
            b_ub.append(-self._sorted_bounds[i, 0] + self.epsilon_ineq)
            
            # Upper bound: x[i] <= upper
            a_upper = np.zeros(self.n)
            a_upper[i] = 1.0
            A_ub.append(a_upper)
            b_ub.append(self._sorted_bounds[i, 1] + self.epsilon_ineq)
        
        # Additional linear constraints (already reordered)
        for a, b in self._sorted_linear_constraints:
            A_ub.append(a)
            b_ub.append(b + self.epsilon_ineq)
        
        # Non-overlapping constraints: x_k < theta_k for k < dim
        # This ensures the new induced simplex doesn't overlap with previous ones
        # We enforce x[k] <= theta_k - epsilon (strictly less than)
        for k in range(dim):
            if previous_thetas[k] > 0:
                constraint = np.zeros(self.n)
                constraint[k] = 1.0
                A_ub.append(constraint)
                # x[k] <= theta_k - small_epsilon to ensure non-overlap
                b_ub.append(previous_thetas[k] - 1e-9)
        
        # Convert to arrays
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(self.n)]
        
        # Solve LP
        result = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            A_eq=A_eq, 
            b_eq=b_eq, 
            bounds=bounds, 
            method='highs',
        )
        
        if result.success:
            return result.x[dim]
        else:
            return 0.0
    
    
    def _sample_flat_dirichlet(self) -> np.ndarray:
        """
        Sample from flat Dirichlet distribution (uniform on standard simplex).
        """
        y = np.random.exponential(1.0, self.n)
        return y / np.sum(y)
    
    
    def _check_all_constraints(self, x: np.ndarray) -> bool:
        """
        Check if x satisfies ALL constraints.
        
        For the DRSC algorithm to work correctly, we must check:
        1. Box bounds (essential - feasible region is subset of standard simplex)
        2. Additional linear constraints (if any)
        3. Nonlinear constraints (if any)
        
        Parameters
        ----------
        * `x` [np.ndarray]:
            Point to check (in sorted dimension order)
            
        Returns
        -------
        * bool
            True if all constraints are satisfied
        """
        # Check box bounds - ESSENTIAL
        # The standard simplex is [0,1]^n with sum=1, but our feasible region
        # is defined by tighter bounds
        for i in range(self.n):
            if x[i] < self._sorted_bounds[i, 0] - self.epsilon_ineq:
                return False
            if x[i] > self._sorted_bounds[i, 1] + self.epsilon_ineq:
                return False
        
        # Check additional linear constraints (if any)
        for a, b in self._sorted_linear_constraints:
            if np.dot(a, x) > b + self.epsilon_ineq:
                return False
        
        # Check nonlinear constraints (if any)
        for f in self._sorted_nonlinear_constraints:
            if f(x) > self.epsilon_ineq:
                return False
        
        return True
    
    
    def _is_in_induced_simplex(self, x: np.ndarray) -> Optional[int]:
        """
        Check if x is in any induced simplex.
        
        Returns
        -------
            * Index of the induced simplex, or None if not in any
        """
        for idx, (theta, vertices, dim) in enumerate(self.induced_simplices):
            if x[dim] >= theta:
                return idx
        return None
    
    
    def _find_containing_induced_simplex(self, x: np.ndarray) -> Optional[int]:
        """
        Find which induced simplex contains point x.
        
        An induced simplex S_i is defined as {x in S : x_i >= theta_i}.
        Due to the non-overlapping construction, a point can be in at most
        one induced simplex.
        
        Returns
        -------
            * Index of the induced simplex containing x, or None if not in any
        """
        for i in range(self.n):
            if self._thetas[i] > 0 and x[i] >= self._thetas[i]:
                return i
        return None
    
    
    def _affine_transform_to_standard_simplex(self, x: np.ndarray, dim: int) -> np.ndarray:
        """
        Apply affine transformation from induced simplex S_dim to standard simplex.
        
        Parameters
        ----------
            * `x` [np.ndarray]: 
                Point in induced simplex (sorted order)
            * `dim` [int]: 
                Index of the induced simplex dimension
            
        Returns
        -------
            * `x_new` [np.ndarray]:
                Transformed point in standard simplex (sorted order)
        """
        theta = self._thetas[dim]
        
        # Create the transformed point
        x_new = x.copy()
        x_new[dim] = x[dim] - theta
        
        # Rescale to sum to 1
        # The sum is now 1 - theta, so we divide by (1 - theta)
        scale = 1.0 - theta
        if scale > 1e-10:
            x_new = x_new / scale
        
        return x_new
    
    
    def generate(self) -> np.ndarray:
        """
        Generate a single vector using the DRSC algorithm (Algorithm 3).
        
        Returns
        -------
            * `x` [np.ndarray]:
                n-dimensional vector with sum=1 satisfying all constraints. 
                Vector is returned in ORIGINAL dimension order.
        """
        restarts = 0
        
        while restarts < self.max_restarts:
            # Line 2: Sample from flat Dirichlet (in sorted dimension space)
            x = self._sample_flat_dirichlet()

            while True:
                # Line 3: Check if all constraints satisfied
                if self._check_all_constraints(x):
                    return x[self._inverse_order]
                
                # Line 5: Check if x is inside an induced simplex
                simplex_idx = self._find_containing_induced_simplex(x)
                
                if simplex_idx is not None:
                    # Line 6: Apply affine transformation
                    x = self._affine_transform_to_standard_simplex(x, simplex_idx)
                else:
                    # Line 8: Restart - violated constraint but not in any induced simplex
                    restarts += 1
                    break
        
        raise RuntimeError(
            f"Failed to generate valid vector after {self.max_restarts} restarts. "
            f"The constraint region may be very small or empty."
        )
    
    def generate_sample(self, size: int) -> np.ndarray:
        """
        Generate multiple vectors.
        
        Parameters
        ----------
            * `size`[int]: 
                Number of vectors to generate
            
        Returns
        -------
            * Array of shape (size, n) with vectors in ORIGINAL dimension order
        """
        return np.array([self.generate() for _ in range(size)])