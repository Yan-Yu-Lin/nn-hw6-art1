"""
ART-1 (Adaptive Resonance Theory 1) Neural Network Implementation
HW6 - Classification of Industrial Process

This implementation follows the ART-1 algorithm as presented in the course slides.
"""

import numpy as np


class ART1:
    """
    ART-1 Neural Network for binary pattern classification.

    Architecture:
        - F1 layer (Comparison layer): n neurons (input dimension)
        - F2 layer (Recognition layer): m neurons (dynamic, grows as needed)
        - W^(f): Feedforward weights (F1 -> F2), real values
        - W^(b): Feedback weights (F2 -> F1), binary values
    """

    def __init__(self, n_inputs, vigilance=0.5, initial_neurons=1):
        """
        Initialize ART-1 network.

        Args:
            n_inputs: Number of input features (n)
            vigilance: Vigilance parameter rho (0 < rho < 1)
            initial_neurons: Initial number of neurons in F2 layer
        """
        self.n = n_inputs  # Number of neurons in F1 layer
        self.rho = vigilance  # Vigilance parameter
        self.m = initial_neurons  # Number of neurons in F2 layer (dynamic)

        # Initialize weight matrices
        self._initialize_weights()

        # Track which neurons are active (for static structure approach)
        self.active_neurons = 1  # Start with 1 active neuron

        # Store cluster assignments
        self.cluster_assignments = {}

    def _initialize_weights(self):
        """
        Initialize weight matrices according to course formulas.

        W^(f)_ji = 1/(1+n) for feedforward weights (real values)
        W^(b)_ij = 1 for feedback weights (binary values)
        """
        # Feedforward weights: m x n matrix (from F1 to F2)
        # W^(f)_ji = 1/(1+n)
        self.W_f = np.ones((self.m, self.n)) * (1.0 / (1 + self.n))

        # Feedback weights: n x m matrix (from F2 to F1)
        # W^(b)_ij = 1
        self.W_b = np.ones((self.n, self.m))

    def _add_neuron(self):
        """Add a new neuron to F2 layer."""
        self.m += 1

        # Expand feedforward weights
        new_row = np.ones((1, self.n)) * (1.0 / (1 + self.n))
        self.W_f = np.vstack([self.W_f, new_row])

        # Expand feedback weights
        new_col = np.ones((self.n, 1))
        self.W_b = np.hstack([self.W_b, new_col])

        self.active_neurons = self.m

    def _compute_activation(self, x):
        """
        Compute activation levels for all F2 neurons.

        Formula: u_j = Σ W^(f)_ji · x_i for j = 1,...,m

        Args:
            x: Input pattern (binary vector)

        Returns:
            Array of activation levels for each F2 neuron
        """
        # u = W_f @ x (matrix-vector multiplication)
        return np.dot(self.W_f, x)

    def _find_winner(self, activations, disabled):
        """
        Find the winner neuron with highest activation (not disabled).

        Args:
            activations: Activation levels for all neurons
            disabled: Set of disabled neuron indices

        Returns:
            Index of winner neuron, or -1 if all disabled
        """
        # Create masked activations (set disabled to -inf)
        masked = activations.copy()
        for idx in disabled:
            masked[idx] = -np.inf

        # Check if all neurons are disabled
        if np.all(masked == -np.inf):
            return -1

        return np.argmax(masked)

    def _similarity_test(self, x, k):
        """
        Perform similarity test (vigilance test).

        Formula: R = (Σ W^(b)_jk · x_j) / (Σ x_j)

        This computes the ratio of common unitary elements to
        total unitary elements in x.

        Args:
            x: Input pattern
            k: Index of winner neuron

        Returns:
            Similarity ratio R
        """
        # Get feedback weights for winner neuron k
        w_b_k = self.W_b[:, k]

        # Compute AND operation (element-wise multiplication for binary)
        # Number of common unitary elements
        common = np.sum(w_b_k * x)

        # Number of unitary elements in x
        norm_x = np.sum(x)

        if norm_x == 0:
            return 0.0

        return common / norm_x

    def _update_weights(self, x, k):
        """
        Update weight vectors for winner neuron k.

        Formulas:
            W^(b)_jk(t+1) = W^(b)_jk(t) · x_j  (AND operation)
            W^(f)_kj(t+1) = W^(b)_jk(t) · x_j / (0.5 + Σ W^(b)_ik(t) · x_i)

        Args:
            x: Input pattern
            k: Index of winner neuron
        """
        # Update feedback weights (AND operation)
        # W^(b)[:,k] = W^(b)[:,k] AND x
        self.W_b[:, k] = self.W_b[:, k] * x

        # Compute denominator for feedforward update
        # 0.5 + Σ W^(b)_ik · x_i
        common = np.sum(self.W_b[:, k] * x)
        denominator = 0.5 + common

        # Update feedforward weights
        # W^(f)_kj = W^(b)_jk · x_j / denominator
        self.W_f[k, :] = (self.W_b[:, k] * x) / denominator

    def train_single(self, x, idx):
        """
        Train network on a single input pattern.

        Implements the 6 phases of ART-1:
        1. Recognition: Find winner neuron
        2. Comparison: Similarity test
        3. Search: If failed, search for another
        4. Update: Update weights

        Args:
            x: Input pattern (binary vector)
            idx: Index of the pattern (for display)

        Returns:
            Tuple of (initial_winner, final_winner, action)
            action is 'Update' or 'Add'
        """
        x = np.array(x, dtype=float)
        disabled = set()
        initial_winner = None

        while True:
            # Phase 2: Recognition - Compute activations and find winner
            activations = self._compute_activation(x)
            k = self._find_winner(activations, disabled)

            if initial_winner is None:
                initial_winner = k + 1  # 1-indexed for display

            # Phase 4: Search - Check if we need to add new neuron
            if k == -1:
                # All neurons disabled, add new neuron
                self._add_neuron()
                k = self.m - 1  # Index of new neuron

                # Update weights for new neuron
                self._update_weights(x, k)
                self.cluster_assignments[idx] = k + 1  # 1-indexed

                return initial_winner, k + 1, 'Add'

            # Phase 3: Comparison - Similarity test
            R = self._similarity_test(x, k)

            if R > self.rho:
                # Passed vigilance test
                # Phase 5: Update weights
                self._update_weights(x, k)
                self.cluster_assignments[idx] = k + 1  # 1-indexed

                return initial_winner, k + 1, 'Update'
            else:
                # Failed vigilance test, disable this neuron and search again
                disabled.add(k)

    def train(self, patterns):
        """
        Train the ART-1 network on a set of patterns.

        Args:
            patterns: List of input patterns

        Returns:
            List of training results for each pattern
        """
        results = []

        for idx, pattern in enumerate(patterns):
            k_init, k_final, action = self.train_single(pattern, idx + 1)
            results.append({
                'idx': idx + 1,
                'k': k_init,
                'k_final': k_final,
                'action': action
            })

        return results

    def get_clusters(self):
        """
        Get the final cluster groupings.

        Returns:
            Dictionary mapping cluster ID to list of pattern indices
        """
        clusters = {}
        for pattern_idx, cluster_id in self.cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pattern_idx)

        return clusters

    def predict(self, x):
        """
        Classify a single input pattern.

        Args:
            x: Input pattern

        Returns:
            Cluster ID (1-indexed)
        """
        x = np.array(x, dtype=float)
        activations = self._compute_activation(x)
        k = np.argmax(activations)
        return k + 1


def print_results(results, rho, n_clusters):
    """Print results in the format matching course slides."""
    print(f"rho = {rho:.2f}")
    print()
    print("The results of output vector:")

    for r in results:
        # Create output vector representation
        output_vec = [0] * n_clusters
        if r['k_final'] <= n_clusters:
            output_vec[r['k_final'] - 1] = 1

        output_str = str(output_vec)
        print(f"idx = {r['idx']:3d}, k = {r['k']}, k_final = {r['k_final']}, "
              f"OutputVector = {output_str} ({r['action']})")

    print()


def print_groups(cluster_dict):
    """Print cluster groupings."""
    print("Group:")
    for cluster_id in sorted(cluster_dict.keys()):
        members = cluster_dict[cluster_id]
        members_str = ' '.join(map(str, sorted(members)))
        print(f"Class-{cluster_id}: [ {members_str} ]")
    print()


def run_experiment(data, rho, initial_neurons=1):
    """
    Run ART-1 experiment with given parameters.

    Args:
        data: Training data
        rho: Vigilance parameter
        initial_neurons: Initial number of F2 neurons
    """
    print("=" * 60)
    print(f"Experiment: rho = {rho}, initial F2 neurons = {initial_neurons}")
    print("=" * 60)
    print()

    # Create and train network
    n_inputs = len(data[0])
    art1 = ART1(n_inputs=n_inputs, vigilance=rho, initial_neurons=initial_neurons)

    # Train on all patterns
    results = art1.train(data)

    # Get final clusters
    clusters = art1.get_clusters()
    n_clusters = len(clusters)

    # Print results
    print_results(results, rho, n_clusters)
    print_groups(clusters)

    return art1, results, clusters


def main():
    """Main function to run all experiments."""

    # Training data: 10 situations with 16 binary status variables
    # From the assignment PDF
    data = [
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situation 1
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  # Situation 2
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situation 3
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],  # Situation 4
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situation 5
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situation 6
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],  # Situation 7
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situation 8
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # Situation 9
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situation 10
    ]

    print("=" * 60)
    print("ART-1 Neural Network - Industrial Process Classification")
    print("=" * 60)
    print()
    print(f"Input dimension (n): {len(data[0])}")
    print(f"Number of patterns: {len(data)}")
    print()

    # Experiment 1: rho = 0.4 (lower vigilance, fewer clusters)
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Low Vigilance (rho = 0.4)")
    print("#" * 60)
    run_experiment(data, rho=0.4, initial_neurons=1)

    # Experiment 2: rho = 0.7 (higher vigilance, more clusters)
    print("\n" + "#" * 60)
    print("# EXPERIMENT 2: High Vigilance (rho = 0.7)")
    print("#" * 60)
    run_experiment(data, rho=0.7, initial_neurons=1)

    # Additional analysis: Effect of different rho values
    print("\n" + "#" * 60)
    print("# SUMMARY: Effect of Vigilance Parameter")
    print("#" * 60)
    print()
    print("Vigilance (rho) | Number of Clusters")
    print("-" * 40)

    for rho in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        art1 = ART1(n_inputs=len(data[0]), vigilance=rho, initial_neurons=1)
        art1.train(data)
        clusters = art1.get_clusters()
        print(f"     {rho:.1f}         |        {len(clusters)}")

    print()
    print("Note: Higher vigilance (rho) leads to more clusters")
    print("      because the similarity threshold is stricter.")


if __name__ == "__main__":
    main()
