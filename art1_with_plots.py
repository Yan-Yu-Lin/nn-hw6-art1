"""
ART-1 (Adaptive Resonance Theory 1) Neural Network Implementation
HW6 - Classification of Industrial Process

This version includes visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


class ART1:
    """
    ART-1 Neural Network for binary pattern classification.
    """

    def __init__(self, n_inputs, vigilance=0.5, initial_neurons=1):
        self.n = n_inputs
        self.rho = vigilance
        self.m = initial_neurons
        self._initialize_weights()
        self.active_neurons = 1
        self.cluster_assignments = {}

    def _initialize_weights(self):
        self.W_f = np.ones((self.m, self.n)) * (1.0 / (1 + self.n))
        self.W_b = np.ones((self.n, self.m))

    def _add_neuron(self):
        self.m += 1
        new_row = np.ones((1, self.n)) * (1.0 / (1 + self.n))
        self.W_f = np.vstack([self.W_f, new_row])
        new_col = np.ones((self.n, 1))
        self.W_b = np.hstack([self.W_b, new_col])
        self.active_neurons = self.m

    def _compute_activation(self, x):
        return np.dot(self.W_f, x)

    def _find_winner(self, activations, disabled):
        masked = activations.copy()
        for idx in disabled:
            masked[idx] = -np.inf
        if np.all(masked == -np.inf):
            return -1
        return np.argmax(masked)

    def _similarity_test(self, x, k):
        w_b_k = self.W_b[:, k]
        common = np.sum(w_b_k * x)
        norm_x = np.sum(x)
        if norm_x == 0:
            return 0.0
        return common / norm_x

    def _update_weights(self, x, k):
        self.W_b[:, k] = self.W_b[:, k] * x
        common = np.sum(self.W_b[:, k] * x)
        denominator = 0.5 + common
        self.W_f[k, :] = (self.W_b[:, k] * x) / denominator

    def train_single(self, x, idx):
        x = np.array(x, dtype=float)
        disabled = set()
        initial_winner = None

        while True:
            activations = self._compute_activation(x)
            k = self._find_winner(activations, disabled)

            if initial_winner is None:
                initial_winner = k + 1

            if k == -1:
                self._add_neuron()
                k = self.m - 1
                self._update_weights(x, k)
                self.cluster_assignments[idx] = k + 1
                return initial_winner, k + 1, 'Add'

            R = self._similarity_test(x, k)

            if R > self.rho:
                self._update_weights(x, k)
                self.cluster_assignments[idx] = k + 1
                return initial_winner, k + 1, 'Update'
            else:
                disabled.add(k)

    def train(self, patterns):
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
        clusters = {}
        for pattern_idx, cluster_id in self.cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pattern_idx)
        return clusters


def plot_input_data(data):
    """Plot the input data as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data_array = np.array(data)
    im = ax.imshow(data_array, cmap='Blues', aspect='auto')

    ax.set_xticks(range(16))
    ax.set_xticklabels([f'x{i+1}' for i in range(16)])
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Situation {i+1}' for i in range(10)])

    ax.set_xlabel('Status Variables')
    ax.set_ylabel('Situations')
    ax.set_title('Input Data: 10 Situations × 16 Binary Status Variables')

    # Add value annotations
    for i in range(10):
        for j in range(16):
            text = ax.text(j, i, data_array[i, j], ha='center', va='center',
                          color='white' if data_array[i, j] == 1 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig('plot_input_data.png', dpi=150)
    plt.close()
    print("Saved: plot_input_data.png")


def plot_vigilance_effect(data):
    """Plot how vigilance parameter affects number of clusters."""
    rho_values = np.arange(0.1, 1.0, 0.05)
    cluster_counts = []

    for rho in rho_values:
        art1 = ART1(n_inputs=len(data[0]), vigilance=rho, initial_neurons=1)
        art1.train(data)
        clusters = art1.get_clusters()
        cluster_counts.append(len(clusters))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rho_values, cluster_counts, 'b-o', linewidth=2, markersize=8)

    ax.set_xlabel('Vigilance Parameter (ρ)', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Effect of Vigilance Parameter on Cluster Count', fontsize=14)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks(range(1, max(cluster_counts) + 2))

    # Add annotations for key points
    ax.axvline(x=0.4, color='r', linestyle='--', alpha=0.5, label='ρ=0.4 (Experiment 1)')
    ax.axvline(x=0.7, color='g', linestyle='--', alpha=0.5, label='ρ=0.7 (Experiment 2)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plot_vigilance_effect.png', dpi=150)
    plt.close()
    print("Saved: plot_vigilance_effect.png")


def plot_clustering_results(data, rho_values=[0.4, 0.7]):
    """Plot clustering results for different vigilance values."""
    fig, axes = plt.subplots(1, len(rho_values), figsize=(6 * len(rho_values), 6))

    if len(rho_values) == 1:
        axes = [axes]

    for ax, rho in zip(axes, rho_values):
        art1 = ART1(n_inputs=len(data[0]), vigilance=rho, initial_neurons=1)
        art1.train(data)
        clusters = art1.get_clusters()

        # Create cluster assignment array
        assignments = [art1.cluster_assignments.get(i+1, 0) for i in range(len(data))]

        # Color map for clusters
        n_clusters = len(clusters)
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 3)))

        # Bar plot showing cluster assignments
        bars = ax.bar(range(1, 11), [1]*10, color=[colors[a-1] for a in assignments])

        ax.set_xticks(range(1, 11))
        ax.set_xticklabels([f'S{i}' for i in range(1, 11)])
        ax.set_xlabel('Situation')
        ax.set_title(f'ρ = {rho} → {n_clusters} Clusters')
        ax.set_ylim(0, 1.5)
        ax.set_yticks([])

        # Add cluster labels on bars
        for i, (bar, cluster) in enumerate(zip(bars, assignments)):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5, f'C{cluster}',
                   ha='center', va='center', fontsize=12, fontweight='bold')

        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=colors[c-1], label=f'Class {c}: {clusters[c]}')
                         for c in sorted(clusters.keys())]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.suptitle('Clustering Results Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('plot_clustering_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_clustering_results.png")


def print_results(results, rho, n_clusters):
    """Print results in the format matching course slides."""
    print(f"rho = {rho:.2f}")
    print()
    print("The results of output vector:")

    for r in results:
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
    """Run ART-1 experiment with given parameters."""
    print("=" * 60)
    print(f"Experiment: rho = {rho}, initial F2 neurons = {initial_neurons}")
    print("=" * 60)
    print()

    n_inputs = len(data[0])
    art1 = ART1(n_inputs=n_inputs, vigilance=rho, initial_neurons=initial_neurons)
    results = art1.train(data)
    clusters = art1.get_clusters()
    n_clusters = len(clusters)

    print_results(results, rho, n_clusters)
    print_groups(clusters)

    return art1, results, clusters


def main():
    """Main function to run all experiments."""

    # Training data: 10 situations with 16 binary status variables
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

    # Generate plots
    print("\n" + "#" * 60)
    print("# GENERATING PLOTS")
    print("#" * 60 + "\n")

    plot_input_data(data)
    plot_vigilance_effect(data)
    plot_clustering_results(data, [0.4, 0.7])

    # Experiment 1: rho = 0.4
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Low Vigilance (rho = 0.4)")
    print("#" * 60)
    run_experiment(data, rho=0.4, initial_neurons=1)

    # Experiment 2: rho = 0.7
    print("\n" + "#" * 60)
    print("# EXPERIMENT 2: High Vigilance (rho = 0.7)")
    print("#" * 60)
    run_experiment(data, rho=0.7, initial_neurons=1)

    # Summary
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
    print()
    print("Plots saved: plot_input_data.png, plot_vigilance_effect.png, plot_clustering_results.png")


if __name__ == "__main__":
    main()
