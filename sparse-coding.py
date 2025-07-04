import random
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt 

class SparseCodingModel():
    def __init__(self, n_bases: int, patch_size: int, lambda_: float, eta: float):
        """
        Initializes the sparse coding model.

        Args:
            n_bases (int): Number of basis functions (dictionary elements).
            patch_size (int): Side length of square image patches.
            lambda_ (float): Sparsity regularization parameter. Higher values promote sparsity.
            eta (float): Learning rate for updating basis functions.
        """
        self.n_bases = n_bases
        self.patch_size = patch_size
        self.lambda_ = lambda_
        self.eta = eta
        self.bases = self.init()
        self.activations = None

    def init(self) -> NDArray[np.float64]:
        """
        Initializes basis functions with random values and normalizes each one.

        Returns:
            NDArray[np.float64]: Initialized and normalized basis filters of shape (n_bases, patch_size, patch_size).
        """
        basis = np.random.randn(self.n_bases, self.patch_size, self.patch_size)
        for i in range(self.n_bases):
            basis[i] /= np.linalg.norm(basis[i])
        return basis

    def encode(self, patch: np.ndarray, n_iters: int = 50, step_size: float = 0.1) -> np.ndarray:
        """
        Encodes a single image patch using ISTA (Iterative Shrinkage-Thresholding Algorithm).

        Args:
            patch (np.ndarray): 2D array of shape (patch_size, patch_size).
            n_iters (int): Number of optimization iterations.
            step_size (float): Step size for gradient descent.

        Returns:
            np.ndarray: Sparse activation vector of shape (n_bases,).
        """
        patch_vec = patch.flatten()
        phi_flat = self.bases.reshape(self.n_bases, -1)
        a = np.zeros(self.n_bases)
        phi_T = phi_flat.T

        for _ in range(n_iters):
            recon = phi_T @ a
            grad = phi_flat @ (recon - patch_vec)
            a = a - step_size * grad
            a = self.soft_threshold(a, self.lambda_ * step_size)

        return a

    def soft_threshold(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """
        Applies soft thresholding function element-wise.

        Args:
            x (np.ndarray): Input array.
            lmbda (float): Threshold value.

        Returns:
            np.ndarray: Thresholded array.
        """
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

    def update_bases(self, patch: np.ndarray, a: np.ndarray):
        """
        Updates the basis functions using the reconstruction error and sparse activations.

        Args:
            patch (np.ndarray): Original image patch.
            a (np.ndarray): Sparse activation vector corresponding to the patch.
        """
        I_hat = np.sum(a[:, None, None] * self.bases, axis=0)
        R = patch - I_hat
        for i in range(self.n_bases):
            self.bases[i] += self.eta * a[i] * R
            self.bases[i] /= np.linalg.norm(self.bases[i])

    def train(self, patches: list[np.ndarray], n_epochs: int):
        """
        Trains the sparse coding model on a set of image patches.

        Args:
            patches (list of np.ndarray): List of 2D image patches.
            n_epochs (int): Number of training epochs.
        """
        for _ in range(n_epochs):
            random.shuffle(patches)
            for patch in patches:
                a = self.encode(patch)
                self.update_bases(patch, a)
        self.activations = a


def create_data_b(patch_size: int = 16, n_active: int = 10) -> np.ndarray:
    """
    Generates a synthetic patch with sparse frequency components in the Fourier domain.

    Args:
        patch_size (int): Size of square patch.
        n_active (int): Number of active frequencies.

    Returns:
        np.ndarray: Normalized spatial domain patch.
    """
    fourier_patch = np.zeros((patch_size, patch_size), dtype=complex)
    indices = np.random.choice(patch_size * patch_size, size=n_active, replace=False)
    for idx in indices:
        x = idx // patch_size
        y = idx % patch_size
        amplitude = np.random.exponential(scale=1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        fourier_patch[x, y] = amplitude * np.exp(1j * phase)
    spatial_patch = np.fft.ifft2(fourier_patch).real
    spatial_patch -= spatial_patch.mean()
    spatial_patch /= spatial_patch.std() + 1e-8
    return spatial_patch


def generate_dataset(num_patches: int, patch_size: int = 16, n_active_freqs: int = 3) -> np.ndarray:
    """
    Generates a dataset of synthetic image patches with sparse frequency content.

    Args:
        num_patches (int): Number of patches to generate.
        patch_size (int): Size of each patch.
        n_active_freqs (int): Number of active frequencies per patch.

    Returns:
        np.ndarray: Array of shape (num_patches, patch_size, patch_size).
    """
    return np.array([create_data_b(patch_size, n_active_freqs) for _ in range(num_patches)])


def plot_bases(bases: np.ndarray, patch_size: int, n_cols: int = 10):
    """
    Plots the learned basis functions in a grid layout.

    Args:
        bases (np.ndarray): Array of shape (n_bases, patch_size, patch_size).
        patch_size (int): Size of each basis patch.
        n_cols (int): Number of columns in the plot grid.
    """
    n_bases = bases.shape[0]
    n_rows = (n_bases + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows * patch_size / 10))
    for i, ax in enumerate(axes.flat):
        if i < n_bases:
            ax.imshow(bases[i], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    dataset = generate_dataset(100)
    
    # note that a high value of lambda puts an emphasis on the sparsity of the activations
    model = SparseCodingModel(n_bases=50, patch_size=16, lambda_=1, eta=0.01)
    model.train(dataset, n_epochs=10)
    plot_bases(model.bases, model.patch_size)
