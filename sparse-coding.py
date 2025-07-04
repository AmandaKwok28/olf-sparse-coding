import random
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt 

class SparseCodingModel():
    def __init__(self, n_bases, patch_size, lambda_, eta):
        '''
        Args:
            n_bases (int)   : number of bases you'd like to use (ndict for example), allowed to be overcomplete
            patch_size (int): side length of the square image patches we'll train on
            lambda_         : tradeoff importance between sparsity and reconstruction accuracy. high values prioritize sparsity
            eta             : learning rate for basis functions
        '''
        self.n_bases = n_bases
        self.patch_size = patch_size
        self.lambda_ = lambda_
        self.eta = eta
        self.bases = self.init()      # this is S in GraFT
        self.activations = None       # this is D in GraFT
    
    def init(self) -> NDArray[np.float64]:
        '''
        initializes our basis functions to be random normalized filters
        '''
        basis = np.random.randn(self.n_bases, self.patch_size, self.patch_size)
        for i in range(self.n_bases):
            basis[i] /= np.linalg.norm(basis[i])
        return basis
    
    def encode(
            self, 
            patch: np.ndarray, 
            n_iters: int = 50,
            step_size: float = 0.1
        ) -> np.ndarray:
        '''
        This function updates the activations of each filter (ith basis function) using ISTA
        
        Args:
            patch: (patch_size, patch_size) numpy array
            n_iters: number of iterations for optimization
            step_size: gradient descent step size

        Returns:
            a: (n_bases,) numpy array of sparse activations
        '''
        patch_vec = patch.flatten()                    # shape: (patch_size * patch_size,)
        phi_flat = self.bases.reshape(self.n_bases, -1)  # shape: (n_bases, patch_size^2)
        a = np.zeros(self.n_bases)                     # initialize a to zero

        # precompute for speed
        phi_T = phi_flat.T                             # shape: (patch_size^2, n_bases)

        for _ in range(n_iters):
            recon = phi_T @ a                          # shape: (patch_size^2,), reconstructed image of activations * bases
            grad = phi_flat @ (recon - patch_vec)      # shape: (n_bases,). Note gradient = bases.T * (reconstruction error) 
            a = a - step_size * grad
            a = self.soft_threshold(a, self.lambda_ * step_size)  # apply sparsity

        return a
    
    def soft_threshold(self, x, lmbda):
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

    
    def update_bases(self, patch: np.ndarray, a):
        '''
        This function updates the basis functions for the ith activation per pixel
        
        Args:
            patch: (patch_size, patch_size) numpy array
        '''
        # use a[:, None, None] to broadcast a from (n_bases,) to (n_basis, 1, 1)
        I_hat = np.sum(a[:, None, None] * self.bases, axis=0)    # reconstructed image
        R = patch - I_hat                                        # the residual = original image - reconstruction
        for i in range(self.n_bases):                           
            self.bases[i] += self.eta * a[i] * R
            self.bases[i] /= np.linalg.norm(self.bases[i])       # normalize the filter
            
    
    def train(self, patches, n_epochs):
        for _ in range(n_epochs):
            random.shuffle(patches)              # shuffle for stability
            for patch in patches:
                a = self.encode(patch)
                self.update_bases(patch, a)
                
        self.activations = a
                
                
                
def create_data_b(patch_size=16, n_active=10):
    """
    Generate a synthetic image patch with sparse frequency components.

    This function creates a patch in the Fourier domain by activating a small number 
    (`n_active`) of frequency components with random amplitude and phase, then converts 
    it back to the spatial domain using inverse FFT. The resulting spatial patch is 
    normalized to have zero mean and unit variance.

    Args:
        patch_size (int, optional): The height and width of the square image patch. 
            Defaults to 16.
        n_active (int, optional): The number of active frequency components in the 
            Fourier domain. Controls sparsity. Defaults to 10.

    Returns:
        np.ndarray: A (patch_size x patch_size) numpy array representing the spatial 
            domain image patch with sparse frequency structure.
    """
    fourier_patch = np.zeros((patch_size, patch_size), dtype=complex)
    indices = np.random.choice(patch_size * patch_size, size=n_active, replace=False)
    for idx in indices:
        x = idx // patch_size
        y = idx % patch_size
        amplitude = np.random.exponential(scale=1.0)
        phase = np.random.uniform(0, 2*np.pi)
        fourier_patch[x, y] = amplitude * np.exp(1j * phase)
    spatial_patch = np.fft.ifft2(fourier_patch).real
    spatial_patch -= spatial_patch.mean()
    spatial_patch /= spatial_patch.std() + 1e-8
    
    return spatial_patch
    
def generate_dataset(num_patches, patch_size=16, n_active_freqs=3):
    dataset = [create_data_b(patch_size, n_active_freqs) for _ in range(num_patches)]
    return np.array(dataset)

def plot_bases(bases, patch_size, n_cols=10):
    n_bases = bases.shape[0]
    n_rows = (n_bases + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows * patch_size / 10))
    for i, ax in enumerate(axes.flat):
        if i < n_bases:
            ax.imshow(bases[i], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()
    plt.show()



                
if __name__ == "__main__":
    
    # example patch
    # patch = create_data_b()
    # plt.imshow(patch, cmap='gray')
    # plt.show()
    
    # recall, a high lambda means we care more about sparsity
    dataset = generate_dataset(1000)
    model = SparseCodingModel(n_bases=100, patch_size=16, lambda_=1, eta=0.01)  # use overcomplete and weight sparsity higher
    model.train(dataset, 20)
    
    # plot the basis functions
    # plot_bases(dataset, patch_size=16)
    plot_bases(model.bases, model.patch_size)
    