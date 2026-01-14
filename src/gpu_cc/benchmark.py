"""Benchmark connected component labeling libraries."""

import torch
import numpy as np
from pathlib import Path

from skimage import measure

import cc3d
from .gpu_wrapper import gpu_connected_components

from timeit import timeit

import matplotlib.pyplot as plt

# Benchmark libraries (native 3D only)
import SimpleITK as sitk
import mahotas
import diplib as dip
from cupyx.scipy import ndimage as cupy_ndimage
import cupy as cp
from cucim.skimage import measure as cucim_measure

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Output directory for plots
PLOT_DIR = Path(__file__).parent.parent.parent / "plots"


def create_matrix(size: int) -> np.ndarray:
    """Create a test matrix with true 3D volumetric connected components."""
    matrix = np.zeros((50, size, size))

    # 3D Component 1: Large cube spanning multiple slices
    matrix[5:15, size//4:size//2, size//4:size//2] = 1

    # 3D Component 2: Another cube in different location
    matrix[20:30, 3*size//4-5:7*size//8+5, size//4:size//2] = 1

    # 3D Component 3: Thin vertical column (tall in Z direction)
    matrix[2:40, size//8:size//8+3, size//8:size//8+3] = 1

    # 3D Component 4: L-shaped volumetric region
    matrix[35:45, 5*size//8:3*size//4, 5*size//8:3*size//4] = 1

    # Add a small 3D component for variation if size allows
    if size > 64:
        matrix[10:18, 7*size//8-4:7*size//8, 7*size//8-4:7*size//8] = 1

    return matrix


# Benchmark functions

def torch_cc(matrix):
    """PyTorch GPU with sync."""
    result, _ = gpu_connected_components(matrix)
    torch.cuda.synchronize()
    return result


def torch_cc_nosync(matrix):
    """PyTorch GPU without sync."""
    result, _ = gpu_connected_components(matrix)
    return result


def cc3d_cc(matrix):
    """cc3d CPU."""
    return cc3d.connected_components(matrix)


def skimage_cc(matrix):
    """scikit-image CPU."""
    return measure.label(matrix)


def numpy_cc(matrix):
    """Numpy to GPU."""
    matrix_torch = torch.as_tensor(matrix, dtype=torch.float32).to(DEVICE)
    result, _ = gpu_connected_components(matrix_torch)
    return result


def simpleitk_cc(matrix):
    """SimpleITK CPU (native 3D)."""
    image = sitk.GetImageFromArray(matrix.astype(np.uint8))
    labeled = sitk.ConnectedComponent(image)
    return sitk.GetArrayFromImage(labeled)


def mahotas_cc(matrix):
    """Mahotas CPU (native 3D)."""
    return mahotas.label(matrix.astype(np.uint8))[0]


def diplib_cc(matrix):
    """DIPlib CPU (native 3D)."""
    img = dip.Image(matrix > 0)  # Convert to binary
    labeled = dip.Label(img)
    return np.array(labeled)


def cupy_ndimage_cc(matrix):
    """CuPy ndimage GPU (native 3D)."""
    matrix_cp = cp.asarray(matrix)
    labeled, num_features = cupy_ndimage.label(matrix_cp)
    cp.cuda.Stream.null.synchronize()
    return labeled


def cucim_cc(matrix):
    """cuCIM raw GPU (native 3D) - no wrapper."""
    matrix_cp = cp.asarray(matrix)
    labeled, num_features = cucim_measure.label(matrix_cp, return_num=True)
    cp.cuda.Stream.null.synchronize()
    return labeled


def run_benchmark(sizes=None, runs=10):
    """Run the benchmark suite."""
    if sizes is None:
        sizes = [32, 64, 128, 256, 512, 1024, 1200]

    results = {
        'sizes': sizes,
        'torch': [],
        'torch_nosync': [],
        'numpy': [],
        'cc3d': [],
        'skimage': [],
        'simpleitk': [],
        'mahotas': [],
        'cupy_ndimage': [],
        'cucim': [],
        'diplib': [],
    }

    for size in sizes:
        print(f"Benchmarking size: 50x{size}x{size} (3D)")
        matrix = create_matrix(size)
        matrix_torch = torch.as_tensor(matrix, dtype=torch.float32).to(DEVICE)

        # Helper function to safely run benchmarks
        def safe_benchmark(func, *args, **kwargs):
            try:
                return timeit(lambda: func(*args, **kwargs), number=runs) / runs
            except Exception as e:
                print(f"  ⚠️  {func.__name__} failed: {type(e).__name__}: {e}")
                return float('nan')

        # GPU benchmarks
        results['torch'].append(safe_benchmark(torch_cc, matrix_torch))
        results['torch_nosync'].append(safe_benchmark(torch_cc_nosync, matrix_torch))
        results['numpy'].append(safe_benchmark(numpy_cc, matrix))
        results['cupy_ndimage'].append(safe_benchmark(cupy_ndimage_cc, matrix))
        results['cucim'].append(safe_benchmark(cucim_cc, matrix))

        # CPU native 3D benchmarks
        results['cc3d'].append(safe_benchmark(cc3d_cc, matrix))
        results['skimage'].append(safe_benchmark(skimage_cc, matrix))
        results['simpleitk'].append(safe_benchmark(simpleitk_cc, matrix))
        results['mahotas'].append(safe_benchmark(mahotas_cc, matrix))
        results['diplib'].append(safe_benchmark(diplib_cc, matrix))

    return results


def plot_results(results, output_dir=None):
    """Generate benchmark plots."""
    if output_dir is None:
        output_dir = PLOT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = results['sizes']

    # Combined line graph - Native 3D implementations only
    plt.figure(figsize=(14, 8))

    # GPU implementations
    plt.plot(sizes, results['torch'], marker='s', label='PyTorch (GPU, 3D)')
    plt.plot(sizes, results['torch_nosync'], marker='s', linestyle='--', label='PyTorch no sync (GPU, 3D)')
    plt.plot(sizes, results['numpy'], marker='x', label='Numpy→GPU (3D)')
    plt.plot(sizes, results['cupy_ndimage'], marker='d', label='CuPy ndimage (GPU, 3D)')
    plt.plot(sizes, results['cucim'], marker='>', label='cuCIM (GPU, 3D)')

    # CPU implementations
    plt.plot(sizes, results['cc3d'], marker='^', label='cc3d (CPU, 3D)')
    plt.plot(sizes, results['skimage'], marker='o', label='skimage (CPU, 3D)')
    plt.plot(sizes, results['simpleitk'], marker='v', label='SimpleITK (CPU, 3D)')
    plt.plot(sizes, results['mahotas'], marker='p', label='Mahotas (CPU, 3D)')
    plt.plot(sizes, results['diplib'], marker='P', label='DIPlib (CPU, 3D)')

    plt.title('Connected Components: Execution Time vs Matrix Size (50 x N x N)')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='upper left', fontsize=8)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'connected_components_comparison_line_graph.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Separate subplot grid - Native 3D implementations only
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # GPU methods (top row)
    axes[0, 0].plot(sizes, results['torch'], marker='s', color='orange', label='PyTorch')
    axes[0, 0].plot(sizes, results['torch_nosync'], marker='s', linestyle='--', color='darkorange', label='PyTorch no sync')
    axes[0, 0].set_title('PyTorch (GPU, 3D)')
    axes[0, 0].set_xlabel('Matrix Size')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(sizes, results['cupy_ndimage'], marker='d', color='purple', label='CuPy ndimage')
    axes[0, 1].plot(sizes, results['cucim'], marker='>', color='lime', label='cuCIM')
    axes[0, 1].set_title('CuPy & cuCIM (GPU, 3D)')
    axes[0, 1].set_xlabel('Matrix Size')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(sizes, results['numpy'], marker='x', color='red')
    axes[0, 2].set_title('Numpy→GPU (3D, Conversion Overhead)')
    axes[0, 2].set_xlabel('Matrix Size')
    axes[0, 2].set_ylabel('Time (seconds)')
    axes[0, 2].set_xscale('log', base=2)
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    # CPU methods (bottom row)
    axes[1, 0].plot(sizes, results['cc3d'], marker='^', color='green', label='cc3d')
    axes[1, 0].plot(sizes, results['skimage'], marker='o', color='blue', label='skimage')
    axes[1, 0].set_title('cc3d & skimage (CPU, 3D)')
    axes[1, 0].set_xlabel('Matrix Size')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(sizes, results['simpleitk'], marker='v', color='teal', label='SimpleITK')
    axes[1, 1].plot(sizes, results['mahotas'], marker='p', color='brown', label='Mahotas')
    axes[1, 1].plot(sizes, results['diplib'], marker='P', color='navy', label='DIPlib')
    axes[1, 1].set_title('SimpleITK, Mahotas, DIPlib (CPU, 3D)')
    axes[1, 1].set_xlabel('Matrix Size')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # All CPU libraries comparison
    axes[1, 2].plot(sizes, results['cc3d'], marker='^', color='green', label='cc3d')
    axes[1, 2].plot(sizes, results['skimage'], marker='o', color='blue', label='skimage')
    axes[1, 2].plot(sizes, results['simpleitk'], marker='v', color='teal', label='SimpleITK')
    axes[1, 2].plot(sizes, results['mahotas'], marker='p', color='brown', label='Mahotas')
    axes[1, 2].plot(sizes, results['diplib'], marker='P', color='navy', label='DIPlib')
    axes[1, 2].set_title('All CPU Libraries Comparison (3D)')
    axes[1, 2].set_xlabel('Matrix Size')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].set_xscale('log', base=2)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.suptitle('Connected Components Benchmark: GPU vs CPU Libraries', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'connected_components_comparison_separate_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    """Run benchmark and generate plots."""
    results = run_benchmark()
    plot_results(results)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
