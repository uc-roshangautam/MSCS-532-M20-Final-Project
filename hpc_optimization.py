"""
HPC Performance Optimization: Data Locality Improvement
Demonstrating cache-friendly matrix multiplication optimization
Based on empirical study findings on HPC performance bugs
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
import psutil
import os

class MatrixOptimizationDemo:
    """
    Demonstrates data locality optimization techniques for matrix operations
    Based on HPC performance bug patterns identified in empirical research
    """
    
    def __init__(self):
        self.results = {}
        self.timing_data = []
        
    def naive_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive matrix multiplication with poor cache locality
        Demonstrates the performance bug pattern from the empirical study
        """
        n, m, p = A.shape[0], A.shape[1], B.shape[1]
        C = np.zeros((n, p))
        
        # Poor cache locality: accessing B column-wise (non-contiguous memory)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i][j] += A[i][k] * B[k][j]  # Poor spatial locality for B
        
        return C
    
    def optimized_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Cache-optimized matrix multiplication using loop reordering
        Implements the optimization technique from empirical study
        """
        n, m, p = A.shape[0], A.shape[1], B.shape[1]
        C = np.zeros((n, p))
        
        # Better cache locality: reordered loops for spatial locality
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    C[i][j] += A[i][k] * B[k][j]  # Better spatial locality
        
        return C
    
    def blocked_matrix_multiply(self, A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
        """
        Cache-blocked matrix multiplication for optimal data locality
        Advanced optimization technique addressing cache hierarchy
        """
        n, m, p = A.shape[0], A.shape[1], B.shape[1]
        C = np.zeros((n, p))
        
        # Cache blocking to fit data in L1/L2 cache
        for i in range(0, n, block_size):
            for j in range(0, p, block_size):
                for k in range(0, m, block_size):
                    # Process blocks that fit in cache
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, p)
                    k_end = min(k + block_size, m)
                    
                    for ii in range(i, i_end):
                        for kk in range(k, k_end):
                            for jj in range(j, j_end):
                                C[ii][jj] += A[ii][kk] * B[kk][jj]
        
        return C
    
    def measure_performance(self, matrix_size: int, iterations: int = 3) -> dict:
        """
        Measure and compare performance of different optimization levels
        """
        print(f"Testing matrix size: {matrix_size}x{matrix_size}")
        
        # Generate test matrices
        A = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        B = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        
        results = {}
        
        # Test naive implementation
        print("Testing naive implementation...")
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            C_naive = self.naive_matrix_multiply(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        results['naive'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'result_shape': C_naive.shape
        }
        
        # Test optimized implementation
        print("Testing optimized implementation...")
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            C_opt = self.optimized_matrix_multiply(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        results['optimized'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'result_shape': C_opt.shape
        }
        
        # Test blocked implementation
        print("Testing blocked implementation...")
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            C_blocked = self.blocked_matrix_multiply(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        results['blocked'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'result_shape': C_blocked.shape
        }
        
        # Calculate speedup
        naive_time = results['naive']['avg_time']
        results['optimized']['speedup'] = naive_time / results['optimized']['avg_time']
        results['blocked']['speedup'] = naive_time / results['blocked']['avg_time']
        
        # Verify correctness
        np.testing.assert_allclose(C_naive, C_opt, rtol=1e-5)
        np.testing.assert_allclose(C_naive, C_blocked, rtol=1e-5)
        print("✓ All implementations produce identical results")
        
        return results
    
    def run_comprehensive_benchmark(self, sizes: List[int] = None) -> None:
        """
        Run comprehensive benchmarks across different matrix sizes
        """
        if sizes is None:
            sizes = [64, 128, 256, 512]
        
        all_results = {}
        
        for size in sizes:
            print(f"\n{'='*50}")
            print(f"Benchmarking {size}x{size} matrices")
            print(f"{'='*50}")
            
            try:
                results = self.measure_performance(size)
                all_results[size] = results
                
                # Print results
                print(f"\nResults for {size}x{size}:")
                print(f"Naive:     {results['naive']['avg_time']:.4f}s ± {results['naive']['std_time']:.4f}s")
                print(f"Optimized: {results['optimized']['avg_time']:.4f}s ± {results['optimized']['std_time']:.4f}s (speedup: {results['optimized']['speedup']:.2f}x)")
                print(f"Blocked:   {results['blocked']['avg_time']:.4f}s ± {results['blocked']['std_time']:.4f}s (speedup: {results['blocked']['speedup']:.2f}x)")
                
            except MemoryError:
                print(f"Skipping size {size} due to memory constraints")
                continue
        
        self.results = all_results
        self.plot_results()
    
    def plot_results(self) -> None:
        """
        Generate performance visualization plots
        """
        if not self.results:
            print("No results to plot")
            return
        
        sizes = list(self.results.keys())
        naive_times = [self.results[s]['naive']['avg_time'] for s in sizes]
        opt_times = [self.results[s]['optimized']['avg_time'] for s in sizes]
        blocked_times = [self.results[s]['blocked']['avg_time'] for s in sizes]
        
        # Performance comparison plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(sizes, naive_times, 'r-o', label='Naive', linewidth=2, markersize=8)
        plt.plot(sizes, opt_times, 'g-s', label='Optimized', linewidth=2, markersize=8)
        plt.plot(sizes, blocked_times, 'b-^', label='Blocked', linewidth=2, markersize=8)
        plt.xlabel('Matrix Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Performance Comparison: Matrix Multiplication')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Speedup plot
        plt.subplot(1, 2, 2)
        opt_speedups = [self.results[s]['optimized']['speedup'] for s in sizes]
        blocked_speedups = [self.results[s]['blocked']['speedup'] for s in sizes]
        
        plt.bar([s - 10 for s in sizes], opt_speedups, width=20, alpha=0.7, label='Optimized', color='green')
        plt.bar([s + 10 for s in sizes], blocked_speedups, width=20, alpha=0.7, label='Blocked', color='blue')
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup Factor')
        plt.title('Optimization Speedup')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hpc_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_cache_behavior(self, size: int = 256) -> None:
        """
        Analyze cache behavior patterns for educational purposes
        """
        print(f"\n{'='*60}")
        print("CACHE BEHAVIOR ANALYSIS")
        print(f"{'='*60}")
        
        A = np.random.random((size, size)).astype(np.float32)
        B = np.random.random((size, size)).astype(np.float32)
        
        print(f"Matrix size: {size}x{size}")
        print(f"Element size: {A.dtype.itemsize} bytes")
        print(f"Total matrix memory: {A.nbytes / (1024**2):.2f} MB each")
        
        # Estimate cache behavior
        l1_cache_size = 32 * 1024  # Typical L1 cache size (32KB)
        l2_cache_size = 256 * 1024  # Typical L2 cache size (256KB)
        
        elements_per_cache_line = 64 // A.dtype.itemsize  # Assume 64-byte cache lines
        matrix_elements = size * size
        
        print(f"\nCache Analysis:")
        print(f"L1 cache can hold ~{l1_cache_size // A.dtype.itemsize:,} elements")
        print(f"L2 cache can hold ~{l2_cache_size // A.dtype.itemsize:,} elements")
        print(f"Matrix has {matrix_elements:,} elements")
        
        if matrix_elements * A.dtype.itemsize > l2_cache_size:
            print("⚠️  Matrix doesn't fit in L2 cache - optimization is crucial!")
        else:
            print("✓ Matrix fits in L2 cache")
        
        print(f"\nOptimization Impact:")
        print("• Naive: Poor spatial locality, many cache misses")
        print("• Optimized: Better loop ordering, improved spatial locality")
        print("• Blocked: Optimal cache utilization, minimizes cache misses")

def main():
    """
    Main demonstration function
    """
    print("HPC Performance Optimization Demonstration")
    print("Data Locality Improvement for Matrix Operations")
    print("Based on empirical study of HPC performance bugs\n")
    
    demo = MatrixOptimizationDemo()
    
    # Quick demonstration
    print("Quick demonstration with small matrices:")
    demo.run_comprehensive_benchmark([64, 128, 256])
    
    # Cache behavior analysis
    demo.analyze_cache_behavior(256)
    
    print(f"\n{'='*60}")
    print("SUMMARY OF OPTIMIZATION TECHNIQUES DEMONSTRATED")
    print(f"{'='*60}")
    print("1. Loop Reordering: Improved spatial locality")
    print("2. Cache Blocking: Optimal cache hierarchy utilization")
    print("3. Memory Access Patterns: Reduced cache misses")
    print("\nThese optimizations address the performance bug categories")
    print("identified in the HPC empirical study:")
    print("• Inefficient algorithm implementation (39.3%)")
    print("• Inefficient code for micro-architecture (31.2%)")
    print("• Memory/data locality issues (19.4%)")

if __name__ == "__main__":
    main()
