#!/usr/bin/env python3
"""
Convert Sparse Voxel Object (SVO) to Dense Voxel Grid

This script reads a sparse voxel binary file produced by the Rust implementation
and converts it to a dense numpy array with correct orientation.

The binary format for sparse voxel grid:
1. N (int32) - grid size
2. VOXEL_SIZE (float32) - voxel size
3. num_voxels (int32) - number of non-zero voxels
4. For each voxel:
   - index (int64) - linear index in the dense grid
   - value (float32) - voxel value

Index encoding: idx = z * N * N + y * N + x
Where:
  - x: fastest varying (columns)
  - y: middle varying (rows)
  - z: slowest varying (depth)
"""

import struct
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def read_sparse_voxel_bin(filename, orientation='xyz'):
    """
    Read sparse voxel binary file and return dense grid.
    
    Args:
        filename: Path to the binary file
        orientation: 'xyz' for idx = x*N*N + y*N + z (dense format)
                    'zyx' for idx = z*N*N + y*N + x (sparse format)
        
    Returns:
        tuple: (dense_grid: numpy.ndarray, voxel_size: float, N: int)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        # Read header
        N_bytes = f.read(4)
        if len(N_bytes) < 4:
            raise ValueError("File too small to contain grid size")
        N = struct.unpack('<i', N_bytes)[0]
        
        voxel_size_bytes = f.read(4)
        if len(voxel_size_bytes) < 4:
            raise ValueError("File too small to contain voxel size")
        voxel_size = struct.unpack('<f', voxel_size_bytes)[0]
        
        # Read number of voxels
        num_voxels_bytes = f.read(4)
        if len(num_voxels_bytes) < 4:
            # This is a dense grid file, not sparse
            f.seek(0)
            return read_dense_voxel_bin(filename)
        
        num_voxels = struct.unpack('<i', num_voxels_bytes)[0]
        
        # Create dense grid (z, y, x order for numpy)
        grid = np.zeros((N, N, N), dtype=np.float32)
        
        # Read each voxel
        for i in range(num_voxels):
            # Read index (8 bytes) and value (4 bytes)
            idx_bytes = f.read(8)
            val_bytes = f.read(4)
            
            if len(idx_bytes) < 8 or len(val_bytes) < 4:
                print(f"Warning: Incomplete data at voxel {i}, stopping")
                break
            
            idx = struct.unpack('<q', idx_bytes)[0]
            val = struct.unpack('<f', val_bytes)[0]
            
            if orientation == 'zyx':
                # Sparse format: idx = z * N * N + y * N + x
                x = idx % N
                y = (idx // N) % N
                z = idx // (N * N)
            else:  # 'xyz' (dense format)
                # Dense format: idx = x * N * N + y * N + z
                z = idx % N
                y = (idx // N) % N
                x = idx // (N * N)
            
            # Store in the dense grid (z, y, x order for numpy)
            if 0 <= x < N and 0 <= y < N and 0 <= z < N:
                grid[z, y, x] = val
            else:
                print(f"Warning: Index {idx} out of bounds for N={N}, x={x}, y={y}, z={z}")
        
        return grid, voxel_size, N

def read_dense_voxel_bin(filename):
    """
    Read dense voxel binary file.
    
    The dense format stores values in the order they were written:
    For the sparse version: grid is stored as (z, y, x) in the file
    For the dense version: it depends on the Rust implementation
    
    Args:
        filename: Path to the binary file
        
    Returns:
        tuple: (dense_grid: numpy.ndarray, voxel_size: float, N: int)
    """
    with open(filename, 'rb') as f:
        # Read header
        N_bytes = f.read(4)
        if len(N_bytes) < 4:
            raise ValueError("File too small to contain grid size")
        N = struct.unpack('<i', N_bytes)[0]
        
        voxel_size_bytes = f.read(4)
        if len(voxel_size_bytes) < 4:
            raise ValueError("File too small to contain voxel size")
        voxel_size = struct.unpack('<f', voxel_size_bytes)[0]
        
        # Read all voxel values
        total_voxels = N * N * N
        grid_data = np.zeros(total_voxels, dtype=np.float32)
        
        # Read all voxel data
        for i in range(total_voxels):
            val_bytes = f.read(4)
            if len(val_bytes) < 4:
                print(f"Warning: Incomplete data at voxel {i}, stopping")
                break
            grid_data[i] = struct.unpack('<f', val_bytes)[0]
        
        # Reshape to (z, y, x) order
        grid = grid_data.reshape((N, N, N))
        return grid, voxel_size, N

def save_dense_grid(grid, voxel_size, N, filename):
    """
    Save dense grid to binary file in the format expected by the Rust code.
    
    The dense format stores values in the same order as the sparse version:
    (z, y, x) order in the file
    
    Args:
        grid: numpy.ndarray of shape (N, N, N)
        voxel_size: float
        N: int
        filename: output file path
    """
    # Ensure grid is in row-major order (z, y, x)
    if grid.shape != (N, N, N):
        raise ValueError(f"Grid shape {grid.shape} doesn't match N={N}")
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('<i', N))
        f.write(struct.pack('<f', voxel_size))
        
        # Write all voxel values in (z, y, x) order
        for val in grid.ravel():
            f.write(struct.pack('<f', val))

def save_grid_xyz_order(grid, voxel_size, N, filename):
    """
    Save grid in xyz order (x varies fastest) for compatibility with some tools.
    """
    # Reorder from (z,y,x) to (x,y,z)
    grid_xyz = np.transpose(grid, (2, 1, 0))
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('<i', N))
        f.write(struct.pack('<f', voxel_size))
        for val in grid_xyz.ravel():
            f.write(struct.pack('<f', val))

def write_vtk(grid, voxel_size, N, filename):
    """
    Write grid to VTK format for visualization in ParaView or similar.
    """
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <ImageData WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}" Origin="0 0 0" Spacing="{voxel_size} {voxel_size} {voxel_size}">\n')
        f.write(f'    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">\n')
        f.write('      <PointData>\n')
        f.write('        <DataArray Name="voxel_values" type="Float32" format="ascii" NumberOfComponents="1">\n')
        
        # Write all values in order (z, y, x)
        for val in grid.ravel():
            f.write(f'{val:.6f} ')
        
        f.write('\n        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </ImageData>\n')
        f.write('</VTKFile>\n')

def write_npy(grid, filename):
    """Save grid as numpy .npy file for easy loading."""
    np.save(filename, grid)

def print_grid_stats(grid):
    """Print statistics about the grid."""
    non_zero = np.count_nonzero(grid)
    total = grid.size
    print(f"Grid statistics:")
    print(f"  Total voxels: {total}")
    print(f"  Non-zero voxels: {non_zero}")
    print(f"  Sparsity: {non_zero/total*100:.2f}%")
    print(f"  Min value: {grid.min():.6f}")
    print(f"  Max value: {grid.max():.6f}")
    print(f"  Mean value: {grid.mean():.6f}")
    print(f"  Std dev: {grid.std():.6f}")
    
    # Find the center of mass (useful for checking orientation)
    if non_zero > 0:
        z, y, x = np.indices(grid.shape)
        com_z = np.sum(z * grid) / np.sum(grid)
        com_y = np.sum(y * grid) / np.sum(grid)
        com_x = np.sum(x * grid) / np.sum(grid)
        print(f"  Center of mass: ({com_x:.2f}, {com_y:.2f}, {com_z:.2f})")
        
        # Check if voxels are near the expected center
        half = grid.shape[0] / 2
        if abs(com_x - half) < half/2 and abs(com_y - half) < half/2 and abs(com_z - half) < half/2:
            print("  ✅ Center of mass is near the grid center (orientation likely correct)")
        else:
            print("  ⚠️  Center of mass is far from grid center (orientation might be wrong)")

def main():
    parser = argparse.ArgumentParser(description='Convert Sparse Voxel Binary to Dense Grid')
    parser.add_argument('input', help='Input binary file (sparse or dense format)')
    parser.add_argument('-o', '--output', help='Output file for dense binary (optional)')
    parser.add_argument('--npy', help='Output .npy file (optional)')
    parser.add_argument('--vtk', help='Output .vti file for visualization (optional)')
    parser.add_argument('--stats', action='store_true', help='Print statistics only')
    parser.add_argument('--orientation', choices=['auto', 'zyx', 'xyz'], default='auto',
                       help='Index orientation: zyx (sparse format) or xyz (dense format)')
    
    args = parser.parse_args()
    
    try:
        # Read the grid
        print(f"Reading voxel grid from: {args.input}")
        
        if args.orientation == 'auto':
            # Try both orientations and use the one that gives better results
            try:
                grid_zyx, voxel_size, N = read_sparse_voxel_bin(args.input, orientation='zyx')
                # Check if center of mass is near center
                non_zero = np.count_nonzero(grid_zyx)
                if non_zero > 0:
                    z, y, x = np.indices(grid_zyx.shape)
                    com_z = np.sum(z * grid_zyx) / np.sum(grid_zyx)
                    com_y = np.sum(y * grid_zyx) / np.sum(grid_zyx)
                    com_x = np.sum(x * grid_zyx) / np.sum(grid_zyx)
                    half = grid_zyx.shape[0] / 2
                    if (abs(com_x - half) < half/2 and abs(com_y - half) < half/2 and 
                        abs(com_z - half) < half/2):
                        print("  Using 'zyx' orientation (sparse format)")
                        grid = grid_zyx
                    else:
                        print("  'zyx' orientation gave off-center results, trying 'xyz'...")
                        grid_xyz, _, _ = read_sparse_voxel_bin(args.input, orientation='xyz')
                        grid = grid_xyz
                else:
                    grid = grid_zyx
            except:
                grid, voxel_size, N = read_sparse_voxel_bin(args.input, orientation='zyx')
        else:
            grid, voxel_size, N = read_sparse_voxel_bin(args.input, orientation=args.orientation)
        
        print(f"Successfully read grid: N={N}, voxel_size={voxel_size:.6f}")
        print(f"Grid shape: {grid.shape}")
        print(f"Grid dtype: {grid.dtype}")
        
        # Print statistics
        print_grid_stats(grid)
        
        if args.stats:
            return 0
        
        # Save outputs
        if args.output:
            print(f"Saving dense binary to: {args.output}")
            save_dense_grid(grid, voxel_size, N, args.output)
            print("Dense binary saved successfully")
        
        if args.npy:
            print(f"Saving .npy file to: {args.npy}")
            write_npy(grid, args.npy)
            print(".npy saved successfully")
        
        if args.vtk:
            print(f"Saving VTK file to: {args.vtk}")
            write_vtk(grid, voxel_size, N, args.vtk)
            print("VTK saved successfully")
        
        # If no output specified, save to default
        if not any([args.output, args.npy, args.vtk]):
            default_output = Path(args.input).stem + '_dense.bin'
            print(f"No output specified, saving to: {default_output}")
            save_dense_grid(grid, voxel_size, N, default_output)
            print("Dense binary saved successfully")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())