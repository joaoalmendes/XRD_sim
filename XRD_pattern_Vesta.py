import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# === Load your file ===
df = pd.read_csv("structures_factors/CsV3Sb5_StructureFactors.txt", delim_whitespace=True)
df.columns = ['h', 'k', 'l', 'd', 'd_A', 'F_real', 'F_imag', 'F_abs', '2theta', 'I', 'M']
df["F_squared"] = df["F_abs"] ** 2

# === Lattice constant ===
a = 5.48  # Angstroms

# === Convert (h,k,l) into (kx,ky,kz) for a hexagonal lattice ===
def hkl_to_kxkykz_hexagonal(h, k, l, a):
    # Reciprocal lattice vectors for a hexagonal lattice in the (h,k,0) plane
    # b1 = (2π/a) * (1, -1/√3, 0)
    # b2 = (2π/a) * (0, 2/√3, 0)
    kx = (2 * np.pi / a) * h
    ky = (2 * np.pi / a) * (-h / np.sqrt(3) + 2 * k / np.sqrt(3))
    kz = (2 * np.pi / a) * l  # Along c-axis (simplified, assuming orthogonal for now)
    return kx, ky, kz

df["kx"], df["ky"], df["kz"] = hkl_to_kxkykz_hexagonal(df["h"], df["k"], df["l"], a)

# === Apply Symmetry Operations for P6/mmm Space Group ===
def apply_hexagonal_symmetry(df):
    dfs = [df]
    # Symmetry operations for P6/mmm in the (h,k,0) plane
    for op in [
        lambda h, k, l: (k, h, l),           # (h,k,0) -> (k,h,0)
        lambda h, k, l: (-h, -k, l),        # (h,k,0) -> (-h,-k,0)
        lambda h, k, l: (h, -h-k, l),       # (h,k,0) -> (h,-h-k,0)
        lambda h, k, l: (-k, -h, l),        # (h,k,0) -> (-k,-h,0)
        lambda h, k, l: (-h-k, h, l),       # (h,k,0) -> (-h-k,h,0)
        lambda h, k, l: (k, h-k, l),        # (h,k,0) -> (k,h-k,0)
    ]:
        df_sym = df.copy()
        df_sym["h"], df_sym["k"], df_sym["l"] = op(df["h"], df["k"], df["l"])
        # Recalculate kx, ky, kz for the new (h,k,l)
        df_sym["kx"], df_sym["ky"], df_sym["kz"] = hkl_to_kxkykz_hexagonal(df_sym["h"], df_sym["k"], df_sym["l"], a)
        dfs.append(df_sym)
    
    df_combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["h", "k", "l"])
    return df_combined

df = apply_hexagonal_symmetry(df)
print(f"Number of reflections after symmetry: {len(df)}")

# === Apply Form Factor Correction ===
def apply_form_factor_correction(df, a, B=1.0):
    G = np.sqrt(df["kx"]**2 + df["ky"]**2 + df["kz"]**2)
    form_factor = np.exp(-B * (G / (2 * np.pi / a))**2)
    df["F_squared_corrected"] = df["F_squared"] * form_factor**2
    return df

df = apply_form_factor_correction(df, a, B=1.0)

# === 3D Reciprocal Space Reconstruction ===
def reconstruct_3d_reciprocal_space(df, value_column="F_squared_corrected", grid_range=4, grid_points=200):
    # Adjust grid range to account for hexagonal scaling
    k_range_x = grid_range * (2 * np.pi / a)
    k_range_y = grid_range * (2 * np.pi / a) * (2 / np.sqrt(3))  # Adjusted for ky scaling
    k_range_z = grid_range * (2 * np.pi / a)
    
    kx = np.linspace(-k_range_x, k_range_x, grid_points)
    ky = np.linspace(-k_range_y, k_range_y, grid_points)
    kz = np.linspace(-k_range_z, k_range_z, grid_points)
    
    points = np.vstack((df["kx"], df["ky"], df["kz"])).T
    values = df[value_column].values
    
    intensity_3d = np.zeros((grid_points, grid_points, grid_points))
    counts = np.zeros((grid_points, grid_points, grid_points))
    
    for i in range(len(points)):
        ix = np.argmin(np.abs(kx - points[i, 0]))
        iy = np.argmin(np.abs(ky - points[i, 1]))
        iz = np.argmin(np.abs(kz - points[i, 2]))
        intensity_3d[ix, iy, iz] += values[i]
        counts[ix, iy, iz] += 1
    
    mask = counts > 0
    intensity_3d[mask] /= counts[mask]
    
    intensity_3d = gaussian_filter(intensity_3d, sigma=2.0)
    
    interpolator = RegularGridInterpolator((kx, ky, kz), intensity_3d, method='linear', bounds_error=False, fill_value=0)
    return kx, ky, kz, interpolator

# === Extract 2D Slice and Plot ===
def plot_2d_slice(kx, ky, kz, interpolator, plane="kz", plane_value=0, title="Plane (h,k,0)", value_column="F_squared_corrected"):
    grid_range_x = kx[-1] / (2 * np.pi / a)
    grid_range_y = ky[-1] / ((2 * np.pi / a) * (2 / np.sqrt(3)))
    
    kx_fine = np.linspace(-kx[-1], kx[-1], 500)
    ky_fine = np.linspace(-ky[-1], ky[-1], 500)
    kx_2d, ky_2d = np.meshgrid(kx_fine, ky_fine, indexing='ij')
    
    points_2d = np.vstack([kx_2d.ravel(), ky_2d.ravel(), np.full_like(kx_2d.ravel(), plane_value)]).T
    zi = interpolator(points_2d).reshape(len(kx_fine), len(ky_fine))
    
    zi = np.nan_to_num(zi, nan=0.0)
    zi = gaussian_filter(zi, sigma=2.0)
    zi = np.log1p(zi)
    zi[zi < np.percentile(zi, 20)] = 0
    
    # Adjust extent to reflect the hexagonal scaling
    extent = [-grid_range_x, grid_range_x, -grid_range_y, grid_range_y]

    # First Brillouin zone boundary (hexagonal shape)
    bz_boundary = 4 * np.pi / (np.sqrt(3) * a) / (2 * np.pi / a)  # In units of 2π/a

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 5))
    im = plt.imshow(zi, extent=extent, origin='lower', cmap='magma', aspect='equal', 
                    interpolation='bilinear', vmin=np.percentile(zi[zi > 0], 10), vmax=np.percentile(zi, 90))
    
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\log(1 + F^2)$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.xlabel(r"$k_x\ (2\pi/a)$", fontsize=14)
    plt.ylabel(r"$k_y\ (2\pi/a)$", fontsize=14)
    plt.title(title, fontsize=16)
    
    # Plot hexagonal Brillouin zone boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    bz_x = bz_boundary * np.cos(theta)
    bz_y = (bz_boundary * np.sin(theta)) * (2 / np.sqrt(3))  # Adjust for ky scaling
    plt.plot(bz_x, bz_y, 'w--', linewidth=1, label='1st BZ')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

# === Main Execution ===
grid_range = 4
grid_points = 200
kx, ky, kz, interpolator = reconstruct_3d_reciprocal_space(df, grid_range=grid_range, grid_points=grid_points)
plot_2d_slice(kx, ky, kz, interpolator, plane="kz", plane_value=0, title="Plane (h,k,0)")