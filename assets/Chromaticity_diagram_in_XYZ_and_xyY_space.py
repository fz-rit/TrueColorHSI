import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colour

# 1. Generate the spectral locus (chromaticity diagram boundary)
wavelengths = np.linspace(380, 780, 100)
XYZ_locus = np.array([colour.wavelength_to_XYZ(wl) for wl in wavelengths])
# Compute xy coordinates by normalizing X and Y components.
xy_locus = np.array([XYZ[:2] / np.sum(XYZ) for XYZ in XYZ_locus])
Y_const = np.ones_like(xy_locus[:, 0])
xyY_locus = np.hstack((xy_locus, Y_const[:, np.newaxis]))

# Convert spectral locus from XYZ to sRGB for display.
RGB_locus = colour.XYZ_to_sRGB(XYZ_locus)
RGB_locus = np.clip(RGB_locus, 0, 1)

# 2. Generate a dense grid of xyY values within the chromaticity diagram.
x_vals = np.linspace(0, 0.8, 100)
y_vals = np.linspace(0, 0.9, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
x_grid = x_grid.flatten()
y_grid = y_grid.flatten()
Y_grid = np.ones_like(x_grid)  # constant luminance
xyY_values = np.vstack((x_grid, y_grid, Y_grid)).T

# 3. Filter out points outside the chromaticity diagram.
inside_gamut = np.array([colour.is_within_macadam_limits(xyY) for xyY in xyY_values])
xyY_values = xyY_values[inside_gamut]

# 4. Convert xyY → XYZ → sRGB for the grid points.
XYZ_values = colour.xyY_to_XYZ(xyY_values)
RGB_values = colour.XYZ_to_sRGB(XYZ_values)
RGB_values = np.clip(RGB_values, 0, 1)

# 5. Plotting in XYZ space.
fig_xyz = plt.figure(figsize=(10, 8))
ax_xyz = fig_xyz.add_subplot(111, projection='3d')

# Plot the gamut grid points.
ax_xyz.scatter(
    XYZ_values[:, 0], XYZ_values[:, 1], XYZ_values[:, 2],
    c=RGB_values, marker='o', s=5, label='Gamut'
)

# Plot the spectral locus.
ax_xyz.scatter(
    XYZ_locus[:, 0], XYZ_locus[:, 1], XYZ_locus[:, 2],
    c=RGB_locus, marker='o', s=30, label='Spectral Locus'
)

# Annotate the spectral locus points with wavelength numbers (every 10th point).
for i in range(0, len(wavelengths), 10):
    ax_xyz.text(
        XYZ_locus[i, 0],
        XYZ_locus[i, 1],
        XYZ_locus[i, 2],
        f"{int(wavelengths[i])} nm",
        color=RGB_locus[i],
        fontsize=8
    )

ax_xyz.set_xlabel('X')
ax_xyz.set_ylabel('Y')
ax_xyz.set_zlabel('Z')
ax_xyz.set_title('Chromaticity Diagram in XYZ Space')
ax_xyz.legend()

# 6. Plotting in xyY space.
fig_xyY = plt.figure(figsize=(10, 8))
ax_xyY = fig_xyY.add_subplot(111, projection='3d')

# Plot the gamut grid points.
ax_xyY.scatter(
    xyY_values[:, 0], xyY_values[:, 1], xyY_values[:, 2],
    c=RGB_values, marker='o', s=5, label='Gamut'
)

# Plot the spectral locus.
ax_xyY.scatter(
    xyY_locus[:, 0], xyY_locus[:, 1], xyY_locus[:, 2],
    c=RGB_locus, marker='o', s=30, label='Spectral Locus'
)

# Annotate the spectral locus points with wavelength numbers (every 10th point).
for i in range(0, len(wavelengths), 10):
    ax_xyY.text(
        xyY_locus[i, 0],
        xyY_locus[i, 1],
        xyY_locus[i, 2],
        f"{int(wavelengths[i])} nm",
        color=RGB_locus[i],
        fontsize=8
    )

ax_xyY.set_xlabel('x')
ax_xyY.set_ylabel('y')
ax_xyY.set_zlabel('Y')
ax_xyY.set_title('Chromaticity Diagram in xyY Space')
ax_xyY.legend()

plt.show()
