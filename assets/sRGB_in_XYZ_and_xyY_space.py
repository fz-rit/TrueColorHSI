import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colour

# Generate a sparser grid of sRGB values
srgb_values = np.mgrid[0:1:25j, 0:1:25j, 0:1:25j].reshape(3, -1).T

# Convert sRGB to XYZ
XYZ_values = colour.sRGB_to_XYZ(srgb_values)

# Convert XYZ to xyY
xyY_values = colour.XYZ_to_xyY(XYZ_values)

# D65 white point in XYZ and xyY
D65_XYZ = np.array([0.95047, 1.00000, 1.08883])
D65_xyY = colour.XYZ_to_xyY(D65_XYZ)

# Plotting in XYZ space
fig_xyz = plt.figure()
ax_xyz = fig_xyz.add_subplot(111, projection='3d')
ax_xyz.scatter(XYZ_values[:, 0], XYZ_values[:, 1], XYZ_values[:, 2], c=srgb_values, marker='o')
ax_xyz.scatter(D65_XYZ[0], D65_XYZ[1], D65_XYZ[2], color='white', edgecolor='black', s=100, label='D65 White Point')
ax_xyz.set_xlabel('X')
ax_xyz.set_ylabel('Y')
ax_xyz.set_zlabel('Z')
ax_xyz.set_title('sRGB values in XYZ space')
ax_xyz.legend()

# Plotting in xyY space
fig_xyY = plt.figure()
ax_xyY = fig_xyY.add_subplot(111, projection='3d')
ax_xyY.scatter(xyY_values[:, 0], xyY_values[:, 1], xyY_values[:, 2], c=srgb_values, marker='o')
ax_xyY.scatter(D65_xyY[0], D65_xyY[1], D65_xyY[2], color='white', edgecolor='black', s=100, label='D65 White Point')
ax_xyY.set_xlabel('x')
ax_xyY.set_ylabel('y')
ax_xyY.set_zlabel('Y')
ax_xyY.set_title('sRGB values in xyY space')
ax_xyY.legend()

plt.show()