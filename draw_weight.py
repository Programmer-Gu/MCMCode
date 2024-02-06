import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Assuming data is loaded correctly
data = np.load("muti_att_weight.npy")

# Custom colormap
cmap = LinearSegmentedColormap.from_list("my_cmap", ["blue", "green", "yellow", "red"])

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]))
z_layers = np.array([0.5, 2.0, 3.5])  # Z-axis positions for each layer

for i, z in enumerate(z_layers):
    # Draw each layer using the custom colormap and adjust opacity for visual separation
    cf = ax.contourf(x, y, data[i], zdir='z', offset=z, cmap=cmap, levels=np.linspace(data[i].min(), data[i].max(), 20))
    # Add transparent layer to enhance the stacked effect
    ax.contourf(x, y, np.full(x.shape, z + 0.25), zdir='z', levels=[z, z + 0.5], alpha=0.3, colors=['grey'], offset=z)

# Add colorbar to the left side
plt.colorbar(cf, shrink=1, aspect=10, label='Data Value', location='left')

# Set plot title and remove axis labels and ticks
ax.set_title('Advanced and Aesthetic Stacked Heatmaps', fontsize=16)
ax.set_axis_off()

# Adjust view angle
ax.view_init(elev=30, azim=-10)

plt.tight_layout()
plt.show()
