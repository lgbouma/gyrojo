import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate data from log-normal and normal distributions
p_orbit = np.random.lognormal(mean=np.log(10), sigma=1, size=1000)
p_rot = np.random.normal(loc=20, scale=10, size=1000)

# Create a figure and axis using matplotlib's fig, ax interface
fig, ax = plt.subplots(figsize=(8, 6))

# Create the scatter plot
ax.scatter(p_orbit, p_rot, c='k', s=2)

# Set labels and title
ax.set_xlabel('p_orbit')
ax.set_ylabel('p_rot')
ax.set_title('2D Scatter Plot')
ax.update({
'xscale':'log',
'yscale':'log'
})

# Adjust the plot layout
fig.tight_layout()

# Save the figure to a file
fig.savefig('secret_test_results/here_is_the_mcquillan_2013_dearth.png', dpi=300)

plt.show()
