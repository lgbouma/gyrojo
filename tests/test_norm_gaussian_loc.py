import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

def v0():
    # Setting up the grid
    y_grid = np.linspace(-14, 6, 1000)
    x_grid = np.linspace(3800, 6200, 1001)

    # Generating the Gaussian probability distribution for y
    gaussian_y = norm.pdf(y_grid, loc=0, scale=1)

    # Generating the uniform probability distribution for x
    uniform_x = uniform.pdf(x_grid, loc=x_grid.min(), scale=(x_grid.max() - x_grid.min()))

    # Creating a 2D image by multiplying the distributions
    img = gaussian_y[:, None] * uniform_x[None, :]

    # Plotting the image
    plt.figure(figsize=(10, 5))
    plt.imshow(img, extent=(x_grid.min(), x_grid.max(), y_grid.min(),
                         y_grid.max()), aspect='lower')
    plt.colorbar()
    plt.title('2D Probability Distribution: Gaussian in Y, Uniform in X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def mean_function(x, a=-0.00005, b=2):
    """Quadratic polynomial function centered on x = 5000."""
    return a * (x - 5000) ** 2 + b

def mean_function(x, a=-0.00005, b=-3, c=0):
    """Quadratic polynomial function centered on x = 5000."""
    return 4*np.sin(x/(1e2)) + b + c*(x/1e3)


def v1():
    # Setting up the grid
    y_grid = np.linspace(-14, 6, 1000)
    x_grid = np.linspace(3800, 6200, 1001)

    # Apply the mean function to the x_grid
    mean_y = mean_function(x_grid)

    # Generating the uniform probability distribution for x
    uniform_x = uniform.pdf(x_grid, loc=x_grid.min(), scale=(x_grid.max() - x_grid.min()))

    # Correct the reshaping of the Gaussian distribution array
    gaussian_y_adjusted = np.array([norm.pdf(y_grid, loc=mean, scale=1) for mean in mean_y]).T

    # Recalculate the image by correctly multiplying the adjusted Gaussian with the uniform distribution
    img_adjusted = gaussian_y_adjusted * uniform_x[None, :]

    # Replotting the new image
    plt.figure(figsize=(10, 5))
    plt.imshow(img_adjusted, extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('2D Probability Distribution: Gaussian around Quadratic Mean in Y, Uniform in X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def v2():
    # Setting up the grid
    y_grid = np.linspace(-14, 6, 1000)
    x_grid = np.linspace(3800, 6200, 1001)

    # Apply the mean function to the x_grid
    mean_y = mean_function(x_grid)

    # Generating the uniform probability distribution for x
    uniform_x = uniform.pdf(x_grid, loc=x_grid.min(), scale=(x_grid.max() - x_grid.min()))

    # Correct the reshaping of the Gaussian distribution array
    gaussian_y_adjusted = norm.pdf(y_grid[:, None], loc=mean_y, scale=1)

    # Recalculate the image by correctly multiplying the adjusted Gaussian with the uniform distribution
    img_adjusted = gaussian_y_adjusted * uniform_x[None, :]

    # Replotting the new image
    plt.figure(figsize=(10, 5))
    plt.imshow(img_adjusted, extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('2D Probability Distribution: Gaussian around Quadratic Mean in Y, Uniform in X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def v3():
    # Re-importing necessary libraries and redefining all variables and functions due to the reset state
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm, uniform

    y_grid = np.linspace(-14, 6, 1000)
    x_grid = np.linspace(3800, 6200, 1001) - 5000
    uniform_x = uniform.pdf(x_grid, loc=x_grid.min(), scale=(x_grid.max() - x_grid.min()))

    # Generate the original Gaussian distribution
    imgs = []
    bs = list(np.arange(-8,4,2.5))
    cs = bs
    alpha_masks = []
    from matplotlib import cm
    colormaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Purples, cm.Greys]
    for b,c in zip(bs, cs):
        mean_y = mean_function(x_grid, b=b, c=c)
        gaussian_y = norm.pdf(y_grid[:, None], loc=mean_y, scale=1)
        img = gaussian_y * uniform_x[None, :]
        imgs.append(img)
        threshold = np.max(img) / 10000
        alpha_mask = np.where(img >= threshold, 0.8, 0.)
        alpha_masks.append(alpha_mask)


    # Create the plot with two distributions using different colormaps
    plt.figure(figsize=(10, 5))
    for img, am, _cm in zip(imgs, alpha_masks, colormaps):
        plt.imshow(img, extent=(x_grid.min(), x_grid.max(), y_grid.min(),
                                y_grid.max()), aspect='auto', origin='lower',
                   cmap=_cm, alpha=am)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




if __name__ == "__main__":
    v3()
