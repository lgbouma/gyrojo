"""
This script generates a distribution of ages for planet-hosting stars based on a
specific age distribution. The age distribution linearly increases from 0 to 3
billion years, with the probability at 0 billion years being 2.75 times lower
than the probability at 3 billion years. The distribution remains uniform from
3 to 10 billion years. The script draws N ages from this distribution and plots
a histogram of the resulting age distribution.
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_star_ages(N):
    """
    Draw N ages from the specified age distribution of planet-hosting stars.

    Args:
        N (int): The number of ages to draw from the distribution.

    Returns:
        numpy.ndarray: An array of N ages drawn from the distribution.
    """
    ages = np.zeros(N)

    for i in range(N):
        while True:
            # Draw a random age between 0 and 10 billion years
            age = np.random.uniform(0, 10)

            age_break = 3
            if age <= age_break:
                # If age is between 0 and 3 billion years, apply the linear increase factor
                factor = (1 / 2.49) + (age / age_break) * (1 - (1 / 2.49))
                if np.random.random() < factor:
                    ages[i] = age
                    break
            else:
                # If age is between 3 and 10 billion years, accept it as is
                ages[i] = age
                break

    return ages


def plot_age_distribution(ages):
    """
    Plot a histogram of the age distribution of planet-hosting stars.

    Args:
        ages (numpy.ndarray): An array of ages to plot in the histogram.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(ages, bins=50, density=True, alpha=0.7)
    plt.xlabel('Age (billion years)')
    plt.ylabel('Probability Density')
    plt.title('Age Distribution of Planet-Hosting Stars')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to run the script.
    """
    # Number of ages to draw
    N = 100000

    # Draw ages from the distribution
    star_ages = draw_star_ages(N)

    # Plot the age distribution
    plot_age_distribution(star_ages)

    N_young = len(star_ages[star_ages < 1])
    f = N_young / N
    print(f"If N_exoplanets=5000, then would expect N={f*5000:.1f} <1gyr planets")



if __name__ == '__main__':
    main()
