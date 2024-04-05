"""
Reproduce Fig5 of Fulton+2017
    (Unclear whether you should use this or Christiansen's parametrization.
    For a quick hack, either is fine)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parameters for the gamma CDF
k = 17.56  # shape parameter
theta = 0.49  # scale parameter (θ)
l = 1  # location parameter is not directly used in gammacdf but can adjust the input

# Range of "t" values
t_values = np.linspace(0, 25, 400)

# Compute gamma CDF for each t value
# Note: gamma.cdf(x, a, scale) where x is the quantile, a is the shape parameter, and scale is the scale parameter.
cdf_values = gamma.cdf(t_values, k, scale=theta, loc=l)

# Plotting
plt.figure(figsize=(6, 6))
plt.plot(t_values, cdf_values, label=f'Gamma CDF\nk={k}, θ={theta}, l={l}')
plt.title('Gamma Cumulative Distribution Function (CDF)')
plt.xlabel('t')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.savefig("test_gammacdf_test.png", bbox_inches='tight')

def completeness_correction(snr):

    k = 17.56  # shape parameter
    theta = 0.49  # scale parameter (θ)
    l = 1  # location parameter is not directly used in gammacdf but can adjust the input

    return gamma.cdf(snr, k, scale=theta, loc=l)

# reading from Fig5, SNR of 10 gives ~60% recovery
assert abs ( completeness_correction(10) - 0.6 ) < 1e-2
