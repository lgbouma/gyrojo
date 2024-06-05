from scipy.stats import binom

def surprise_level(x, y, p=0.8):
    """
    Calculates the probability of observing y or fewer successes out of x trials,
    given the expected probability of success p.
    """
    prob = binom.cdf(y, x, p)
    return prob

# Example usage
x = 1064  # Initial sample size
y = 810   # Number of planets with b < 0.8
p = 0.8   # Expected probability of a planet having b < 0.8

surprise = surprise_level(x, y, p)
print(f"The probability of observing {y} or fewer planets out of {x} is: {surprise:.4f}")
