"""
Contents:
    | get_empirical_prot_uncertainties
"""
import numpy as np, matplotlib.pyplot as plt

def get_empirical_prot_uncertainties(prot):
    """
    Assert 1% Prot uncertainties below 15 days, and a linear increase
    thereafter, with slope set to guarantee 3% Prot uncertainties at
    30 days.
    """

    assert isinstance(prot, (int, float, np.ndarray, list))

    if isinstance(prot, (int, float)):
        prot = np.ndarray([prot])

    prot_err = np.zeros(len(prot))

    # fixed component
    sel = (prot <= 15)
    prot_err[sel] = 0.01 * prot[sel]

    # linear envelope
    y1 = 0.03*30
    x1 = 30
    y0 = 0.01*15
    x0 = 15
    m = (y1-y0)/(x1-x0)
    b = y1 - m*x1

    fn = lambda x: m*x + b

    sel = (prot > 15)
    prot_err[sel] = fn(prot[sel])

    return prot_err


def test_prot_uncerts():

    prots = np.linspace(0,50,1000)
    prot_uncs = get_empirical_prot_uncertainties(prots)
    plt.plot(prots, prot_uncs)
    plt.show()


if __name__ == "__main__":
    test_prot_uncerts()
