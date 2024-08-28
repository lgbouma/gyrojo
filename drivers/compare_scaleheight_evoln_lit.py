from astropy import units as u
import numpy as np, matplotlib.pyplot as plt

def h_z(t, z0=95, t0=0.5*u.Gyr, alpha=5/3):
    # e.g. eq 4 rana 1991
    return z0 * (1 + t / t0) ** alpha

def mean_h_z(t, z0=95, t0=0.5*u.Gyr):
    # eq 7 rana 1991
    return 3/5 * z0 * t0 / t * (  (1 + t/t0)**(5/3) - 1  )

t = np.linspace(0, 10, 100)*u.Gyr
h_g05 = h_z(t, z0=94.69, t0=5.55*u.Gyr, alpha=5/3)
h_r91 = h_z(t, z0=95, t0=0.5*u.Gyr, alpha=2/3)
mean_h_r91 = mean_h_z(t, z0=95, t0=0.5*u.Gyr)

fig, ax = plt.subplots()
ax.plot(t, h_g05, label='girardi05')
#ax.plot(t, h_r91, label='rana91')
ax.plot(t, mean_h_r91, label='mean rana91 (eq 7)')
ax.legend(fontsize='x-small')
ax.update({
    'xlabel' : 'age [gyr]',
    'ylabel' : 'h_z [pc]',
})

plt.savefig('../results/scale_height_evoln/giradi05_vs_rana91_scaleheight_evoln.png',
           dpi=400, bbox_inches='tight')
