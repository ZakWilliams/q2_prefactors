import numpy as np
import matplotlib.pyplot as plt

import mplhep as hep

hep.style.use("LHCb2")

plt.rcParams.update({
    'font.size': 6,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{color}'
})




ctl = np.linspace(-1, 1, 1000)
J0 = np.array([0.5 for ctl_ in ctl])
J1 = np.array([1 - ctl_**2 for ctl_ in ctl])
J2 = np.array([ctl_**2 for ctl_ in ctl])

#J0 = J0 / np.trapz(J0, ctl)
#J1 = J1 / np.trapz(J1, ctl)
#J2 = J2 / np.trapz(J2, ctl)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(ctl, J0, label=r'$J = 0; j = S,P$', lw=2, color='blue')
plt.plot(ctl, J1, label=r'$J = 1; j = 9,10$', lw=2, color='red')
plt.plot(ctl, J2, label=r'$J = 2; j = T,T5$', lw=2, color='green')
plt.xlabel(r'$\cos\theta_\mu$', loc='center')
plt.ylabel(r'$A_J\frac{d\Gamma}{d\cos\theta_\mu} - B_J$', loc='center')
ax.get_yaxis().set_ticks([])
plt.xlim(-1, 1)
plt.ylim(0, J2[-1])
plt.legend(loc = (0.32, 0.56))
ax.set_aspect(1.2)
plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/ratios/ctl_shapes.pdf')
plt.close()
