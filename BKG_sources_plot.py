import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff

import mplhep as hep

hep.style.use("LHCb2")

plt.rcParams.update({
    'font.size': 6,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{color}'
})


mmu = 105.6583755 # https://pdglive.lbl.gov/Particle.action?node=S004&init=0
mmu2 = mmu**2
mB = 5279.34 # https://pdglive.lbl.gov/Particle.action?init=0&node=S041&home=MXXX045
mB2 = mB**2
mK = 493.677 # https://pdglive.lbl.gov/Particle.action?node=S010&home=sumtabM
mK2 = mK**2
mpi = 139.6
hbar_in_MeV = 6.582119569 * 10E-19
B_lifetime = 1638 * 10E-12

def double_crystal_ball(x, alphaL, nL, alphaR, nR, mean, sigma):
    """
    Double Crystal Ball function.
    
    Parameters:
    x      : float or array_like : Input variable
    alphaL : float               : Alpha parameter for the left side
    nL     : float               : n parameter for the left side
    alphaR : float               : Alpha parameter for the right side
    nR     : float               : n parameter for the right side
    mean   : float               : Mean of the Gaussian core
    sigma  : float               : Standard deviation of the Gaussian core
    
    Returns:
    float or array_like : Output of the double crystal ball function
    """
    
    # Define constants
    A_L = (nL / np.abs(alphaL))**nL * np.exp(-alphaL**2 / 2)
    B_L = nL / np.abs(alphaL) - np.abs(alphaL)
    A_R = (nR / np.abs(alphaR))**nR * np.exp(-alphaR**2 / 2)
    B_R = nR / np.abs(alphaR) - np.abs(alphaR)
    
    # Normalized value
    z = (x - mean) / sigma
    
    # Initialize output array
    out = np.zeros_like(z)
    
    # Gaussian core
    mask_gauss = (z > -alphaL) & (z < alphaR)
    out[mask_gauss] = np.exp(-0.5 * z[mask_gauss]**2)
    
    # Left tail
    mask_left = (z <= -alphaL)
    out[mask_left] = A_L * (B_L - z[mask_left])**(-nL)
    
    # Right tail
    mask_right = (z >= alphaR)
    out[mask_right] = A_R * (B_R + z[mask_right])**(-nR)
    
    return out






Q = np.linspace(mB-100, mB+600, 1000)
Q2 = Q**2

A = 200
B = 1000
C = -Q[0]
# create exponental
CMB = A*np.exp(-((Q+ C)/B ))

# create peaking
width = 20#hbar_in_MeV / B_lifetime
SIG = double_crystal_ball(Q,
                          alphaL=2.1,
                          nL=0.8,
                          alphaR=2.2,
                          nR=2.8,
                          mean=mB,
                          sigma=width)
SIG /= np.max(SIG)
SIG *= 4*10E4



# create pi as ka
MIS = double_crystal_ball(Q,
                          alphaL=2.2,
                          nL=2,
                          alphaR=0.4,
                          nR=5.0,
                          mean=mB+70,
                          sigma=width)
MIS /= np.max(MIS)
MIS *= 0.6*10E2



# combine and plot
plt.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Total Reconstructed', zorder=3)
plt.plot(Q, SIG, lw=2, linestyle='dashed', color='red', label=r'$B\to K\mu\mu$ True Signal', zorder=2)
plt.plot(Q, CMB, lw=2, linestyle='dashed', color='blue', label=r'Combinatorial Background', zorder=2)
plt.plot(Q, MIS, lw=1, linestyle='dashed', color='gray', label=r'$\pi\Rightarrow K$ Misidentified', zorder=1)
plt.axvspan(mB-50,mB+50, zorder=0, alpha=0.2, color='red', lw=0)
plt.axvspan(5460,5860, zorder=0, alpha=0.2, color='blue', lw=0)
plt.axvline(mB, zorder=0, lw = 1.5, color='red')
plt.legend()
plt.xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
plt.ylabel(r'$\frac{1}{A}\frac{d\Gamma}{dm_{K\mu\mu}}$ [MeV$^{-1}$]', loc='center')
plt.yscale("log")
plt.xlim(Q[0], Q[-1])
plt.ylim(10+0.0001, (10**6)-1)
plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/ratios/backgrounds.pdf')