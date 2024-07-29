import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff
import scipy.interpolate

from BKG_plotter import plot_backgrounds_along_m_Kmumu

import mplhep as hep

hep.style.use("LHCb2")

plt.rcParams.update({
    'font.size': 6,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{color}'
})


from BKG_constants import mmu, mmu2, mB, mB2, mK, mK2, mpi, mpi2, hbar_in_MeV, B_lifetime

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

spread = 10

TOT = MIS + CMB + SIG
# Normalize TOT
TOT_area = np.trapz(TOT, Q)
TOT /= TOT_area

# Generate bin boundaries and midpoints
bin_boundaries = np.linspace(Q[0], Q[-1], 11)
bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

# Interpolate TOT over the bin midpoints
biased_bin_data_gen = scipy.interpolate.interp1d(Q, TOT)
binned_freq = np.array(biased_bin_data_gen(bin_midpoints))

# Add random noise to each binned frequency
for i in range(len(binned_freq)):
    binned_freq[i] += np.random.normal(0, spread*np.sqrt(binned_freq[i]/TOT_area))

# Calculate binned errors
binned_errs_top = [spread*np.sqrt(binned_freq[i]/TOT_area) for i in range(len(binned_freq))]
binned_errs_bot = [spread*np.sqrt(binned_freq[i]/TOT_area) for i in range(len(binned_freq))]


folder = 'PDF_helicities_expanded/plots'
# make so can take binned data or unbinned data

plot_backgrounds_along_m_Kmumu(Q=Q,
                               CMB=CMB,
                               SIG=SIG,
                               MIS=MIS,
                               binned_data = [bin_boundaries, binned_freq, binned_errs_top, binned_errs_bot],
                               #unbinned_data = 2,
                               folder_name=folder,
                               fill_or_lines='lines',
                               plot_total_line_above_fill=True,
                               logarithmic=True,
                               alpha=0.4,
                               xlims=[Q[0], Q[-1]],
                               normalise=True,
                               ylims=[0.000001, 0.1],
                               plot_frac_underneath=True,
                               )