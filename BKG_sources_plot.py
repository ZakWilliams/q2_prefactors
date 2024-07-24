import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff
#from termcolor import colored

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

# define background plotting function with aesthetics options
# fill or lines?
# vline in umsb
# colors


def plot_backgrounds_along_m_Kmumu(
        Q,
        CMB,
        SIG,
        MIS,
        folder_name = 'plots',
        file_name = 'backgrounds_along_m_Kmumu',
        highlight_regions = True,
        show_windows = True,
        show_mB = True,
        vlines_in_UMSB = False,
        alpha=0.2,
        fill_or_lines = 'lines',
        plot_total_line_above_fill = False,
        normalise=True,
        logarithmic = True,
        ylims = [None, None],
        xlims = [mB-100, mB+600],
        plot_frac_underneath = False,
    ):

    TOTAL = CMB + SIG + MIS
    area = np.trapz(TOTAL, Q)
    #print(area)
    if normalise:
        CMB /= area
        SIG /= area
        MIS /= area
    TOTAL = CMB + SIG + MIS
    centres = [5500, 5580, 5660, 5740, 5820]
    edges = [5540, 5620, 5700, 5780]

    # do the drawing of the data:
    if fill_or_lines == 'lines':
        plt.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        plt.plot(Q, SIG, lw=1.5, color='red', label=r'$B\to K\mu\mu$ Truth', zorder=2)
        plt.plot(Q, CMB, lw=1.5, color='blue', label=r'Combinatorial Background', zorder=2)
        plt.plot(Q, MIS, lw=1.5, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', zorder=1)
    if fill_or_lines == 'fill':
        if plot_total_line_above_fill: plt.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        plt.fill_between(Q, CMB, color='blue', label=r'Combinatorial Background', alpha=alpha, lw=0)
        plt.fill_between(Q, CMB, CMB+MIS, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', alpha=alpha, lw=0)
        plt.fill_between(Q, CMB+MIS, CMB+MIS+SIG, color='red', label=r'$B\to K\mu\mu$ Truth', alpha=alpha, lw=0)
    else:
        raise Exception("ERR")#colored(f"the argument 'fill_or_lines' in plot_backgrounds_along_m_Kmumu is invalid. You have entered: {fill_or_lines}, please enter either 'lines' or 'fill'.", "red"))
        
    if show_windows:
        plt.axvspan(mB-50,mB+50, alpha=0.2, color='red', lw=0, zorder=4)
        plt.axvspan(5460,5860, alpha=0.2, color='blue', lw=0, zorder=4)

    if vlines_in_UMSB:
        for centre in centres:
            plt.axvline(centre, zorder=0, lw = 1, color='blue')
        for edge in edges:
            plt.axvline(edge, zorder=0, lw = 0.3, color='blue')
    if show_mB:
        plt.axvline(mB, zorder=0, lw = 1, color='red')
    plt.legend()
    plt.xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
    plt.ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{dm_{K\mu\mu}}$', loc='center')
    if logarithmic: plt.yscale("log")
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    plt.savefig(f'{folder_name}/{file_name}.pdf')
    plt.close()
    

    if plot_frac_underneath:
        # plot frac of combinatorial background below
        CMB_frac = CMB/TOTAL
        MIS_frac = MIS/TOTAL
        SIG_frac = SIG/TOTAL
        plt.fill_between(Q, CMB_frac, alpha=alpha, color='blue', lw=0)
        plt.fill_between(Q, CMB_frac, CMB_frac+MIS_frac, alpha=alpha, color='green', lw=0)
        plt.fill_between(Q, CMB_frac+MIS_frac, CMB_frac+MIS_frac+SIG_frac, alpha=alpha, color='red', lw=0)
        plt.ylim(0, 1)
        plt.xlim(xlims[0], xlims[1])
        plt.xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
        plt.ylabel(r'$K\mu\mu$ Signal Fraction', loc='center')
        plt.savefig(f'{folder_name}/TEMPORARY_frac_plot.pdf')


    return


folder = 'PDF_helicities_expanded/plots'

plot_backgrounds_along_m_Kmumu(Q=Q,
                               CMB=CMB,
                               SIG=SIG,
                               MIS=MIS,
                               folder_name=folder,
                               fill_or_lines='fill',
                               plot_total_line_above_fill=True,
                               logarithmic=True,
                               alpha=0.4,
                               xlims=[Q[0], Q[-1]],
                               normalise=True,
                               ylims=[0.000001, 0.1],
                               #vlines_in_UMSB=True,
                               plot_frac_underneath=True,
                               )