import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff
#from termcolor import colored

import mplhep as hep

from BKG_constants import mmu, mmu2, mB, mB2, mK, mK2, mpi, mpi2, hbar_in_MeV, B_lifetime

hep.style.use("LHCb2")

plt.rcParams.update({
    'font.size': 6,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{color}'
})

binned_data_sentinel = object()
unbinned_data_sentinel = object()

def plot_backgrounds_along_m_Kmumu(
        Q,
        CMB,
        SIG,
        MIS,
        binned_data = binned_data_sentinel,
        unbinned_data = unbinned_data_sentinel,
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

    #fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])
    if plot_frac_underneath:
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
        #ax2.sharex(ax1)

    # do the drawing of the data:
    if fill_or_lines == 'lines':
        ax1.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        ax1.plot(Q, SIG, lw=1.5, color='red', label=r'$B\to K\mu\mu$ Truth', zorder=2)
        ax1.plot(Q, MIS, lw=1.5, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', zorder=2)
        ax1.plot(Q, CMB, lw=1.5, color='blue', label=r'Combinatorial Background', zorder=2)
    elif fill_or_lines == 'fill':
        if plot_total_line_above_fill: ax1.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        ax1.fill_between(Q, CMB+MIS, CMB+MIS+SIG, color='red', label=r'$B\to K\mu\mu$ Truth', alpha=alpha, lw=0)
        ax1.fill_between(Q, CMB, CMB+MIS, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', alpha=alpha, lw=0)
        ax1.fill_between(Q, CMB, color='blue', label=r'Combinatorial Background', alpha=alpha, lw=0)
    else:
        raise Exception("ERR")#colored(f"the argument 'fill_or_lines' in plot_backgrounds_along_m_Kmumu is invalid. You have entered: {fill_or_lines}, please enter either 'lines' or 'fill'.", "red"))
        
    if show_windows:
        ax1.axvspan(mB-50,mB+50, alpha=0.2, color='red', lw=0, zorder=4, label='Signal Window')
        ax1.axvspan(5460,5860, alpha=0.2, color='blue', lw=0, zorder=4, label='UMSB')

    if vlines_in_UMSB:
        for centre in centres:
            ax1.axvline(centre, zorder=0, lw = 1, color='blue')
        for edge in edges:
            ax1.axvline(edge, zorder=0, lw = 0.3, color='blue')
    if show_mB:
        ax1.axvline(mB, zorder=0, lw = 1, color='red')
    ax1.legend()
    if not plot_frac_underneath: ax1.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
    if plot_frac_underneath: ax1.set_xticklabels([])
    ax1.set_ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{dm_{K\mu\mu}}$', loc='center')
    if logarithmic: ax1.set_yscale("log")
    ax1.set_xlim(xlims[0], xlims[1])
    ax1.set_ylim(ylims[0], ylims[1])







    if plot_frac_underneath:
        # plot frac of combinatorial background below
        CMB_frac = CMB/TOTAL
        MIS_frac = MIS/TOTAL
        SIG_frac = SIG/TOTAL
        ax2.fill_between(Q, CMB_frac, alpha=alpha, color='blue', lw=0)
        ax2.fill_between(Q, CMB_frac, CMB_frac+MIS_frac, alpha=alpha, color='green', lw=0)
        ax2.fill_between(Q, CMB_frac+MIS_frac, CMB_frac+MIS_frac+SIG_frac, alpha=alpha, color='red', lw=0)
        ax2.set_ylim(0.000001, 1-0.000001)
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
        ax2.set_ylabel(r'Fraction', loc='center')
        #plt.savefig(f'{folder_name}/TEMPORARY_frac_plot.pdf')

    fig.savefig(f'{folder_name}/{file_name}.pdf')
    plt.close(fig)
    return