import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff
from termcolor import colored
import scipy.interpolate

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
bin_count_sentinel = object()
unbinned_data_sentinel = object()

def plot_backgrounds_along_m_Kmumu(
        Q,
        CMB,
        SIG,
        MIS,
        binned_data = binned_data_sentinel,
        unbinned_data = unbinned_data_sentinel,
        bin_count = bin_count_sentinel,
        folder_name = 'plots',
        file_name = 'backgrounds_along_m_Kmumu',
        highlight_regions = True,
        show_windows = True,
        show_mB = True,
        vlines_in_UMSB = False,
        alpha=0.2,
        fill_or_lines = 'lines',
        plot_total_line_above_fill = False,
        logarithmic = True,
        ylims = [None, None],
        xlims = [mB-100, mB+600],
        plot_frac_underneath = False,
        plot_disagreement_underneath = False,
    ):

    TOTAL = CMB + SIG + MIS
    area = np.trapz(TOTAL, Q)
    #print(area)
    TOTAL = CMB + SIG + MIS
    centres = [5500, 5580, 5660, 5740, 5820]
    edges = [5540, 5620, 5700, 5780]

    #fig, (ax1, ax3) = plt.subplots(nrows=2, sharex=True)
    fig = plt.figure(figsize=(18, 14.4))
    ax1 = fig.add_axes([0.1/1.5, 0.5/1.2, 0.8/1.5, 0.6/1.2])
    if plot_disagreement_underneath:
        ax2 = fig.add_axes([0.1/1.5, 0.3/1.2, 0.8/1.5, 0.2/1.2])
    if plot_frac_underneath:
        ax3 = fig.add_axes([0.1/1.5, 0.1/1.2, 0.8/1.5, 0.2/1.2])
        #ax3.sharex(ax1)
        

    

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
    if not plot_frac_underneath: ax1.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
    if plot_frac_underneath: ax1.set_xticklabels([])
    ax1.set_ylabel(r'Events', loc='center')
    if logarithmic: ax1.set_yscale("log")
    ax1.set_xlim(xlims[0], xlims[1])
    ax1.set_ylim(ylims[0], ylims[1])

    plotting_data=None
    if (binned_data is not binned_data_sentinel) and (unbinned_data is not unbinned_data_sentinel):
        raise Exception(colored("Your call of plot_backgrounds_along_m_Kmumu is attempting to load both binned and unbinned data for plotting. Please pass only one data set.", "red"))
    elif binned_data is not binned_data_sentinel:
        # calculate bin centres and bin widths from binned_data[0]
        bin_lower = binned_data[0][:-1]
        bin_upper = binned_data[0][1:]

        Q_bin_centres = (bin_lower + bin_upper) / 2
        Q_bin_widths = (bin_upper - bin_lower) / 2

        plotting_data = [Q_bin_centres, Q_bin_widths, binned_data[1], binned_data[2], binned_data[3]]
    elif unbinned_data is not unbinned_data_sentinel:
        if bin_count is not bin_count_sentinel:
            raise Exception(colored("You are loading unbinned data into plot_backgrounfs_along_m_Kmumu. Please specify a number for bin_count.", "red"))
        raise Exception(colored("UNBINNED DATA UNIMPLEMENTED","red"))
        # package data into bins and generate errors


    # plot the now binned data
    if plotting_data is not None:
        if (binned_data is not binned_data_sentinel) or (unbinned_data is not unbinned_data_sentinel):
            #ax1.errorbar(plotting_data[0], plotting_data[2], plotting_data[3], plotting_data[1], color='black', elinewidth=1, capsize=0, capthick=2)
            # some options for more convenient customising
            cap_length = 0.2
            eline_width = 1
            ecolor = 'black'
            for bin_centre, bin_width, y, y_err_pos, y_err_neg in zip(plotting_data[0], plotting_data[1], plotting_data[2], plotting_data[3], plotting_data[4]):
                # mark out the error line
                ax1.vlines(bin_centre, max(0, y - y_err_neg), y + y_err_pos, lw=eline_width, color=ecolor)
                # mark out the bin width
                ax1.hlines(y, bin_centre - bin_width, bin_centre + bin_width, lw=eline_width, color=ecolor)
                # cap the vertical error line
                ax1.hlines(y + y_err_pos, bin_centre - (cap_length * bin_width), bin_centre + (cap_length * bin_width), lw=eline_width, color=ecolor)
                ax1.hlines(max(0, y - y_err_pos), bin_centre - (cap_length * bin_width), bin_centre + (cap_length * bin_width), lw=eline_width, color=ecolor)


    # custom legend entry how?
    ax1.legend(loc=(1.02, 0.58))

    if plot_disagreement_underneath and (plotting_data is not None):
        ax2.set_xlim(xlims[0], xlims[1])
        # bin up the total
        # create interpolation object from total
        total_interpolator = scipy.interpolate.interp1d(Q, TOTAL)
        Q_bin_spans = np.array([np.linspace(bottom, top, 100) for bottom, top in zip(bin_lower, bin_upper)])
        freq_bin_spans = [total_interpolator(Q_bin_span) for Q_bin_span in Q_bin_spans]
        model_bin_heights = [np.trapz(freq_bin_span, Q_bin_span)/(bin_top - bin_bottom) for freq_bin_span, Q_bin_span, bin_top, bin_bottom in zip(freq_bin_spans, Q_bin_spans, bin_upper, bin_lower)]
        
        disparities = (plotting_data[2] - model_bin_heights)
        normalised_disparities = np.zeros(len(disparities))
        for i in range(len(disparities)):
            if disparities[i] < 0:
                normalised_disparities[i] = disparities[i] / plotting_data[3][i]
            elif disparities[i] > 0:
                normalised_disparities[i] = disparities[i] / plotting_data[4][i]
            else: # disparities[i] == 0
                normalised_disparities = 0


        print(plotting_data[2])
        print(disparities)
        print(normalised_disparities)
        print(plotting_data[3])
        print(plotting_data[4])
        print("DUMMY")
        ax2.scatter(plotting_data[0], normalised_disparities)
        ax2.axhline(y=0)


    # plot the fraction of the combinatorial background according to the models.
    if plot_frac_underneath:
        CMB_frac = CMB/TOTAL
        MIS_frac = MIS/TOTAL
        SIG_frac = SIG/TOTAL
        ax3.fill_between(Q, CMB_frac, alpha=alpha, color='blue', lw=0)
        ax3.fill_between(Q, CMB_frac, CMB_frac+MIS_frac, alpha=alpha, color='green', lw=0)
        ax3.fill_between(Q, CMB_frac+MIS_frac, CMB_frac+MIS_frac+SIG_frac, alpha=alpha, color='red', lw=0)
        ax3.set_ylim(0.000001, 1-0.000001)
        ax3.set_xlim(xlims[0], xlims[1])
        ax3.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
        ax3.set_ylabel(r'Fraction', loc='center')



    fig.savefig(f'{folder_name}/{file_name}.pdf')
    plt.close(fig)
    return