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

    # switches for easier referencing later
    binned_data_provided = False if binned_data is binned_data_sentinel else True
    unbinned_data_provided = False if unbinned_data is unbinned_data_sentinel else True

    # argument contradictions
    # if both binned and unbinned data provided
    if binned_data_provided and unbinned_data_provided:
        raise Exception(colored(f"Your call of {plot_backgrounds_along_m_Kmumu.__name__} is attempting to load both binned and unbinned data for plotting. Please pass only one data set.", "red"))
    # if calling to plot disparities without providing data
    if plot_disagreement_underneath and not (binned_data_provided or unbinned_data_provided):
        raise Exception(colored(f"Your call of {plot_backgrounds_along_m_Kmumu.__name__} is called to plot the disparity between data and model without any data being provided.", "red"))














    TOTAL = CMB + SIG + MIS
    area = np.trapz(TOTAL, Q)
    #print(area)
    TOTAL = CMB + SIG + MIS
    centres = [5500, 5580, 5660, 5740, 5820]
    edges = [5540, 5620, 5700, 5780]

    primary_plot_coords = [0.1/1.5, 0.5/1.2, 0.8/1.5, 0.6/1.2]
    secondary_plot_coords = [0.1/1.5, 0.3/1.2, 0.8/1.5, 0.2/1.2]
    tertiary_plot_coords = [0.1/1.5, 0.1/1.2, 0.8/1.5, 0.2/1.2]

    if plot_disagreement_underneath:
        axes_disp_coords = secondary_plot_coords
        axes_frac_coords = tertiary_plot_coords
    elif plot_frac_underneath:
        axes_disp_coords = []
        axes_frac_coords = secondary_plot_coords

    #fig, (axes_main, axes_frac) = plt.subplots(nrows=2, sharex=True)
    fig = plt.figure(figsize=(18, 14.4))
    axes_main = fig.add_axes(primary_plot_coords)
    if plot_disagreement_underneath:
        axes_disp = fig.add_axes(axes_disp_coords)
    if plot_frac_underneath:
        axes_frac = fig.add_axes(axes_frac_coords)
        

    

    # do the drawing of the data:
    if fill_or_lines == 'lines':
        axes_main.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        axes_main.plot(Q, SIG, lw=1.5, color='red', label=r'$B\to K\mu\mu$ Truth', zorder=2)
        axes_main.plot(Q, MIS, lw=1.5, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', zorder=2)
        axes_main.plot(Q, CMB, lw=1.5, color='blue', label=r'Combinatorial Background', zorder=2)
    elif fill_or_lines == 'fill':
        if plot_total_line_above_fill: axes_main.plot(Q, CMB + SIG + MIS, color='black', lw=2, label=r'$B\to K\mu\mu$ Reconstructed', zorder=3)
        axes_main.fill_between(Q, CMB+MIS, CMB+MIS+SIG, color='red', label=r'$B\to K\mu\mu$ Truth', alpha=alpha, lw=0)
        axes_main.fill_between(Q, CMB, CMB+MIS, color='green', label=r'$\pi\Rightarrow K$ Mis-ID Background', alpha=alpha, lw=0)
        axes_main.fill_between(Q, CMB, color='blue', label=r'Combinatorial Background', alpha=alpha, lw=0)
    else:
        raise Exception("ERR")#colored(f"the argument 'fill_or_lines' in plot_backgrounds_along_m_Kmumu is invalid. You have entered: {fill_or_lines}, please enter either 'lines' or 'fill'.", "red"))
        
    if show_windows:
        axes_main.axvspan(mB-50,mB+50, alpha=0.2, color='red', lw=0, zorder=4, label='Signal Window')
        axes_main.axvspan(5460,5860, alpha=0.2, color='blue', lw=0, zorder=4, label='UMSB')

    if vlines_in_UMSB:
        for centre in centres:
            axes_main.axvline(centre, zorder=0, lw = 1, color='blue')
        for edge in edges:
            axes_main.axvline(edge, zorder=0, lw = 0.3, color='blue')
    if show_mB:
        axes_main.axvline(mB, zorder=0, lw = 1, color='red')
    if not plot_frac_underneath: axes_main.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
    if plot_frac_underneath: axes_main.set_xticklabels([])
    axes_main.set_ylabel(r'Events', loc='center')
    if logarithmic: axes_main.set_yscale("log")
    axes_main.set_xlim(xlims[0], xlims[1])
    axes_main.set_ylim(ylims[0], ylims[1])

    if binned_data_provided:
        # calculate bin centres and bin widths from binned_data[0]
        bin_lower = binned_data[0][:-1]
        bin_upper = binned_data[0][1:]

        Q_bin_centres = (bin_lower + bin_upper) / 2
        Q_bin_widths = (bin_upper - bin_lower) / 2

        plotting_data = [Q_bin_centres, Q_bin_widths, binned_data[1], binned_data[2], binned_data[3]]
    elif unbinned_data_provided:
        if bin_count is bin_count_sentinel:
            raise Exception(colored("You are loading unbinned data into plot_backgrounds_along_m_Kmumu. Please specify a number for bin_count.", "red"))
        
        pseudo_bins = np.histogram(unbinned_data, bins=bin_count, range=(Q[0], Q[-1]))
        pseudo_err_top = np.sqrt(pseudo_bins[0])
        pseudo_err_bottom = np.sqrt(pseudo_bins[0])

        bin_lower = pseudo_bins[1][:-1]
        bin_upper = pseudo_bins[1][1:]
        Q_bin_centres = (bin_lower + bin_upper) / 2
        Q_bin_widths = (bin_upper - bin_lower) / 2

        event_count = sum(pseudo_bins[0])
        print(colored(f"Area underneath binned histogram: {event_count}", "green"))

        plotting_data = [Q_bin_centres, Q_bin_widths, pseudo_bins[0]/(2*Q_bin_widths), pseudo_err_top, pseudo_err_bottom]




    # plot the now binned data
    if binned_data_provided or unbinned_data_provided:
        # Some options so that you can customize the graph with a little less typing
        cap_length = 0.2
        eline_width = 1
        ecolor = 'black'
        for bin_centre, bin_width, y, y_err_pos, y_err_neg in zip(plotting_data[0], plotting_data[1], plotting_data[2], plotting_data[3], plotting_data[4]):
            # mark out the error line
            axes_main.vlines(bin_centre, max(0, y - y_err_neg), y + y_err_pos, lw=eline_width, color=ecolor)
            # mark out the bin width
            axes_main.hlines(y, bin_centre - bin_width, bin_centre + bin_width, lw=eline_width, color=ecolor)
            # cap the vertical error line
            axes_main.hlines(y + y_err_pos, bin_centre - (cap_length * bin_width), bin_centre + (cap_length * bin_width), lw=eline_width, color=ecolor)
            axes_main.hlines(max(0, y - y_err_pos), bin_centre - (cap_length * bin_width), bin_centre + (cap_length * bin_width), lw=eline_width, color=ecolor)


    # Basic legend call. Replace with custom legend later
    axes_main.legend(loc=(1.02, 0.58))




    if plot_disagreement_underneath:
        axes_disp.set_xlim(xlims[0], xlims[1])
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


        axes_disp.scatter(plotting_data[0], normalised_disparities)
        axes_disp.axhline(y=0)





    # plot the fraction of the combinatorial background according to the models.
    if plot_frac_underneath:
        CMB_frac = CMB/TOTAL
        MIS_frac = MIS/TOTAL
        SIG_frac = SIG/TOTAL
        axes_frac.fill_between(Q, CMB_frac, alpha=alpha, color='blue', lw=0)
        axes_frac.fill_between(Q, CMB_frac, CMB_frac+MIS_frac, alpha=alpha, color='green', lw=0)
        axes_frac.fill_between(Q, CMB_frac+MIS_frac, CMB_frac+MIS_frac+SIG_frac, alpha=alpha, color='red', lw=0)
        axes_frac.set_ylim(0.000001, 1-0.000001)
        axes_frac.set_xlim(xlims[0], xlims[1])
        axes_frac.set_xlabel(r'$m_{K\mu\mu}$ [MeV]', loc='center')
        axes_frac.set_ylabel(r'Fraction', loc='center')



    fig.savefig(f'{folder_name}/{file_name}.pdf')
    plt.close(fig)
    return