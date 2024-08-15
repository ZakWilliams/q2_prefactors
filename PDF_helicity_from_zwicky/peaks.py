from termcolor import colored, cprint
cprint("Starting...", "light_cyan")

cprint("\nImporting packages...", "light_cyan")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
from PDF_formulation import dGamma_dq, mmu, mB, mK, mb, ms
from peaks_hood import parameters as p, Y, DCB_params, res_model
from scipy.signal import convolve
hep.style.use("LHCb2")
cprint("Imports completed.", "light_green")



R1_upper = 1800
R2_upper = 3400
R2_mid = (R1_upper + R2_upper) / 2

conv_lower = -75
conv_upper = 75



cprint("\nConstructing C_j^effs...", "light_cyan")
m_mumu = np.linspace(2*mmu, mB-mK, 10000)
q2 = m_mumu**2

C_7 = np.array([-0.304] * len(m_mumu), dtype=np.complex128)
C_S = np.array([0.0] * len(m_mumu), dtype=np.complex128)
C_P = np.array([0.0] * len(m_mumu), dtype=np.complex128)
C_9 = np.array([4.23] * len(m_mumu), dtype=np.complex128)
C_10 = np.array([-4.11] * len(m_mumu), dtype=np.complex128)
C_T = np.array([0.0] * len(m_mumu), dtype=np.complex128)
C_T5 = np.array([0.0] * len(m_mumu), dtype=np.complex128)

for res_name in p:
    res = p[res_name]
    C_9 += Y(q2, eR=res['mag'], pR=res['phase'], mR=res['mass'], wR=res['width'])
cprint("Constructed C_j^effs.", "light_green")



cprint("\nConstructing PDF_unblurred...", "light_cyan")
PDF_unblurred = dGamma_dq(q2, ctl=None, C_7=C_7, C_S=C_S, C_P=C_P, C_V=C_9, C_A=C_10, C_T=C_T, C_T5=C_T5)
PDF_unblurred_AREA = np.trapz(PDF_unblurred, m_mumu)
cprint(f"PDF_unblurred area: {PDF_unblurred_AREA}. Normalising...","magenta")
PDF_unblurred /= PDF_unblurred_AREA
cprint("Constructed PDF_unblurred.", "light_green")


cprint("\nConstructing Resolution models...", "light_cyan")
# generate resolution models
p1 = DCB_params[0]
p2 = DCB_params[1]
p3 = DCB_params[2]

x = np.linspace(conv_lower, conv_upper, int(len(m_mumu) * (conv_upper-conv_lower)/(m_mumu[-1]-m_mumu[0])))
RES_1 = res_model(x=x, sigma_dcb=p1['sigma_dcb'], sigma_gauss=p1['sigma_gauss'], alpha=p1['alpha'], n_L=p1['n_L'], n_R=p1['n_R'], f_dcb=p1['f_dcb'])
RES_2 = res_model(x=x, sigma_dcb=p2['sigma_dcb'], sigma_gauss=p2['sigma_gauss'], alpha=p2['alpha'], n_L=p2['n_L'], n_R=p2['n_R'], f_dcb=p2['f_dcb'])
RES_3 = res_model(x=x, sigma_dcb=p3['sigma_dcb'], sigma_gauss=p3['sigma_gauss'], alpha=p3['alpha'], n_L=p3['n_L'], n_R=p3['n_R'], f_dcb=p3['f_dcb'])
RES_1 /= np.trapz(RES_1, x)
RES_2 /= np.trapz(RES_2, x)
RES_3 /= np.trapz(RES_3, x)
cprint(f"{np.trapz(RES_1, x)}","magenta")
cprint(f"{np.trapz(RES_2, x)}","magenta")
cprint(f"{np.trapz(RES_3, x)}","magenta")
cprint("Constructed resolution models.", "light_green")

# convolve
blur_1 = convolve(PDF_unblurred, RES_1, mode='same')
blur_2 = convolve(PDF_unblurred, RES_2, mode='same')
blur_3 = convolve(PDF_unblurred, RES_3, mode='same')

# split PDF_unblurred into sections
m_mumu_1 = m_mumu[m_mumu < [R1_upper]]
m_mumu_2 = m_mumu[np.abs(m_mumu - R2_mid) <= (R2_upper - R1_upper) / 2]
m_mumu_3 = m_mumu[m_mumu >= R2_upper]
glue_1 = blur_1[m_mumu < [R1_upper]]
glue_2 = blur_2[np.abs(m_mumu - R2_mid) <= (R2_upper - R1_upper) / 2]
glue_3 = blur_3[m_mumu >= R2_upper]


# recombine the splits
cprint("\nGluing blurred PDFs...", "light_cyan")
PDF_blurred = np.concatenate((glue_1, glue_2, glue_3))
PDF_blurred_AREA = np.trapz(PDF_blurred, m_mumu)
cprint(f"PDF_blurred area: {PDF_blurred_AREA}. Normalising...","magenta")
PDF_blurred /= PDF_blurred_AREA
cprint("blurred PDFs glued.", "light_green")


#LHCb colors
# Example CMYK values
cmyk_LHCb_red = (0.08, 0.95, 0.98, 0.02)  # Pure red in CMYK
cmyk_LHCb_blue = (1, 0.72, 0, 0)  # Pure blue in CMYK

# Convert CMYK to RGB
def cmyk_to_rgb(cmyk):
    c, m, y, k = cmyk
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return r/255, g/255, b/255

# Convert to RGB
LHCb_red = cmyk_to_rgb(cmyk_LHCb_red)
LHCb_darkblue = cmyk_to_rgb(cmyk_LHCb_blue)

# plot
plt.plot(m_mumu, PDF_unblurred)
plt.yscale('log')
plt.savefig("ZAKNOW.pdf")
plt.close()
plt.plot(m_mumu, PDF_blurred, color=LHCb_red, lw=2.6, label=r'$\frac{d\Gamma}{dm_{\mu\mu}}$ Analytic')
plt.plot(m_mumu, PDF_unblurred, color=LHCb_darkblue, lw=2.6, linestyle='dashed', label=r'$\frac{d\Gamma}{dm_{\mu\mu}}$ Detector')
plt.yscale('log')
plt.xlim(m_mumu[0],m_mumu[-1])
plt.ylim(1E-8,1E1)
plt.xlabel(r'$m_{\mu\mu}$ [MeV]', loc='center')
plt.ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{dm_{\mu\mu}}$ [MeV$^{-1}$]', loc='center')
plt.axvline(x=1800, zorder=0, color='gray', lw=1.4)
plt.axvline(x=3400, zorder=0, color='gray', lw=1.4)
plt.legend()
plt.yticks([1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1E0])
# Enable minor ticks and customize their appearance
plt.minorticks_on()
# Manually set the minor ticks on the y-axis
plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
plt.text(900, 0.01, r'Region.1', fontsize=40, verticalalignment='top', rotation='vertical')
plt.text(2500, 0.01, r'Region.2', fontsize=40, verticalalignment='top', rotation='vertical')
plt.text(4000, 0.01, r'Region.3', fontsize=40, verticalalignment='top', rotation='vertical')
plt.savefig("ZAKNOWCOMBINED2.pdf")
plt.close()