from termcolor import colored, cprint
cprint("Starting...", "light_cyan")

cprint("\nImporting packages...", "light_cyan")
import numpy as np
import matplotlib.pyplot as plt
from PDF_formulation import dGamma_dq, mmu, mB, mK, mb, ms
from peaks_hood import parameters as p, Y, DCB_params, res_model
from scipy.signal import convolve
cprint("Imports completed.", "light_green")



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

cprint("\nConstructing PDF...", "light_cyan")
PDF = dGamma_dq(q2, ctl=None, C_7=C_7, C_S=C_S, C_P=C_P, C_V=C_9, C_A=C_10, C_T=C_T, C_T5=C_T5)
cprint("Constructed PDF.", "light_green")

R1_upper = 1800
R2_upper = 3400

R2_mid = (R1_upper + R2_upper) / 2

# split PDF into sections
PDF_1 = PDF[m_mumu < [R1_upper]]
m_mumu_1 = m_mumu[m_mumu < [R1_upper]]
PDF_2 = PDF[np.abs(m_mumu - R2_mid) <= (R2_upper - R1_upper) / 2]
m_mumu_2 = m_mumu[np.abs(m_mumu - R2_mid) <= (R2_upper - R1_upper) / 2]
PDF_3 = PDF[m_mumu >= R2_upper]
m_mumu_3 = m_mumu[m_mumu >= R2_upper]

# generate reoslution models
p1 = DCB_params[0]
p2 = DCB_params[1]
p3 = DCB_params[2]
x1 = np.linspace(-75, 75, int(len(m_mumu) * 150/(m_mumu_1[-1]-m_mumu_1[0]))) # placeholder for now
x2 = np.linspace(-75, 75, int(len(m_mumu) * 150/(m_mumu_2[-1]-m_mumu_2[0]))) # placeholder for now
x3 = np.linspace(-75, 75, int(len(m_mumu) * 150/(m_mumu_3[-1]-m_mumu_3[0]))) # placeholder for now
RES_1 = res_model(x=x1, sigma_dcb=p1['sigma_dcb'], sigma_gauss=p1['sigma_gauss'], alpha=p1['alpha'], n_L=p1['n_L'], n_R=p1['n_R'], f_dcb=p1['f_dcb'])
RES_2 = res_model(x=x2, sigma_dcb=p2['sigma_dcb'], sigma_gauss=p2['sigma_gauss'], alpha=p2['alpha'], n_L=p2['n_L'], n_R=p2['n_R'], f_dcb=p2['f_dcb'])
RES_3 = res_model(x=x3, sigma_dcb=p3['sigma_dcb'], sigma_gauss=p3['sigma_gauss'], alpha=p3['alpha'], n_L=p3['n_L'], n_R=p3['n_R'], f_dcb=p3['f_dcb'])


x = np.linspace(-75, 75, int(len(m_mumu) * 150/(m_mumu[-1]-m_mumu[0])))
RES_12 = res_model(x=x, sigma_dcb=p1['sigma_dcb'], sigma_gauss=p1['sigma_gauss'], alpha=p1['alpha'], n_L=p1['n_L'], n_R=p1['n_R'], f_dcb=p1['f_dcb'])
RES_22 = res_model(x=x, sigma_dcb=p2['sigma_dcb'], sigma_gauss=p2['sigma_gauss'], alpha=p2['alpha'], n_L=p2['n_L'], n_R=p2['n_R'], f_dcb=p2['f_dcb'])
RES_32 = res_model(x=x, sigma_dcb=p3['sigma_dcb'], sigma_gauss=p3['sigma_gauss'], alpha=p3['alpha'], n_L=p3['n_L'], n_R=p3['n_R'], f_dcb=p3['f_dcb'])
RES_12 /= np.trapz(RES_12, x)
RES_22 /= np.trapz(RES_22, x)
RES_32 /= np.trapz(RES_32, x)
cprint(f"{np.trapz(RES_12, x)}","magenta")
cprint(f"{np.trapz(RES_22, x)}","magenta")
cprint(f"{np.trapz(RES_32, x)}","magenta")
# convolve
section_1 = convolve(PDF_1, RES_1, mode='same')
section_2 = convolve(PDF_2, RES_2, mode='same')
section_3 = convolve(PDF_3, RES_3, mode='same')

blur_1 = convolve(PDF, RES_12, mode='same')
blur_2 = convolve(PDF, RES_22, mode='same')
blur_3 = convolve(PDF, RES_32, mode='same')

post_blur = np.concatenate((section_1, section_2, section_3))

glue_1 = blur_1[m_mumu < [R1_upper]]
glue_2 = blur_2[np.abs(m_mumu - R2_mid) <= (R2_upper - R1_upper) / 2]
glue_3 = blur_3[m_mumu >= R2_upper]

post_blur2 = np.concatenate((glue_1, glue_2, glue_3))

# plot
plt.plot(m_mumu, PDF)
plt.yscale('log')
plt.savefig("ZAKNOW.pdf")
plt.close()
plt.plot(m_mumu_1, PDF_1, color='red')
plt.plot(m_mumu_2, PDF_2, color='blue')
plt.plot(m_mumu_3, PDF_3, color='green')
plt.yscale('log')
plt.savefig("ZAKNOWSPLIT.pdf")
plt.close()
#plt.plot(m_mumu_1, section_1, color='red')
#plt.plot(m_mumu_2, section_2, color='red')
#plt.plot(m_mumu_3, section_3, color='red')
plt.plot(m_mumu, post_blur, color='red')
plt.plot(m_mumu, PDF, color='blue')
plt.yscale('log')
plt.savefig("ZAKNOWCOMBINED.pdf")
plt.close()
plt.plot(m_mumu, post_blur2, color='red')
plt.plot(m_mumu, PDF, color='blue')
plt.yscale('log')
plt.savefig("ZAKNOWCOMBINED2.pdf")
plt.close()