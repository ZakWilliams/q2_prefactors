import numpy as np
import matplotlib.pyplot as plt
import PDF_helicities_expanded.make_BK_DK_ffs as ff
import os

import mplhep as hep

hep.style.use("LHCb2")

plt.rcParams.update({
    'font.size': 6,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{color}'
})


#################################################################################################################################################
# WILSON PARAMETERS
#################################################################################################################################################
# folder name
# plots name

C_S_mag = 0
C_P_mag = 0
C_9_mag = 4.270
C_10_mag = -4.110
C_T_mag = 0
C_T5_mag = 0
C_7_mag = -0.304

C_S_phase = 0
C_P_phase = 0
C_9_phase = 0
C_10_phase = 0
C_T_phase = 0
C_T5_phase = 0
C_7_phase = 0
#################################################################################################################################################
#################################################################################################################################################
# ASSEMBLE PROPER WILSONS
#################################################################################################################################################
C_7 = C_7_mag * (np.cos(C_7_phase) + 1j*np.sin(C_7_phase))
C_S = C_S_mag * (np.cos(C_S_phase) + 1j*np.sin(C_S_phase))
C_P = C_P_mag * (np.cos(C_P_phase) + 1j*np.sin(C_P_phase))
C_9 = C_9_mag * (np.cos(C_9_phase) + 1j*np.sin(C_9_phase))
C_10 = C_10_mag * (np.cos(C_10_phase) + 1j*np.sin(C_10_phase))
C_T = C_T_mag * (np.cos(C_T_phase) + 1j*np.sin(C_T_phase))
C_T5 = C_T5_mag * (np.cos(C_T5_phase) + 1j*np.sin(C_T5_phase))
#################################################################################################################################################
# FIXED DEFINED CONSTANTS
#################################################################################################################################################
mmu = 105.6583755 # https://pdglive.lbl.gov/Particle.action?node=S004&init=0
mmu2 = mmu**2
mB = 5279.34 # https://pdglive.lbl.gov/Particle.action?init=0&node=S041&home=MXXX045
mB2 = mB**2
mK = 493.677 # https://pdglive.lbl.gov/Particle.action?node=S010&home=sumtabM
mK2 = mK**2
mb = 4180 # https://pdglive.lbl.gov/Particle.action?node=Q005&init=0
ms = 93.4 # https://pdglive.lbl.gov/DataBlock.action?node=Q123SM
GF = 1.1663787*10**-5*10**-6 #https://physics.nist.gov/cgi-bin/cuu/Value?gf
fine_struc_const = 7.2973525643 * 10**-3 #https://physics.nist.gov/cgi-bin/cuu/Value?alph
V_ts = 0.04110 #https://pdg.lbl.gov/2023/reviews/rpp2022-rev-ckm-matrix.pdf
V_tb = 0.999118 #https://pdg.lbl.gov/2023/reviews/rpp2022-rev-ckm-matrix.pdf

c_H = -(4*GF/np.sqrt(2)) * (fine_struc_const/(4*np.pi)) * V_ts * V_tb
#################################################################################################################################################
#################################################################################################################################################
# Q2 DEPENDENT TERMS
#################################################################################################################################################
if not os.path.exists('PDF_helicities_expanded/plots'):
    os.makedirs('PDF_helicities_expanded/plots')

q = np.linspace(2 * mmu, mB - mK, 1000)
q2 = q**2
ctl = np.linspace(-np.pi, np.pi, 1000)

def kallen(a,b,c):
    return a**2 + b**2 + c**2 - 2*(a*b + b*c + c*a)

beta_ell_sqrd = 1 - ((4 * mmu2) / q2)
m_hat_sqrd = mmu2 / q2
kallen_BK = kallen(mB**2, mK**2, q2)
kallen_gamma = kallen(q2, mmu2, mmu2)

kappa_kin = (np.sqrt(kallen_BK) * np.sqrt(kallen_gamma)) / (2**6 * np.pi**3 * mB**3 * q2)
curly_N_q2 = np.abs(c_H)**2 * kappa_kin * q2 / 8

if not os.path.exists('PDF_helicities_expanded/plots/q2_dependent_components'):
    os.makedirs('PDF_helicities_expanded/plots/q2_dependent_components')

plt.plot(q, np.sqrt(beta_ell_sqrd))
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\beta_\ell(q^2)$')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/beta_ell.pdf')
plt.close()

plt.plot(q, beta_ell_sqrd)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\beta_\ell^2(q^2)$')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/beta_ell_sqrd.pdf')
plt.close()

plt.plot(q, np.sqrt(m_hat_sqrd))
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\hat{m}_\ell(q^2)$')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/mhat_ell.pdf')
plt.close()

plt.plot(q, m_hat_sqrd)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\hat{m}_\ell^2(q^2)$')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/mhat_ell_sqrd.pdf')
plt.close()

plt.plot(q, curly_N_q2)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$q^2\mathcal{N}(q^2)$ (MeV$^{-3}$)')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/curly_N_q2.pdf')
plt.close()

plt.plot(q, kallen_BK)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\lambda_{BK}$ (MeV$^4$)')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/kallen_BK.pdf')
plt.close()

plt.plot(q, kallen_gamma)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\lambda_{\gamma^\ast}$ (MeV$^4$)')
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/kallen_gamma.pdf')
plt.close()
#################################################################################################################################################
#################################################################################################################################################
# FORM FACTORS
#################################################################################################################################################
ff_0 = np.array([ff.make_f0_B(q2_/1000000).mean for q2_ in q2])
ff_p = np.array([ff.make_fp_B(q2_/1000000).mean for q2_ in q2])
ff_T = np.array([ff.make_fT_B(q2_/1000000).mean for q2_ in q2])

plt.plot(q, ff_0, label=r'$f_0(q^2)$', lw=1.5)
plt.plot(q, ff_p, label=r'$f_+(q^2)$', lw=1.5)
plt.plot(q, ff_T, label=r'$f_T(q^2)$', lw=1.5)
plt.legend()
plt.savefig(f'PDF_helicities_expanded/plots/q2_dependent_components/form_factors.pdf')
plt.close()
#################################################################################################################################################
#################################################################################################################################################
# EFFICIENCIES
#################################################################################################################################################
efficiencies_on = True
def f0_efficiency(q2):
    q = np.sqrt(q2)
    A = -2.11650513e-15
    B = 8.91827329e-12
    C = 1.90257993e-08
    D = -6.08647137e-05
    E = 2.92573411e-01
    return A*q**4 + B*q**3 + C*q**2 + D*q + E

def fp_efficiency(q2):
    q = np.sqrt(q2)
    A = -3.70852206e-15
    B = 2.71666358e-11
    C = -5.04080061e-08
    D = 2.39791458e-05
    E = 3.02150905e-01
    return A*q**4 + B*q**3 + C*q**2 + D*q + E

f_0_eff = np.sqrt(f0_efficiency(q2) if efficiencies_on else 1)
f_p_eff = np.sqrt(fp_efficiency(q2) if efficiencies_on else 1)
f_T_eff = np.sqrt(fp_efficiency(q2) if efficiencies_on else 1)

#################################################################################################################################################
#################################################################################################################################################
# MOMENTUM TERM BREAKDOWNS
#################################################################################################################################################
# hXCY - term in h^X before C_Y
hSCS = ((mB2 - mK2)/2) * (1/(mb - ms)) * ff_0 * f_0_eff
hPCP = ((mB2 - mK2)/2) * (1/(mb - ms)) * ff_0 * f_0_eff
hPC10 = ((mB2 - mK2)/2) * ((2*mmu)/q2) * ff_0 * f_0_eff
hVCV = (np.sqrt(kallen_BK)/(2*np.sqrt(q2))) * ff_p * f_p_eff
hVC7 = (np.sqrt(kallen_BK)/(2*np.sqrt(q2))) * ((2*mb)/(mB + mK)) * ff_T * f_T_eff
hAC10 = (np.sqrt(kallen_BK)/(2*np.sqrt(q2))) * ff_p * f_p_eff
hTtCT = -1j * ( np.sqrt(kallen_BK)/ (2*(mB+mK))) * ff_T
hTCT5 = -1j * ( np.sqrt(kallen_BK)/ (np.sqrt(2)*(mB+mK))) * ff_T

# GXhYhZ - term in G^2X before h^Yh^Z
G0hShS = 2 * beta_ell_sqrd
G0hPhP = 2
G0hVhV = (4/3) * (1 + 2*m_hat_sqrd)
G0hAhA = (4/3) * beta_ell_sqrd
G0hTthTt = (8/3) * (1 + 8*m_hat_sqrd)
G0hThT = (4/3) * beta_ell_sqrd
G0hVhTt = 16 * m_hat_sqrd
G1hVhS = -4 * np.sqrt(beta_ell_sqrd) * 2 * np.sqrt(m_hat_sqrd)
G1hTthS = 4 * beta_ell_sqrd * 2
G1hThP = 4 * beta_ell_sqrd * np.sqrt(2)
G2hVhV = -(4/3) * beta_ell_sqrd
G2hAhA = -(4/3) * beta_ell_sqrd
G2hThT = -(4/3) * beta_ell_sqrd * -2
G2hTthTt = -(4/3) * beta_ell_sqrd * -4

#################################################################################################################################################
# DX SEPARATED PREFACTORS
#################################################################################################################################################
if not os.path.exists('PDF_helicities_expanded/plots/prefactors'):
    os.makedirs('PDF_helicities_expanded/plots/prefactors')
if not os.path.exists('PDF_helicities_expanded/plots/prefactors/D_separated'):
    os.makedirs('PDF_helicities_expanded/plots/prefactors/D_separated')

differential_multiplier = 2 * np.sqrt(q2)
# D0
# Self-interactions
CSCS_D0_prefactor = curly_N_q2 * G0hShS * np.abs(hSCS)**2
CPCP_D0_prefactor = curly_N_q2 * G0hPhP * np.abs(hPCP)**2
C7C7_D0_prefactor = curly_N_q2 * G0hVhV * np.abs(hVC7)**2
C9C9_D0_prefactor = curly_N_q2 * G0hVhV * np.abs(hVCV)**2
C10C10_D0_prefactor_from_hPhP = curly_N_q2 * G0hPhP * np.abs(hPC10)**2
C10C10_D0_prefactor_from_hAhA = curly_N_q2 * G0hAhA * np.abs(hAC10)**2
C10C10_D0_prefactor = C10C10_D0_prefactor_from_hPhP + C10C10_D0_prefactor_from_hAhA
CTCT_D0_prefactor = curly_N_q2 * 2 * G0hTthTt * np.abs(hTtCT)**2
CT5CT5_D0_prefactor =curly_N_q2 * 2 * G0hThT * np.abs(hTCT5)**2
# Internal Interference
C7C9_D0_prefactor = curly_N_q2 * 2 * G0hVhV * np.real(hVC7 * np.conj(hVCV))
C10CP_D0_prefactor = curly_N_q2 * 2 * G0hPhP * np.real(hPCP * np.conj(hPC10))
# Exteral interference
C7CT_D0_prefactor = curly_N_q2 * G0hVhTt * np.imag(hVC7 * np.conj(hTtCT))
C9CT_D0_prefactor = curly_N_q2 * G0hVhTt * np.imag(hVCV * np.conj(hTtCT))

# D1 
# External interferences
C7CS_D1_prefactor = curly_N_q2 * G1hVhS * np.real(hVC7 * np.conj(hSCS))
C9CS_D1_prefactor = curly_N_q2 * G1hVhS * np.real(hVCV * np.conj(hSCS))
CTCS_D1_prefactor = curly_N_q2 * G1hTthS * np.imag(hTtCT * np.conj(hSCS))
CT5CP_D1_prefactor = curly_N_q2 * G1hThP * np.imag(hTCT5 * np.conj(hPCP))
CT5C10_D1_prefactor = curly_N_q2 * G1hThP * np.imag(hTCT5 * np.conj(hPC10))

# D2
# Self-interactions
C9C9_D2_prefactor = curly_N_q2 * G2hVhV * np.abs(hVCV)**2
C10C10_D2_prefactor = curly_N_q2 * G2hAhA * np.abs(hAC10)**2
C7C7_D2_prefactor = curly_N_q2 * G2hVhV * np.abs(hVC7)**2
CTCT_D2_prefactor = curly_N_q2 * G2hThT * np.abs(hTtCT)**2
CT5CT5_D2_prefactor = curly_N_q2 * G2hTthTt * np.abs(hTCT5)**2
# Internal interferences
C7C9_D2_prefactor = curly_N_q2 * 2 * G2hVhV * np.real(hVC7 * np.conj(hVCV))

D_separated_prefactors = {
    'D0_CSCS' : CSCS_D0_prefactor,
    'D0_CPCP' : CPCP_D0_prefactor,
    'D0_C7C7' : C7C7_D0_prefactor,
    'D0_C9C9' : C9C9_D0_prefactor,
    'D0_C10C10' : C10C10_D0_prefactor,
    'D0_CTCT' : CTCT_D0_prefactor,
    'D0_CT5CT5' : CT5CT5_D0_prefactor,
    'D0_C7C9' : C7C9_D0_prefactor,
    'D0_C10CP' : C10CP_D0_prefactor,
    'D0_C7CT' : C7CT_D0_prefactor,
    'D0_C9CT' : C9CT_D0_prefactor,

    'D1_C7CS' : C7CS_D1_prefactor,
    'D1_C9CS' : C9CS_D1_prefactor,
    'D1_CTCS' : CTCS_D1_prefactor,
    'D1_CT5CP' : CT5CP_D1_prefactor,
    'D1_CT5C10' : CT5C10_D1_prefactor,

    'D2_C9C9' : C9C9_D2_prefactor,
    'D2_C10C10' : C10C10_D2_prefactor,
    'D2_C7C7' : C7C7_D2_prefactor,
    'D2_CTCT' : CTCT_D2_prefactor,
    'D2_CT5CT5' : CT5CT5_D2_prefactor,
    'D2_C7C9' : C7C9_D2_prefactor,
}

TeX_dict = {
    'D0_CSCS' : r'$\mathbb{A}_{S,S}^0$ (MeV$^6$)',
    'D0_CPCP' : r'$\mathbb{A}_{P,P}^0$ (MeV$^6$)',
    'D0_C7C7' : r'$\mathbb{A}_{7,7}^0$ (MeV$^6$)',
    'D0_C9C9' : r'$\mathbb{A}_{V,V}^0$ (MeV$^6$)',
    'D0_C10C10' : r'$\mathbb{A}_{A,A}^0$ (MeV$^6$)',
    'D0_CTCT' : r'$\mathbb{A}_{T,T}^0$ (MeV$^6$)',
    'D0_CT5CT5' : r'$\mathbb{A}_{T5,T5}^0$ (MeV$^6$)',
    'D0_C7CT' : r'$\mathbb{A}_{T,7}^0$ (MeV$^6$)',
    'D0_C9CT' : r'$\mathbb{A}_{V,T}^0$ (MeV$^6$)',
    'D0_C7C9' : r'$\mathbb{A}_{7,V}^0$ (MeV$^6$)',
    'D0_C10CP' : r'$\mathbb{A}_{A,P}^0$ (MeV$^6$)',

    'D1_C7CS' : r'$\mathbb{A}_{7,S}^1$ (MeV$^6$)',
    'D1_C9CS' : r'$\mathbb{A}_{V,S}^1$ (MeV$^6$)',
    'D1_CTCS' : r'$\mathbb{A}_{T,S}^1$ (MeV$^6$)',
    'D1_CT5CP' : r'$\mathbb{A}_{T5,P}^1$ (MeV$^6$)',
    'D1_CT5C10' : r'$\mathbb{A}_{T5,A}^1$ (MeV$^6$)',

    'D2_C9C9' : r'$\mathbb{A}_{V,V}^2$ (MeV$^6$)',
    'D2_C10C10' : r'$\mathbb{A}_{A,A}^2$ (MeV$^6$)',
    'D2_C7C7' : r'$\mathbb{A}_{7,7}^2$ (MeV$^6$)',
    'D2_CTCT' : r'$\mathbb{A}_{T,T}^2$ (MeV$^6$)',
    'D2_CT5CT5' : r'$\mathbb{A}_{T5,T5}^2$ (MeV$^6$)',
    'D2_C7C9' : r'$\mathbb{A}_{7,V}^2$ (MeV$^6$)',
}

x_data = q
for key in D_separated_prefactors:
    y_data = D_separated_prefactors[key]
    plt.plot(x_data, y_data * differential_multiplier, lw=1.5)

    plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
    plt.ylabel(TeX_dict[key])
    plt.xlim(2*mmu, mB - mK)
    plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/{key}.pdf')
    plt.close()

# Separate plot for more detailed D0_C10C10
plt.plot(x_data, D_separated_prefactors['D0_C10C10'] * differential_multiplier, lw=1.5, label=r'total')
plt.plot(x_data, C10C10_D0_prefactor_from_hPhP * differential_multiplier, linestyle='dashed', lw=1, label=r'from $\vert C_{10}\vert^2$ in $\vert h^P\vert^2$')
plt.plot(x_data, C10C10_D0_prefactor_from_hAhA * differential_multiplier, linestyle='dashed', lw=1, label=r'from $\vert C_{10}\vert^2$ in $\vert h^A\vert^2$')
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.legend()
plt.ylabel(r'$\mathbb{A}_{A,A}^0$ (MeV$^6$)')
plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/D0_C10C10_breakdown.pdf')
plt.close()


# Plot particular ratios
if not os.path.exists('PDF_helicities_expanded/plots/prefactors/D_separated/ratios'):
    os.makedirs('PDF_helicities_expanded/plots/prefactors/D_separated/ratios')

q_CS_over_CP_ratio = []
q_C9_over_C10_ratio = []
CS_over_CP_ratio = []
C9_over_C10_ratio = []

for i, j, q_ in zip(CSCS_D0_prefactor, CPCP_D0_prefactor, q):
    if j != 0:
        CS_over_CP_ratio.append(i/j)
        q_CS_over_CP_ratio.append(q_)
plt.plot(q_CS_over_CP_ratio, CS_over_CP_ratio, lw=1.5)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\frac{\mathbb{A}^0_{S,S}(q^2)}{\mathbb{A}^0_{P,P}(q^2)}$')
plt.axhline(y=1,linestyle='dashed',color='grey',zorder=0,lw=1.5)
plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/ratios/CS2_prefactor_over_CP2_prefactor.pdf')
plt.close()

for i, j, q_ in zip(C9C9_D0_prefactor, C10C10_D0_prefactor, q):
    if j != 0:
        C9_over_C10_ratio.append(i/j)
        q_C9_over_C10_ratio.append(q_)
plt.plot(q_C9_over_C10_ratio, C9_over_C10_ratio, lw=1.5)
plt.xlabel(r'$m_{\mu\mu}$ (MeV)')
plt.ylabel(r'$\frac{\mathbb{A}^0_{V,V}(q^2)}{\mathbb{A}^D_{A,A}(q^2)}$')
plt.axhline(y=1,linestyle='dashed',color='grey',zorder=0,lw=1.5)
plt.savefig(f'PDF_helicities_expanded/plots/prefactors/D_separated/ratios/C92_prefactor_over_C102_prefactor.pdf')
plt.close()

#################################################################################################################################################
#################################################################################################################################################
# PLOT WHOLE PDF
#################################################################################################################################################
def formulate_PDF(C_7, C_S, C_P, C_9, C_10, C_T, C_T5):
    # Self-interactions
    CSCS_D0_contribution = curly_N_q2 * G0hShS * np.abs(hSCS * C_S)**2
    CPCP_D0_contribution = curly_N_q2 * G0hPhP * np.abs(hPCP * C_P)**2
    C7C7_D0_contribution = curly_N_q2 * G0hVhV * np.abs(hVC7 * C_7)**2
    C9C9_D0_contribution = curly_N_q2 * G0hVhV * np.abs(hVCV * C_9)**2
    C10C10_D0_contribution_from_hPhP = curly_N_q2 * G0hPhP * np.abs(hPC10 * C_10)**2
    C10C10_D0_contribution_from_hAhA = curly_N_q2 * G0hAhA * np.abs(hAC10 * C_10)**2
    C10C10_D0_contribution = C10C10_D0_contribution_from_hPhP + C10C10_D0_contribution_from_hAhA
    CTCT_D0_contribution = curly_N_q2 * 2 * G0hTthTt * np.abs(hTtCT * C_T)**2
    CT5CT5_D0_contribution = curly_N_q2 * 2 * G0hThT * np.abs(hTCT5 * C_T5)**2
    # Internal Interference
    C7C9_D0_contribution = curly_N_q2 * 2 * G0hVhV * np.real(hVC7 * C_7 * np.conj(hVCV * C_9))
    C10CP_D0_contribution = curly_N_q2 * 2 * G0hPhP * np.real(hPCP * C_P * np.conj(hPC10 * C_10))
    # Exteral interference
    C7CT_D0_contribution = curly_N_q2 * G0hVhTt * np.imag(hVC7 * C_7 * np.conj(hTtCT * C_T))
    C9CT_D0_contribution = curly_N_q2 * G0hVhTt * np.imag(hVCV * C_9 * np.conj(hTtCT * C_T))

    # D1 
    # External interferences
    C7CS_D1_contribution = curly_N_q2 * G1hVhS * np.real(hVC7 * C_7 * np.conj(hSCS * C_S))
    C9CS_D1_contribution = curly_N_q2 * G1hVhS * np.real(hVCV * C_9 * np.conj(hSCS * C_S))
    CTCS_D1_contribution = curly_N_q2 * G1hTthS * np.imag(hTtCT * C_T * np.conj(hSCS * C_S))
    CT5CP_D1_contribution = curly_N_q2 * G1hThP * np.imag(hTCT5 * C_T5 * np.conj(hPCP * C_P))
    CT5C10_D1_contribution = curly_N_q2 * G1hThP * np.imag(hTCT5 * C_T5 * np.conj(hPC10 * C_10))

    # D2
    # Self-interactions 
    C9C9_D2_contribution = curly_N_q2 * G2hVhV * np.abs(hVCV * C_9)**2
    C10C10_D2_contribution = curly_N_q2 * G2hAhA * np.abs(hAC10 * C_10)**2
    C7C7_D2_contribution = curly_N_q2 * G2hVhV * np.abs(hVC7 * C_7)**2
    CTCT_D2_contribution = curly_N_q2 * G2hThT * np.abs(hTtCT * C_T)**2
    CT5CT5_D2_contribution = curly_N_q2 * G2hTthTt * np.abs(hTCT5 * C_T5)**2
    # Internal interferences
    C7C9_D2_contribution = curly_N_q2 * 2 * G2hVhV * np.real(hVC7 * C_7 * np.conj(hVCV * C_9))

    dGamma_dq2 = 2 * q * (C10C10_D0_contribution + C9C9_D0_contribution + C7C7_D0_contribution + CSCS_D0_contribution + CPCP_D0_contribution + CTCT_D0_contribution + CT5CT5_D0_contribution + C7C9_D0_contribution + C10CP_D0_contribution + C7CT_D0_contribution + C9CT_D0_contribution)
    dGamma_dq2dctl = 0

    return dGamma_dq2, dGamma_dq2dctl
#################################################################################################################################################

original_PDF = formulate_PDF(C_7=C_7, C_S=C_S, C_P=C_P, C_9=C_9, C_10=C_10, C_T=C_T, C_T5=C_T5)[0]
proposed_PDF_CP = formulate_PDF(C_7=C_7, C_S=C_S, C_P=C_P-0.5, C_9=C_9, C_10=C_10, C_T=C_T, C_T5=C_T5)[0]
proposed_PDF_CV = formulate_PDF(C_7=C_7, C_S=C_S, C_P=C_P, C_9=C_9-0.5, C_10=C_10, C_T=C_T, C_T5=C_T5)[0]
proposed_PDF_CS = formulate_PDF(C_7=C_7, C_S=C_S-0.5, C_P=C_P, C_9=C_9, C_10=C_10, C_T=C_T, C_T5=C_T5)[0]
proposed_PDF_CA = formulate_PDF(C_7=C_7, C_S=C_S, C_P=C_P, C_9=C_9, C_10=C_10-0.5, C_T=C_T, C_T5=C_T5)[0]


plt.plot(q, original_PDF, lw=1.5, label=r'SM $C_X$ set')
plt.plot(q, proposed_PDF_CP, lw=1.5, label=r'$C_P \rightarrow C_P - 0.5$')
plt.plot(q, proposed_PDF_CV, lw=1.5, label=r'$C_9 \rightarrow C_9 - 0.5$')
plt.plot(q, proposed_PDF_CS, lw=1.5, label=r'$C_9 \rightarrow C_S - 0.5$')
plt.plot(q, proposed_PDF_CA, lw=1.5, label=r'$C_{10} \rightarrow C_{10} - 0.5$')

plt.legend()
plt.savefig('PDF_helicities_expanded/plots/COMPARISON.pdf')
plt.show()

