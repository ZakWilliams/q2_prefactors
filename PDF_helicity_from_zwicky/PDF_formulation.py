import numpy as np
import make_BK_DK_ffs as ff

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
# INITIAL FUNCTION DEFINITIONS
#################################################################################################################################################
def m_hat(q2):
    return mmu/np.sqrt(q2)

def m_hat_sqrd(q2):
    return mmu2/q2

def beta_ell(q2):
    return np.sqrt(1 - ((4 * mmu2) / q2))

def beta_ell_sqrd(q2):
    return 1 - ((4 * mmu2) / q2)

def kallen(a,b,c):
    return a**2 + b**2 + c**2 - 2*(a*b + b*c + c*a)

def kallen_BK(q2):
    return kallen(mB**2, mK**2, q2)

def kallen_gamma(q2):
    return kallen(q2, mmu2, mmu2)

def kappa_kin(q2):
    numerator = np.sqrt(kallen_BK(q2)) * np.sqrt(kallen_gamma(q2))
    denominator = 2**6 * np.pi**3 * mB**3 * q2
    return numerator / denominator

def curly_N(q2):
    return np.abs(c_H)**2 * kappa_kin(q2)

def common_prefactor(q2):
    return 0.125 * curly_N(q2) * q2

#################################################################################################################################################
# EFFICIENCY TERMS
#################################################################################################################################################

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

#################################################################################################################################################
# FORM FACTORS
#################################################################################################################################################

def f_0(q2):
    return np.array([ff.make_f0_B(q2_/1000000).mean + 0j for q2_ in q2])
def f_p(q2):
    return np.array([ff.make_fp_B(q2_/1000000).mean + 0j for q2_ in q2])
def f_T(q2):
    return np.array([ff.make_fT_B(q2_/1000000).mean + 0j for q2_ in q2])

#################################################################################################################################################
# HELICITY AMPLITUDES
#################################################################################################################################################

# Note: a better one would cover tensorial efficiencies too, or would rather tackle the effficiency issue in the whole 2D parameter space seperately

efficiencies_on = True

def hS(q2, C_S):
    prefactor = ((mB2 - mK2)/2) + 0j
    C_S_prefactor = 1 / (mb - ms)
    f_0_eff = np.sqrt(f0_efficiency(q2) if efficiencies_on else 1)
    return prefactor * (C_S_prefactor * C_S) * f_0(q2) * f_0_eff

def hP(q2, C_P, C_A):
    prefactor = ((mB2 - mK2)/2) + 0j
    C_P_prefactor = 1 / (mb - ms)
    C_A_prefactor = 2*mmu / q2
    f_0_eff = np.sqrt(f0_efficiency(q2) if efficiencies_on else 1)
    return prefactor * (C_P_prefactor * C_P + C_A_prefactor * C_A) * f_0(q2) * f_0_eff

def hV(q2, C_7, C_V):
    prefactor = np.sqrt(kallen_BK(q2))/(2 * np.sqrt(q2)) + 0j
    C_7_prefactor = (2 * mb) / (mB + mK)
    C_V_prefactor = 1
    f_p_eff = np.sqrt(fp_efficiency(q2)) if efficiencies_on else 1
    return prefactor * ((C_7_prefactor * C_7 * f_T(q2)) + (C_V_prefactor * C_V * f_p(q2))) * f_p_eff

def hA(q2, C_A):
    prefactor = np.sqrt(kallen_BK(q2))/(2 * np.sqrt(q2)) + 0j
    C_A_prefactor = 1
    f_p_eff = np.sqrt(fp_efficiency(q2)) if efficiencies_on else 1
    return prefactor * (C_A_prefactor * C_A) * f_p(q2) * f_p_eff

def hT(q2, C_T5):
    prefactor = 0 + 1j*((-1 * np.sqrt(kallen_BK(q2))) / (np.sqrt(2) * (mB + mK)))
    f_p_eff = np.sqrt(fp_efficiency(q2)) if efficiencies_on else 1
    return prefactor * C_T5 * f_T(q2) * f_p_eff

def hTt(q2, C_T):
    prefactor = 0 + 1j*((-1 * np.sqrt(kallen_BK(q2))) / (2 * (mB + mK)))
    f_p_eff = np.sqrt(fp_efficiency(q2)) if efficiencies_on else 1
    return prefactor * C_T * f_T(q2) * f_p_eff

#################################################################################################################################################
# G TERMS
#################################################################################################################################################

def G0(q2, C_7, C_S, C_P, C_V, C_A, C_T, C_T5):
    hVhV_term = (4/3) * (1 + 2*m_hat_sqrd(q2)) * np.abs(hV(q2, C_7, C_V))**2
    hAhA_term = (4/3) * beta_ell_sqrd(q2) * np.abs(hA(q2, C_A))**2
    hShS_term = 2 * beta_ell_sqrd(2) * np.abs(hS(q2, C_S))**2
    hPhP_term = 2 * np.abs(hP(q2, C_P, C_A))**2
    hTthTt_term = (8/3) * (1 + 8*m_hat_sqrd(q2))* np.abs(hTt(q2, C_T))**2
    hThT_term = (4/3) * (beta_ell_sqrd(q2)) * np.abs(hT(q2, C_T5))**2
    hVhTt_term = 16 * m_hat(q2) * np.imag(hV(q2, C_7, C_V) * np.conjugate(hTt(q2, C_T)))
    return hVhV_term + hAhA_term + hShS_term + hPhP_term + hTthTt_term + hThT_term + hVhTt_term

def G1(q2, C_7, C_S, C_P, C_V, C_A, C_T, C_T5):
    prefactor = -4 * beta_ell_sqrd(q2)
    hVhS_term = 2 * m_hat(q2) * np.real(hV(q2, C_7, C_V) * np.conjugate(hS(q2, C_S)))
    hTthS_term = -2 * np.imag(hTt(q2, C_T) * np.conjugate(hS(q2, C_S)))
    hThP_term = -np.sqrt(2) * np.imag(hT(q2, C_T5) * np.conjugate(hP(q2, C_P, C_A)))
    return prefactor * (hVhS_term + hTthS_term + hThP_term)

def G2(q2, C_7, C_S, C_P, C_V, C_A, C_T, C_T5):
    prefactor = -4 * beta_ell_sqrd(q2) / 3
    hVhV_term = np.abs(hV(q2, C_7, C_V))**2
    hAhA_term = np.abs(hA(q2, C_A))**2
    hThT_term = -2 * np.abs(hT(q2, C_T5))**2
    hTthTt_term = -4 * np.abs(hTt(q2, C_T))**2
    return prefactor * (hVhV_term + hAhA_term + hThT_term + hTthTt_term)

#################################################################################################################################################
# D TERMS
#################################################################################################################################################

# 2D profile
#def dGamma_dq2dctl():

def dGamma_dq(q2, ctl, C_7, C_S, C_P, C_V, C_A, C_T, C_T5):
    return common_prefactor(q2) * G0(q2, C_7, C_S, C_P, C_V, C_A, C_T, C_T5) * 2 * np.sqrt(q2) 