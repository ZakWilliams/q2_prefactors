import numpy as np
import matplotlib.pyplot as plt
import PDF_helicity_from_zwicky.make_BK_DK_ffs as ff
import os
import shutil
import PDF_helicity_from_zwicky.PDF_formulation as PDF
#import cmath

import mplhep as hep

hep.style.use("LHCb2")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

#################################################################################################################################################
# WILSON PARAMETERS
#################################################################################################################################################
savename = 'PDF_helicity_from_zwicky/PDF_C7_unitary.pdf'

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
# PHASESPACE 
#################################################################################################################################################

q = np.linspace(2 * mmu, mB - mK, 1000)
q2 = q**2
ctl = np.linspace(-np.pi, np.pi, 1000)

#################################################################################################################################################
# CREATE PROPER WILSON PARAMETERS
#################################################################################################################################################
C_S = C_S_mag * (np.cos(C_S_phase) + 1j*np.sin(C_S_phase))
C_P = C_P_mag * (np.cos(C_P_phase) + 1j*np.sin(C_P_phase))
C_9 = C_9_mag * (np.cos(C_9_phase) + 1j*np.sin(C_9_phase))
C_10 = C_10_mag * (np.cos(C_10_phase) + 1j*np.sin(C_10_phase))
C_T = C_T_mag * (np.cos(C_T_phase) + 1j*np.sin(C_T_phase))
C_T5 = C_T5_mag * (np.cos(C_T5_phase) + 1j*np.sin(C_T5_phase))
C_7 = C_7_mag * (np.cos(C_7_phase) + 1j*np.sin(C_7_phase))

#
# GENERATE PDF
#

q2_dependant_PDF = PDF.dGamma_dq(q2=q2, ctl=ctl, C_7=C_7, C_S=C_S, C_P=C_P, C_V=C_9, C_A=C_10, C_T=C_T, C_T5=C_T5)

plt.plot(q, q2_dependant_PDF)
plt.savefig(savename)