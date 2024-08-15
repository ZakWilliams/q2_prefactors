import numpy as np
from PDF_formulation import mmu, mB, mK, mb, ms, mmu2, mB2, mK2

q02 = -1.5E6

# parameters for peaks
parameters = {
    'rho_770': {
        'mass': 775.3,
        'width': 149.1,
        'phase': -0.35,
        'mag': 0.6832186538804599
    },
    'omega_782': {
        'mass': 782.7,
        'width': 8.5,
        'phase': 0.26,
        'mag': 4.662989285716814
    },
    'phi_1020': {
        'mass': 1019.4,
        'width': 4.3,
        'phase': 0.47,
        'mag': 12.948131275558348
    },
    'Jpsi': {
        'mass': 3096.65,
        'width': 0.09,
        'phase': -1.66,
        'mag': 8345.20
    },
    'psi2S': {
        'mass': 3685.90,
        'width': 0.3,
        'phase': -1.93,
        'mag': 1287.8248307668339
    },
    'psi_3770': {
        'mass': 3773,
        'width': 27.2,
        'phase': -2.13,
        'mag': 2.9692335397339633
    },
    'psi_4040': {
        'mass': 4039,
        'width': 80,
        'phase': -2.52,
        'mag': 1.8754608465591804
    },
    'psi_4160': {
        'mass': 4196,
        'width': 72,
        'phase': -1.90,
        'mag': 2.7653967732679225
    },
    'psi_4415': {
        'mass': 4420,
        'width': 62,
        'phase': -2.52,
        'mag': 2.8462873781
    }
}

# functions for peaks
def kallen(A,B,C):
    return np.square(A) + np.square(B) + np.square(C) - 2*(A*B + B*C + C*A)

def p(q2):
    return np.sqrt(kallen(q2,mmu2,mmu2)) / np.sqrt(4 * q2)

def running_width(q2, mR, wR):
    frac1 = (p(q2) / p(np.square(mR)))
    frac2 = mR / np.sqrt(q2)
    return frac1 * frac2 * wR

def beta2(q2):
    return 1. - 4.*mmu2/q2

def Y(q2, eR, pR, mR, wR):
    #q2 = np.square(m_mumu)
    mR2 = np.square(mR)

    num1 = q2 - q02
    den1 = mR2 - q02
    num2 = mR * wR
    den2_re = mR2 - q2
    den2_im = -1. * mR * running_width(q2, mR, wR)
    den2 = den2_re + den2_im*1j

    term1 = eR
    term2 = np.cos(pR) + np.sin(pR)*1j
    term3 = num1 / den1
    term4 = num2 / den2

    return term1 * term2 * term3 * term4


# double crystal ball parameters
DCB_params = [
    {'sigma_dcb' : 2.81, 'sigma_gauss' : 4.71, 'alpha' : 1.44, 'n_L' : 1.99, 'n_R' : 6.79, 'f_dcb' : 0.52},
    {'sigma_dcb' : 5.23, 'sigma_gauss' : 5.97, 'alpha' : 1.16, 'n_L' : 30.23, 'n_R' : 31.61, 'f_dcb' : 0.54}, #31.61
    {'sigma_dcb' : 4.34, 'sigma_gauss' : 5.15, 'alpha' : 1.14, 'n_L' : 11.28, 'n_R' : 11.37, 'f_dcb' : 0.53},
]


# Double-sided Crystal Ball function with separately defined tails and Gaussian core
def crystal_ball(x, alpha_left, n_left, alpha_right, n_right, sigma, mu):
    # Standardize the x value
    t = (x - mu) / sigma
    
    # Gaussian core
    gaussian = np.exp(-0.5 * t**2)
    
    # Left-side tail
    A_left = (n_left / abs(alpha_left))**n_left * np.exp(-0.5 * alpha_left**2)
    B_left = n_left / abs(alpha_left) - abs(alpha_left)
    left = A_left / (B_left - t)**n_left
    
    # Right-side tail
    A_right = (n_right / abs(alpha_right))**n_right * np.exp(-0.5 * alpha_right**2)
    B_right = n_right / abs(alpha_right) - abs(alpha_right)
    right = A_right / (B_right + t)**n_right
    
    # Combine the Gaussian core with the tails
    result = np.where(t < -abs(alpha_left), left, 
                      np.where(t > abs(alpha_right), right, gaussian))
    
    return result

def gaussian(x, mu, sigma):
    t = (x - mu) / sigma
    N = 1/np.sqrt(2 * np.pi *sigma**2)

    result = N * np.exp(-1*t**2)

    return result

def res_model(x, sigma_dcb, sigma_gauss, alpha, n_L, n_R, f_dcb):
    dcb_component = crystal_ball(x=x, alpha_left=alpha, alpha_right=alpha, n_left=n_L, n_right=n_R, sigma=sigma_dcb, mu=0)
    gaussian_component = gaussian(x=x, mu=0, sigma=sigma_gauss)

    resolution = f_dcb * dcb_component + (1-f_dcb) * gaussian_component

    return resolution