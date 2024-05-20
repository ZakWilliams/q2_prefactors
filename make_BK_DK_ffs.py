introstring ='''This script reads data from BtoKandDtoKformfacs.txt and uses it to reproduce correlated B to K and D to K form factor results at any given q^2. 
Usage: put this script in the same directory as BtoKandDtoKformfacs.txt and run `python3 make_BK_DK_ffs.py' to check test outputs. These numbers should agree with the results given after the `c.f.'. This confirms that the form factors have been loaded properly and correlations preserved. 

After performing this check, one can load this script as a module and call any of the functions `make_fX_Y(qsq)', where `X'=`0',`p',`T' for f_0, f_+ and f_T and `Y'=`B',`D' for B->K or D->K. These functions will take any float or gvar value of q^2 as a input, but our results aren't valid outside of the physical q^2 range of the respective decays. 

W. G. Parrott 2021'''


import numpy as np
import gvar as gv

############################################# Read numbers and correlations from text file ###########################

f = open('BtoKandDtoKformfacs.txt','r')
lines = f.readlines()
toggle = False
for line in lines:
    if toggle == False:
        if len(line.split()) == 0:
            pass
        elif line.split()[0] == 'N':
            N = int(line.split()[2]) 
        elif line.split()[0] == 'A':
            A = line.split('A = ')[1]
        elif line.split()[0] == 'Checks':
            Checks = line.split('Checks = ')[1]
        elif line.split()[0] == 'Cov':
            Cov = line.split('Cov = ')[1]
            toggle = True
    else:
        Cov += line
f.close()
List = []
Test = []
for element in A[1:-2].split(', '):
    List.append(float(element))   # Make a list of means
for element in Checks[1:-2].split(', '):
    Test.append(gv.gvar(element))
length = len(List)
M = np.zeros((length,length))
Cov = Cov.replace('[',' ')
Cov = Cov.replace(']',' ')
for i in range(length):
    for j in range(length):
        M[i][j] = Cov.split()[i*(length)+j] # Make the correlation matrix
        
############################################## Correlate data and take out parameters we need##########################

List = gv.gvar(List,M) # makes correlated gvars from the means and covarience matrix

MK = List[0]
MB = List[1]
MBs0 = List[2]
MBsstar = List[3]
logsB = List[4]
a0B = []
apB = []
aTB = []
for n in range(N):
    a0B.append(List[5+n])
    apB.append(List[5+N+n])
    aTB.append(List[5+2*N+n])
N_D = 5+3*N
MD = List[N_D]
MDs0 = List[N_D+1]
MDsstar = List[N_D+2]
logsD = List[N_D+3]
a0D = []
apD = []
aTD = []
for n in range(N):
    a0D.append(List[N_D+4+n])
    apD.append(List[N_D+4+N+n])
    aTD.append(List[N_D+4+2*N+n])
qsq_max_B = List[-2]
qsq_max_D = List[-1]


######################################### Make form factors ####################################################

def make_z_B(qsq):
    if qsq < 0:
        print('WARNING! Using a -ve q^2 value')
    if qsq > qsq_max_B.mean:
        print('WARNING! Using a q^2 value which exceeds the physical maximum of {0}'.format(qsq_max_B))
    t_plus = (MB+MK)**2
    t_0 = 0
    z = (gv.sqrt(t_plus - qsq) - gv.sqrt(t_plus - t_0)) / (gv.sqrt(t_plus - qsq) + gv.sqrt(t_plus - t_0))
    if qsq == 0:
        z = 0
    return(z)

def make_z_D(qsq):
    if qsq < 0:
        print('WARNING! Using a -ve q^2 value')
    if qsq > qsq_max_D.mean:
        print('WARNING! Using a q^2 value which exceeds the physical maximum of {0}'.format(qsq_max_D))
    t_plus = (MD+MK)**2
    t_0 = 0
    z = (gv.sqrt(t_plus - qsq) - gv.sqrt(t_plus - t_0)) / (gv.sqrt(t_plus - qsq) + gv.sqrt(t_plus - t_0))
    if qsq == 0:
        z = 0
    return(z)

def make_f0_B(qsq):
    pole = 1/(1-qsq/MBs0**2)
    f0 = 0
    z = make_z_B(qsq)
    for n in range(N):
        f0 += a0B[n] * z**n
    f0 *= pole * logsB
    return(f0)

def make_fp_B(qsq):
    pole = 1/(1-qsq/MBsstar**2)
    fp = 0
    z = make_z_B(qsq)
    for n in range(N):
        fp += apB[n] * (z**n - (n/N) * (-1)**(n-N) * z**N)
    fp *= pole * logsB
    return(fp)

def make_fT_B(qsq):
    pole = 1/(1-qsq/MBsstar**2)
    fT = 0
    z = make_z_B(qsq)
    for n in range(N):
        fT += aTB[n] * (z**n - (n/N) * (-1)**(n-N) * z**N)
    fT *= pole * logsB
    return(fT)


def make_f0_D(qsq):
    pole = 1/(1-qsq/MDs0**2)
    f0 = 0
    z = make_z_D(qsq)
    for n in range(N):
        f0 += a0D[n] * z**n
    f0 *= pole * logsD
    return(f0)

def make_fp_D(qsq):
    pole = 1/(1-qsq/MDsstar**2)
    fp = 0
    z = make_z_D(qsq)
    for n in range(N):
        fp += apD[n] * (z**n - (n/N) * (-1)**(n-N) * z**N)
    fp *= pole * logsD
    return(fp)

def make_fT_D(qsq):
    pole = 1/(1-qsq/MDsstar**2)
    fT = 0
    z = make_z_D(qsq)
    for n in range(N):
        fT += aTD[n] * (z**n - (n/N) * (-1)**(n-N) * z**N)
    fT *= pole * logsD
    return(fT)

########################################## Tests ###################################################
# Tests of the form factors. The ratios f_+(0)/f_0(0) should be 1 (possibly with a very tiny uncertainty). Other results are compared with the results from our paper which are loaded from the .txt file in 'Checks'. A non trivial combination of form factors is included to check correlations.
'''def do_tests():
    print(introstring)
    print('')
    print('Examples to check that load has worked correctly. Numbers should agree with the (c.f.) reference values. The ratios f_+(0)/f_0(0) should be 1(0) but may appear with a very small non zero uncertainty:')
    print('#################################### B #############################')
    print('f_+(0)/f_0(0) =  ',make_fp_B(0)/make_f0_B(0))
    print('f_0(0) = {0} c.f. {1}'.format(make_f0_B(0),Test[0]))
    print('f_+(0) = {0} c.f. {1}'.format(make_fp_B(0),Test[1]))
    print('f_T(0) = {0} c.f. {1}'.format(make_fT_B(0),Test[2]))
    print('f_0(qsq_max) = {0} c.f. {1}'.format(make_f0_B(qsq_max_B),Test[3]))
    print('f_+(qsq_max) = {0} c.f. {1}'.format(make_fp_B(qsq_max_B),Test[4]))
    print('f_T(qsq_max) = {0} c.f. {1}'.format(make_fT_B(qsq_max_B),Test[5]))
    print('#################################### D ############################')
    print('f_+(0)/f_0(0) = ',make_fp_D(0)/make_f0_D(0))
    print('f_0(0) = {0} c.f. {1}'.format(make_f0_D(0),Test[6]))
    print('f_+(0) = {0} c.f. {1}'.format(make_fp_D(0),Test[7]))
    print('f_T(0) = {0} c.f. {1}'.format(make_fT_D(0),Test[8]))
    print('f_0(qsq_max) = {0} c.f. {1}'.format(make_f0_D(qsq_max_D),Test[9]))
    print('f_+(qsq_max) = {0} c.f. {1}'.format(make_fp_D(qsq_max_D),Test[10]))
    print('f_T(qsq_max) = {0} c.f. {1}'.format(make_fT_D(qsq_max_D),Test[11]))
    print('############## Random Combination to check correlations ###########')
    print('(f_0(1)^B-f_+(0)^D)/f_T(0.5)^D = {0} c.f. {1}'.format((make_f0_B(1)-make_fp_D(0))/make_fT_D(0.5),Test[12]))
    return()

do_tests()'''
#######################################################################################################
