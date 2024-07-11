# Hydrate Composition Calculation Algorithm without Interface

# import math, scipy
from scipy.integrate import quad
from math import exp, pi, log, sqrt, cos
import numpy as np
from scipy.special import cbrt

### Definition of constants

K_BOLTZMANN = 1.38 * 10**(-23)
R_IDEAL_GAS = 8.314
T_ZERO = 273.15
P_ZERO = 0

MAX_DIFFERENCE = 10E-5

# Inside structure dictionary
KEY_FOR_MU0 = 'deltaMu0'
KEY_FOR_H0 = 'deltaH0'
KEY_FOR_CP0 = 'deltaCp0'
KEY_FOR_B = 'b'
KEY_FOR_V0 = 'deltaV0'
KEY_FOR_CAVITIES = 'cavities'       # list (2 items) of lists containing cavities caracteristics

# Inside 'cavities' value (list of lists) of structure dictionnary
INDEX_FOR_NU = 0            # index pointing to nu (number of this type of cavity) in the cavity info table in the structure info table
INDEX_FOR_R = 1             # index pointing to R (cavity ray) in the cavity info table in the structure info table
INDEX_FOR_Z = 2             # index pointing to z (coordination number) in the cavity info table

# Inside component dictionnary in components list (components is a list of dictionnaries)
KEY_FOR_ID = 'id'
KEY_FOR_EPSILON = 'epsilon'     # key pointing to epsilon in the component info table
KEY_FOR_SIGMA = 'sigma'         # key pointing to sigma in the component info table
KEY_FOR_A = 'a'                 # key pointing to a in the component info table
KEY_FOR_VINF = 'Vinf'
KEY_FOR_K1 = 'k1'
KEY_FOR_K2 = 'k2'
KEY_FOR_Y = 'y'
KEY_FOR_OMEGA = 'omega'
KEY_FOR_TC = 'Tc'
KEY_FOR_PC = 'Pc'

# Note: for greater readibility, indexes will be used as follows:
# i for cavity types
# j for components

### Initialization

## File imports


## Determination of the physical values

#



### Equilibrium condition: solving deltaMu_L = deltaMu_H

## Calculation of deltaMu_L using classical thermodynamcis

# deltah

def deltah_L(Temp, structure: dict):
    deltaH0 = structure[KEY_FOR_H0]
    deltaCp0 = structure[KEY_FOR_CP0]
    b =  structure[KEY_FOR_B]

    # definition of deltaCp
    def deltaCp(T):
        return deltaCp0 + b*(T - T_ZERO)
    integral = quad(deltaCp, T_ZERO, Temp)[0]

    return deltaH0 + integral

# non ideality term ln(a_w)

def a_w(T, P, components: list, coefficients: list):
    sumj = 0
    for j in range(len(components)):
        Vinf = components[j][KEY_FOR_VINF]
        K1 = components[j][KEY_FOR_K1]
        K2 = components[j][KEY_FOR_K2]
        print(K1, K2)
        H = exp(K1 + K2/T)
        fj = fugacity_j(T, P, components, coefficients, j)
        sumj += fj / (H * exp( P*Vinf / (R_IDEAL_GAS*T)))
    return 1 - sumj


# Injecting in deltaMu_L

def deltaMu_L(T, P , structure: dict, components: list, coefficients: list):
    deltaMu0 = structure[KEY_FOR_MU0]
    deltah = deltah_L(T, structure)
    deltaV0 = structure[KEY_FOR_V0]
    aw = a_w(T, P, components, coefficients)

    termMu = T * deltaMu0 / T_ZERO

    def integrandH(Temp):
        return deltah / (Temp**2)
    termH = T * quad(integrandH, T_ZERO, T)[0]

    termV = deltaV0 * (P - P_ZERO)

    termAw = R_IDEAL_GAS * T * log(aw)

    return termMu - termH + termV - termAw


## Calculation of deltaMu_H using statistical thermodynamics

# Cij: Langmuir constant for component j in cavity i

def langmuir_ij(T, structure_cavities: list, components: list, i: int, j: int) -> float:

    # determination of R, z, and Kihara parameters for a component j in a type i cavity
    R = structure_cavities[i][INDEX_FOR_R]
    z = structure_cavities[i][INDEX_FOR_Z]
    epsilon = components[j][KEY_FOR_EPSILON]
    sigma = components[j][KEY_FOR_SIGMA]
    a = components[j][KEY_FOR_A]

    # definition of the function inside the integral
    def integrand(r):

        # definition of w(r)
        def w_r(r):
            # definition of delta_N
            def delta_N(N):
                return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

            return ( 2 * z * epsilon
                    * ( (sigma**12 / (r * R**11)) * (delta_N(10) + delta_N(11) * a/R )
                    -   (sigma** 6 / (r * R** 5)) * (delta_N(4) + delta_N(5) * a/R ) )
            )

        # print('w : ', w_r(r))
        return np.exp( - w_r(r) / (K_BOLTZMANN * T)) * r**2
    integral = quad(integrand, 0, R)[0]
    # print('0.1 :', integrand(0.1), 'R :', integrand(1), R)

    # print(integral)
    return 4 * pi * 1  / (K_BOLTZMANN * T)
    # return 4 * pi * integral  / (K_BOLTZMANN * T)

# a_m

def a_j(T, components: list, j: int):
    Pc = components[j][KEY_FOR_PC]
    Tc = components[j][KEY_FOR_TC]
    omegaj = components[j][KEY_FOR_OMEGA]
    m = 0.48 + 1.1574 * omegaj - 0.176 * (omegaj**2)
    alphaj = ( 1 + m * (1- sqrt(T/Tc)) ) ** 2
    return 0.42747 * (R_IDEAL_GAS**2) * (Tc**2) * sqrt(T) * alphaj / Pc

def a_jk(T, components: list, coefficients: list, j: int, k: int):
    aj = a_j(T, components, j)
    ak = a_j(T, components, k)
    idj = components[j][KEY_FOR_ID]
    idk = components[k][KEY_FOR_ID]
    kjk = coefficients[idj][idk]
    return sqrt(aj * ak) * (1 - kjk)

def a_m(T, components: list, coefficients: list):
    sumj = 0
    for j in range(len(components)):
        yj = components[j][KEY_FOR_Y]
        sumk = 0
        for k in range(len(components)):
            yk = components[k][KEY_FOR_Y]
            ajk = a_jk(T, components, coefficients, j, k)
            sumk += yk * ajk
        sumj += yj * sumk
    return sumj

# b_m

def b_j(components: list, j: int):
    Tc = components[j][KEY_FOR_TC]
    Pc = components[j][KEY_FOR_PC]
    return 0.0866 * R_IDEAL_GAS * Tc / Pc

def b_m(components: list):
    sumj = 0
    for j in range(len(components)):
        yj = components[j][KEY_FOR_Y]
        bj = b_j(components, j)
        sumj += yj * bj
    return sumj

# phij

def lnphi_j(T, Pres, components: list, coefficients: list, j: int):
    bj = b_j(components, j)
    bm = b_m(components)
    aj = a_j(T, components, j)
    am = a_m(T, components, coefficients)

    # from equation of state, Z = PV/RT is solution of the cubic equation
    # Z**3 - Z**2 + (A - B**2 - B)*Z - AB = 0 , where
    A = am * Pres / ((R_IDEAL_GAS**2) * (T**2))
    B = bm * Pres / (R_IDEAL_GAS * T)

    # for better readibility, let's define
    p = -1
    q = (A - B**2 - B)
    r = A * B
    m = q - (p**2) / 3
    n = r + (2 * p**3 - 9*p*q) / 27
    delt = (n**2)/4 + (m**3) / 27

    if delt > 0:
        Zj = (- p / 3
             + cbrt(sqrt(delt) - n/2)
             + cbrt(- sqrt(delt) - n/2)
        )

    elif delt < 0:
        angle = - (n / abs(n)) * sqrt( (- 27 * n**2) / (4 * m**3) )
        Zj = ( - p / 3
            + 2 * sqrt(-m / 3) * cos(angle/3)
        )
    else:
        print('Error: no known value of Z for delta = 0')
        # TODO raise error here

    # print(Zj, B, B/Zj)
    # print(( (bj / bm) * (Zj - 1)
    #         - log(abs(Zj - B))
    #         - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(abs(1 + B/Zj))
    # ))
    return ( (bj / bm) * (Zj - 1)
            - log(abs(Zj - B))
            - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(abs(1 + B/Zj))
    )


# fj

def fugacity_j(T, P, components: list, coefficients, j: int):
    # phij = exp(lnphi_j(T, P, components, coefficients, j))
    phij = exp(-lnphi_j(T, P, components, coefficients, j))
    yj = components[j][KEY_FOR_Y]
    return phij * yj * P


# Injecting in deltaMu_H

def deltaMu_H(T, P, structure_cavities: list, components: list, coefficients: list):
    sumi = 0
    for i in range(len(structure_cavities)):
        sumj = 0
        for j in range(len(components)):
            cij = langmuir_ij(T, structure_cavities, components, i, j)

            # TODO ARGUMENTS TO BE MODIFIED ONCE DEF IS UPDATED
            fj = fugacity_j(T, P, components, coefficients, j)

            sumj += cij * fj
        nui = structure_cavities[i][INDEX_FOR_NU]
        sumi += nui * log(1 - sumj)

    return R_IDEAL_GAS * T * sumi


# thetaij

def theta_ij(T, P, structure_cavities: list, components: list, coefficients: list, i: int, j:int) -> float:
    sum = 0
    for k in range(len(components)):
        cik = langmuir_ij(T, structure_cavities, components, i, k)

        # TODO ARGUMENTS TO BE MODIFIED ONCE DEF IS UPDATED
        fk = fugacity_j(T, P, components, coefficients, k)

        sum += cik * fk
    cij = langmuir_ij(T, structure_cavities, components, i, j)
    fj = fugacity_j(T, P, components, coefficients, j)

    return cij * fj / (1 + sum)


## Optimisation (Kihara parameters)




## Determination of hydrate composition xj_H

def xj_H(T, P, structure_cavities: list, components: list, coefficients: list, j:int):

    sumk = 0
    for k in range(len(components)):
        sumi = 0
        for i in range(len(structure_cavities)):
            nui = structure_cavities[i][INDEX_FOR_NU]
            thetaik = theta_ij(T, P, structure_cavities, components, coefficients, i, k)
            sumi += nui * thetaik
        sumk += sumi

    sumi2 = 0
    for i in range(len(structure_cavities)):
        nui = structure_cavities[i][INDEX_FOR_NU]
        thetaij = theta_ij(T, P, structure_cavities, components, coefficients, i, j)
        sumi2 += nui * thetaij

    return sumi2 / sumk

def xj_H(structure_cavities: list, components: list, thetas: list, j:int):

    sumk = 0
    for k in range(len(components)):
        sumi = 0
        for i in range(len(structure_cavities)):
            nui = structure_cavities[i][INDEX_FOR_NU]
            thetaik = thetas[i][k]
            sumi += nui * thetaik
        sumk += sumi

    sumi2 = 0
    for i in range(len(structure_cavities)):
        nui = structure_cavities[i][INDEX_FOR_NU]
        thetaij = thetas[i][j]
        sumi2 += nui * thetaij

    return sumi2 / sumk


#### Algorithm

# TODO put this main in the file with the interface
def main(T, P, structure, components, coefficients, results: dict):
    deltaMuH = deltaMu_H(T, P, structure[KEY_FOR_CAVITIES], components, coefficients)
    deltaMUL = deltaMu_L(T, P, structure, components, coefficients)
    results['Temperature'] = T
    if abs(deltaMuH - deltaMUL) <= MAX_DIFFERENCE:
        results['Pressure'] = determineFinalValues(T, P, structure, components, coefficients)[0]
        results['Thetas'] = determineFinalValues(T, P, structure, components, coefficients)[1]
        print(determineFinalValues(T, P, structure, components, coefficients))
        # calculate x with all the values of theta
    else:
        print('no')

def determineFinalValues(T, P, structure, components, coefficients):
    Peq = P
    theta = []
    for j in range(len(components)):
        thetaS = theta_ij(T=T, P=P, structure_cavities=structure[KEY_FOR_CAVITIES], components=components, coefficients=coefficients, i=0, j=j)
        thetaL = theta_ij(T=T, P=P, structure_cavities=structure[KEY_FOR_CAVITIES], components=components, coefficients=coefficients, i=1, j=j)
        # xjH = xj_H(T, P, structure[KEY_FOR_CAVITIES], components, j)
        theta.append((thetaS, thetaL))
    return Peq, theta

### TESTS

# tab0 = [1, 2, 3]
# tab1 = 4
# tab = tab0 + [tab1]


# print(tab)

T_test = 10
P_test = 0.1
y0_test = 0.4
y1_test = 0.6
results_test = {'Components' : [], 'Composition' : [], 'Temperature' : -1, 'Pressure' : -1, 'Structure' : '', 'Thetas' : []}
cavities_test = [[None] * 3] * 2
cavities_test[0][INDEX_FOR_NU] = 2
cavities_test[1][INDEX_FOR_NU] = 8
cavities_test[0][INDEX_FOR_R] = 391
cavities_test[1][INDEX_FOR_R] = 433
cavities_test[0][INDEX_FOR_Z] = 20
cavities_test[1][INDEX_FOR_Z] = 24
structure_test = {KEY_FOR_MU0 : 1299, KEY_FOR_H0 : 1861, KEY_FOR_CP0 : -38.12, KEY_FOR_B : 0.141, KEY_FOR_V0 : 3, KEY_FOR_CAVITIES : cavities_test}
components_test = [{KEY_FOR_ID : 0, KEY_FOR_EPSILON : 154.54 * K_BOLTZMANN, KEY_FOR_SIGMA : 3.165, KEY_FOR_A : 0.3834, KEY_FOR_VINF : 32, KEY_FOR_K1 : 15.826277, KEY_FOR_K2 : -1559.0631, KEY_FOR_Y : y0_test, KEY_FOR_OMEGA : 0.0115, KEY_FOR_TC : 190.56, KEY_FOR_PC : 4.599},
                   {KEY_FOR_ID : 1, KEY_FOR_EPSILON : 168.77 * K_BOLTZMANN, KEY_FOR_SIGMA : 2.9818, KEY_FOR_A : 0.6805, KEY_FOR_VINF : 32, KEY_FOR_K1 : 14.283146, KEY_FOR_K2 : -2050.3269, KEY_FOR_Y : y1_test, KEY_FOR_OMEGA : 0.2276, KEY_FOR_TC : 304.19, KEY_FOR_PC : 7.342}
            ]
coefficients_test = [[0,0.1107], [0.1107, 0]]

main(T=T_test, P=P_test, structure=structure_test, components=components_test, coefficients=coefficients_test, results=results_test)
print(results_test)

# langmuir_ij(T_test, structure_test[KEY_FOR_CAVITIES], components_test, 0, 0)