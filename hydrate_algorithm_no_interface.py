# Hydrate Composition Calculation Algorithm without Interface

# import math, scipy
# from scipy.integrate import quad
from scipy import integrate
from math import exp, pi, log, sqrt, cos
import numpy as np
import sympy as sy
from scipy.special import cbrt
import hydrate_algorithm_with_interface as hydinter

### Definition of constants

K_BOLTZMANN = 1.38E-23
R_IDEAL_GAS = 8.314
T_ZERO = 273.15
P_ZERO = 0

MAX_DIFFERENCE = 1E-5

# Inside structure dictionary
# KEY_FOR_MU0 = 'deltaMu0'
# KEY_FOR_H0 = 'deltaH0'
# KEY_FOR_CP0 = 'deltaCp0'
# KEY_FOR_B = 'b'
# KEY_FOR_V0 = 'deltaV0'
# KEY_FOR_CAVITIES = 'cavities'       # list (2 items) of lists containing cavities caracteristics

# # Inside 'cavities' value (list of lists) of structure dictionnary
# INDEX_FOR_NU = 0            # index pointing to nu (number of this type of cavity) in the cavity info table in the structure info table
# INDEX_FOR_R = 1             # index pointing to R (cavity ray) in the cavity info table in the structure info table
# INDEX_FOR_Z = 2             # index pointing to z (coordination number) in the cavity info table

# # Inside component dictionnary in components list (components is a list of dictionnaries)
# KEY_FOR_ID = 'id'
# KEY_FOR_EPSILON_OVER_K = 'epsilon_k'     # key pointing to epsilon in the component info table
# KEY_FOR_SIGMA = 'sigma'         # key pointing to sigma in the component info table
# KEY_FOR_A = 'a'                 # key pointing to a in the component info table
# KEY_FOR_VINF = 'Vinf'
# KEY_FOR_K1 = 'k1'
# KEY_FOR_K2 = 'k2'
# KEY_FOR_Y = 'y'
# KEY_FOR_OMEGA = 'omega'
# KEY_FOR_TC = 'Tc'
# KEY_FOR_PC = 'Pc'

# Note: for greater readibility, indexes will be used as follows:
# i for cavity types
# j for components

### Equilibrium condition: solving deltaMu_L = deltaMu_H

## Calculation of deltaMu_L using classical thermodynamcis

# deltah

def deltah_L(Temp : float, structure: hydinter.Structure):
    deltaH0 = structure.deltaH0
    deltaCp0 = structure.deltaCp0
    b =  structure.b

    # definition of deltaCp
    def deltaCp(T):
        return deltaCp0 + b*(T - T_ZERO)
    integral = integrate.quad(deltaCp, T_ZERO, Temp)[0]

    return deltaH0 + integral

# non ideality term ln(a_w)

def a_w(T : float, P : float, components: list[hydinter.Component], coefficients: list):
    sumj = 0
    for component_key in components:
        component = components[component_key]
        Vinf = component.vinf
        K1 = component.k1
        K2 = component.k2
        print(K1, K2)
        H = exp(K1 + K2/T)
        fj = fugacity_j(T, P, components, coefficients, component_key)
        sumj += fj / (H * exp( P*Vinf / (R_IDEAL_GAS*T)))
    return 1 - sumj


# Injecting in deltaMu_L

def deltaMu_L(T : float ,P : float, structure: hydinter.Structure, components: list[hydinter.Component], coefficients: list):
    deltaMu0 = structure.deltaMu0
    deltah = deltah_L(T, structure)
    deltaV0 = structure.deltaV0
    # aw = a_w(T, P, components, coefficients)

    termMu = T * deltaMu0 / T_ZERO

    def integrandH(Temp):
        return deltah / (Temp**2)
    termH = T * integrate.quad(integrandH, T_ZERO, T)[0]

    termV = deltaV0 * (P - P_ZERO)

    # termAw = R_IDEAL_GAS * T * log(aw)

    # return termMu - termH + termV - termAw
    return termMu - termH + termV


## Calculation of deltaMu_H using statistical thermodynamics

# Cij: Langmuir constant for component j in cavity i

def langmuir_ij(T : float, structure_cavities: list[hydinter.Cavity], components: dict[str, hydinter.Component], i: int, component_keyj) -> float:

    # determination of R, z, and Kihara parameters for a component j in a type i cavity
    R = structure_cavities[i].r
    z = structure_cavities[i].z
    epsilonk = components[component_keyj].epsilon
    sigma = components[component_keyj].sigma
    a = components[component_keyj].a

    # definition of the function inside the integral
    def integrand(r):

        # definition of w(r)
        def w_r(r):
            # definition of delta_N
            def delta_N(N):
                # print('delta : ', ((1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
                return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

            return ( 2 * z * epsilonk
                    * ( (sigma**12 / (r * R**11)) * (delta_N(10) + delta_N(11) * a/R )
                    -   (sigma** 6 / (r * R** 5)) * (delta_N(4) + delta_N(5) * a/R ) )
            )
        # print('w : ', w_r(r)/ T)
        # print('exp : ' , sy.exp( - w_r(r) / T))
        print(sy.exp( - w_r(r) / T) * (r**2 / K_BOLTZMANN))
        return sy.exp( - w_r(r) / T) * (r**2 / K_BOLTZMANN)
        # TODO pb : k is 1E-23 so it makes the inside of exp really big

    r = sy.Symbol("r")
    print('encore bloque ici')
    integral = sy.integrate(integrand(r), (r, 0, R))
    # integral = integrate.quad(integrand, 0, R)[0]
    # print('0.1 :', integrand(1E-30), 'R :', integrand(R *1E-30), R)
    print('arrive ici')

    print('int : ' , integral)
    # return 4 * pi * 1  / (K_BOLTZMANN * T)
    return 4 * pi * integrate.quad(integrand, 0, R)[0]  / T

# a_m

def a_j(T : float, components: dict[str, hydinter.Component], component_keyj):
    Pc = components[component_keyj].pc
    Tc = components[component_keyj].tc
    omegaj = components[component_keyj].omega
    m = 0.48 + 1.1574 * omegaj - 0.176 * (omegaj**2)
    alphaj = ( 1 + m * (1- sqrt(T/Tc)) ) ** 2
    return 0.42747 * (R_IDEAL_GAS**2) * (Tc**2) * sqrt(T) * alphaj / Pc

def a_jk(T : float, components: dict[str, hydinter.Component], coefficients: list, component_keyj, component_keyk):
    aj = a_j(T, components, component_keyj)
    ak = a_j(T, components, component_keyk)
    idj = components[component_keyj].id
    idk = components[component_keyk].id
    kjk = coefficients[idj][idk]
    return sqrt(aj * ak) * (1 - kjk)

def a_m(T : float, components: dict[str, hydinter.Component], coefficients: list[list]):
    sumj = 0
    for component_keyj in components:
        yj = components[component_keyj].y
        sumk = 0
        for component_keyk in components:
            yk = components[component_keyk].y
            ajk = a_jk(T, components, coefficients, component_keyj, component_keyk)
            sumk += yk * ajk
        sumj += yj * sumk
    return sumj

# b_m

def b_j(components: dict[str, hydinter.Component], component_keyj):
    Tc = components[component_keyj].tc
    Pc = components[component_keyj].pc
    return 0.0866 * R_IDEAL_GAS * Tc / Pc

def b_m(components: dict[str, hydinter.Component]):
    sumj = 0
    for component_key in components:
        yj = components[component_key].y
        bj = b_j(components, component_key)
        sumj += yj * bj
    return sumj

# phij

def lnphi_j(T : float, Pres : float, components: dict[str, hydinter.Component], coefficients: list[list], component_keyj):
    bj = b_j(components, component_keyj)
    bm = b_m(components)
    aj = a_j(T, components, component_keyj)
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
    #         - log(Zj - B)
    #         - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(abs(1 + B/Zj))
    # ))
    return ( (bj / bm) * (Zj - 1)
            - log(abs(Zj - B))
            - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(abs(1 + B/Zj))
    )


# fj

def fugacity_j(T : float, P : float, components: list, coefficients : list[list], component_keyj):
    # phij = exp(lnphi_j(T, P, components, coefficients, j))
    phij = exp(-lnphi_j(T, P, components, coefficients, component_keyj))
    yj = components[component_keyj].y
    return phij * yj * P


# Injecting in deltaMu_H

def deltaMu_H(T, P, structure_cavities: list[hydinter.Cavity], components: dict[str, hydinter.Component], coefficients: list[list]):
    sumi = 0
    for i in range(len(structure_cavities)):
        sumj = 0
        for component_key in components:
            cij = langmuir_ij(T, structure_cavities, components, i, component_key)

            # TODO ARGUMENTS TO BE MODIFIED ONCE DEF IS UPDATED
            fj = fugacity_j(T, P, components, coefficients, component_key)

            sumj += cij * fj
        nui = structure_cavities[i].nu
        sumi += nui * log(1 - sumj)

    return R_IDEAL_GAS * T * sumi


# thetaij

def theta_ij(T : float, P : float, structure_cavities: list[hydinter.Cavity], components: dict[str, hydinter.Component], coefficients: list[list], i: int, component_keyj) -> float:
    sumtheta = 0
    for component_keyk in components:
        cik = langmuir_ij(T, structure_cavities, components, i, component_keyk)
        fk = fugacity_j(T, P, components, coefficients, component_keyk)
        sumtheta += cik * fk
    cij = langmuir_ij(T, structure_cavities, components, i, component_keyj)
    fj = fugacity_j(T, P, components, coefficients, component_keyj)

    return cij * fj / (1 + sumtheta)


## Optimisation (Kihara parameters)




## Determination of hydrate composition xj_H

# def xj_H(T, P, structure_cavities: list, components: list, coefficients: list, j:int):

    # sumk = 0
    # for k in range(len(components)):
    #     sumi = 0
    #     for i in range(len(structure_cavities)):
    #         nui = structure_cavities[i][INDEX_FOR_NU]
    #         thetaik = theta_ij(T, P, structure_cavities, components, coefficients, i, k)
    #         sumi += nui * thetaik
    #     sumk += sumi

    # sumi2 = 0
    # for i in range(len(structure_cavities)):
    #     nui = structure_cavities[i][INDEX_FOR_NU]
    #     thetaij = theta_ij(T, P, structure_cavities, components, coefficients, i, j)
    #     sumi2 += nui * thetaij

    # return sumi2 / sumk

def xj_H(structure_cavities: list[hydinter.Cavity], components: dict[str, hydinter.Component], thetas: dict, component_keyj: int):      # theta = dict[tuple] or dict[list]

    sumk = 0
    for component_keyk in components:
        sumi = 0
        for i in range(len(structure_cavities)):
            nui = structure_cavities[i].nu
            # thetaik = thetas[i][component_keyk]
            # or ki ?
            thetaik = thetas[component_keyk][i]
            sumi += nui * thetaik
        sumk += sumi

    sumi2 = 0
    for i in range(len(structure_cavities)):
        nui = structure_cavities[i].nu
        # thetaij = thetas[i][component_keyj]
        thetaij = thetas[component_keyj][i]
        sumi2 += nui * thetaij

    return sumi2 / sumk


#### Algorithm

# TODO put this main in the file with the interface
def main(T : float, P : float, structure : hydinter.Structure, components : dict[str, hydinter.Component], coefficients : list[list], results: dict):
    deltaMuH = deltaMu_H(T, P, structure.cavities, components, coefficients)
    deltaMUL = deltaMu_L(T, P, structure, components, coefficients)
    results['Temperature'] = T
    if abs(deltaMuH - deltaMUL) <= MAX_DIFFERENCE:
        results['Pressure'] = determineFinalValues(T, P, structure, components, coefficients)[0]
        results['Thetas'] = determineFinalValues(T, P, structure, components, coefficients)[1]
        print(determineFinalValues(T, P, structure, components, coefficients))
        # calculate x with all the values of theta
    else:
        print('no')

def determineFinalValues(T : float, P: float, structure: hydinter.Structure, components : dict[str, hydinter.Component], coefficients : list[list]):
    Peq = P
    # regression algo for P estimation
    # theta = []
    # for j in range(len(components)):
    #     thetaS = theta_ij(T=T, P=P, structure_cavities=structure.cavities, components=components, coefficients=coefficients, i=0, j=j)
    #     thetaL = theta_ij(T=T, P=P, structure_cavities=structure.cavities, components=components, coefficients=coefficients, i=1, j=j)
    #     theta.append((thetaS, thetaL))
    thetas = [(theta_ij(T=T, P=P, structure_cavities=structure.cavities, components=components, coefficients=coefficients, i=0, component_keyj=component_key),
              theta_ij(T=T, P=P, structure_cavities=structure.cavities, components=components, coefficients=coefficients, i=1, component_keyj=component_key))
             for component_key in components]
    xH = [xj_H(structure.cavities, components, thetas, component_keyj=component_key) for component_key in components]
    return Peq, thetas, xH

### TESTS

# tab0 = [1, 2, 3]
# tab1 = 4
# tab = tab0 + [tab1]


T_test = 100
P_test = 0.1
y0_test = 0.4
y1_test = 0.6
results_test = {'Components' : [], 'Composition' : [], 'Temperature' : -1, 'Pressure' : -1, 'Structure' : '', 'Thetas' : []}
cavities_test = [hydinter.Cavity(3.91E-10, 20, 2), hydinter.Cavity(4.33E-10, 24, 8)]
structure_test = hydinter.Structure('I', 1299, 1861, -38.12, 0.141, 3, cavities_test)
components_test = {'CH4' : hydinter.Component('CH4', 0, y0_test, 190.56, 4.599, 154.54, 3.165, 0.3834, 32, 15.826277, -1559.0631, 0.0115),
                   'CO2' : hydinter.Component('CO2', 1, y1_test, 304.19, 7.342, 168.77, 2.9818, 0.6805, 32, 14.283146, -2050.3269, 0.2276)
                   }

# cavities_test = [[None for i in range(3)] for j in range(2)]
# cavities_test[0][INDEX_FOR_NU] = 2
# cavities_test[1][INDEX_FOR_NU] = 8
# cavities_test[0][INDEX_FOR_R] = 3.91E-10
# cavities_test[1][INDEX_FOR_R] = 4.33E-10
# cavities_test[0][INDEX_FOR_Z] = 20
# cavities_test[1][INDEX_FOR_Z] = 24
# print(cavities_test)
# structure_test = {KEY_FOR_MU0 : 1299, KEY_FOR_H0 : 1861, KEY_FOR_CP0 : -38.12, KEY_FOR_B : 0.141, KEY_FOR_V0 : 3, KEY_FOR_CAVITIES : cavities_test}
# # components_test = [{KEY_FOR_ID : 0, KEY_FOR_EPSILON_OVER_K : 154.54, KEY_FOR_SIGMA : 3.165, KEY_FOR_A : 0.3834, KEY_FOR_VINF : 32, KEY_FOR_K1 : 15.826277, KEY_FOR_K2 : -1559.0631, KEY_FOR_Y : y0_test, KEY_FOR_OMEGA : 0.0115, KEY_FOR_TC : 190.56, KEY_FOR_PC : 4.599},
# #                    {KEY_FOR_ID : 1, KEY_FOR_EPSILON_OVER_K : 168.77, KEY_FOR_SIGMA : 2.9818, KEY_FOR_A : 0.6805, KEY_FOR_VINF : 32, KEY_FOR_K1 : 14.283146, KEY_FOR_K2 : -2050.3269, KEY_FOR_Y : y1_test, KEY_FOR_OMEGA : 0.2276, KEY_FOR_TC : 304.19, KEY_FOR_PC : 7.342}
# #             ]
# components_test = [{KEY_FOR_ID : 0, KEY_FOR_EPSILON_OVER_K : 154.54, KEY_FOR_SIGMA : 3.165, KEY_FOR_A : 0.3834, KEY_FOR_VINF : 32, KEY_FOR_K1 : 15.826277, KEY_FOR_K2 : -1559.0631, KEY_FOR_Y : y0_test, KEY_FOR_OMEGA : 0.0115, KEY_FOR_TC : 190.56, KEY_FOR_PC : 4.599},
#             ]
coefficients_test = [[0,0.1107], [0.1107, 0]]

# print(components_test)
# print(structure_test)
# main(T=T_test, P=P_test, structure=structure_test, components=components_test, coefficients=coefficients_test, results=results_test)
# print(results_test)

# print('langmuir : ', langmuir_ij(T_test, structure_test.cavities, components_test, 0, 'CH4'))

# print(deltaMu_L(T_test, P_test, structure_test, components_test, coefficients_test))

# R en metres
# k en m kg s K
# P et fugacite en Pa