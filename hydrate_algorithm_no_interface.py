# Hydrate Composition Calculation Algorithm without Interface

# import math, scipy
from scipy.integrate import quad
from math import exp, pi, log, sqrt

### Definition of constants

K_BOLTZMANN = 1.38 * 10**(-23)
R_IDEAL_GAS = 8.314
T_ZERO = 273.15
P_ZERO = 0

# Inside structure dictionary
KEY_FOR_MU0 = 'deltaMu0'
KEY_FOR_H0 = 'deltaH0'
KEY_FOR_CP0 = 'deltaCp0'
KEY_FOR_B = 'b'
KEY_FOR_V0 = 'deltaV0'
KEY_FOR_CAVITIES = 'cavities'

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

def a_w(T, P, components: list):
    sumj = 0
    for j in range(len(components)):
        Vinf = components[j][KEY_FOR_VINF]
        K1 = components[j][KEY_FOR_K1]
        K2 = components[j][KEY_FOR_K2]
        H = exp(K1 + K2/T)
        fj = fugacity_j(T, P, j)
        sumj += fj / (H * exp( P*Vinf / (R_IDEAL_GAS*T)))
    return 1 - sumj


# Injecting in deltaMu_L

def deltaMu_L(T, P , structure: dict, components: list):
    deltaMu0 = structure[KEY_FOR_MU0]
    deltah = deltah_L(T, structure)
    deltaV0 = structure[KEY_FOR_V0]
    aw = a_w(T, P, components)

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
        def w(r):
            # definition of delta_N
            def delta_N(N):
                return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

            return ( 2 * z * epsilon
                    * ( (sigma**12 / (r * R**11)) * (delta_N(10) + delta_N(11) * a/R )
                        (sigma** 6 / (r * R** 5)) * (delta_N( 4) + delta_N( 5) * a/R ) )
            )

        return exp( - w(r) / (K_BOLTZMANN * T)) * r**2
    integral = quad(integrand, 0, R)[0]

    return 4 * pi * integral  / (K_BOLTZMANN * T)

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
    kjk = coefficients[j][k]
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

def lnphi_j(T, P, components: list, coefficients: list, j: int):
    bj = b_j(components, j)
    bm = b_m(components)
    aj = a_j(T, components, j)
    am = a_m(T, components, coefficients)
    # TODO
    nu = 0

    return ( (bj / bm) * (P * nu - R_IDEAL_GAS * T) / (R_IDEAL_GAS * T)
            - log(P * (nu - bm) / (R_IDEAL_GAS * T))
            - am * (2 * sqrt(aj/am) - bj/bm ) * log((nu + bm) / nu) / (bm * R_IDEAL_GAS * T)
    )

# fj

def fugacity_j(T, P, components: list, coefficients, j: int):
    phij = exp(lnphi_j(T, P, components, coefficients, j))
    yj = components[j][KEY_FOR_Y]
    return phij * yj * P


# Injecting in deltaMu_H

def deltaMu_H(T, P, structure_cavities: list, components: list):
    sumi = 0
    for i in range(len(structure_cavities)):
        sumj = 0
        for j in range(len(components)):
            cij = langmuir_ij(T, structure_cavities, components, i, j)

            # TODO ARGUMENTS TO BE MODIFIED ONCE DEF IS UPDATED
            fj = fugacity_j(T, P, j)

            sumj += cij * fj
        nui = structure_cavities[i][INDEX_FOR_NU]
        sumi += nui * log(1 - sumj)

    return R_IDEAL_GAS * T * sumi


# thetaij

def theta_ij(T, P, structure_cavities: list, components: list, i: int, j:int) -> float:
    sum = 0
    for k in range(len(components)):
        cik = langmuir_ij(T, structure_cavities, components, i, k)

        # TODO ARGUMENTS TO BE MODIFIED ONCE DEF IS UPDATED
        fk = fugacity_j(T, P, k)

        sum += cik * fk
    cij = langmuir_ij(T, structure_cavities, components, i, j)
    fj = fugacity_j(T, P, j)

    return cij * fj / (1 + sum)


## Optimisation (Kihara parameters)


## Determination of hydrate composition xj_H




### TESTS

### fortran code
