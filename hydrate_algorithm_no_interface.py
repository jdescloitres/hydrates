# Hydrate Composition Calculation Algorithm without Interface

import math, scipy, cmath
# from scipy.integrate import quad
from scipy import integrate
from math import exp, pi, log, sqrt, cos
import numpy as np
import scipy.integrate
import sympy as sy
from scipy.special import cbrt
from scipy.optimize import newton, curve_fit
from scipy.interpolate import interp1d
# import hydrate_algorithm_with_interface as hydinter

import matplotlib.pyplot as plt

### Definition of constants

K_BOLTZMANN = 1.38E-23
R_IDEAL_GAS = 8.314
T_ZERO = 273.15
P_ZERO = 0

MAX_DIFFERENCE = 1E-5

# Note: epsilon/k is prefered over epsilon for calculations in this program

# Note: for greater readibility, indexes will be used as follows:
# i for cavity types
# j for components

    ### Equilibrium condition: solving deltaMu_L = deltaMu_H

## Calculation of deltaMu_L using classical thermodynamcis
# deltah
def deltah_L(Temp : float, structure):
    deltaH0 = structure.deltaH0
    deltaCp0 = structure.deltaCp0
    b =  structure.b

    # definition of deltaCp
    def deltaCp(T):
        return deltaCp0 + b*(T - T_ZERO)
    integral = integrate.quad(deltaCp, T_ZERO, Temp)[0]

    return deltaH0 + integral

# non ideality term ln(a_w)
def a_w(T : float, P : float, components, bips: dict):
    sumj = 0
    for component_key in components:
        component = components[component_key]
        Vinf = component.vinf * 1E-6
        K1 = component.k1
        K2 = component.k2
        # print(K1, K2)
        H = exp(K1 + K2/T) * 101325
        fj = fugacity_j(T, P, components, bips, component_key)
        # print('P, fj :' , P, fj)
        sumj += fj / (H * exp( P*1E6*Vinf / (R_IDEAL_GAS*T)))
    return 1 - sumj


# Injecting in deltaMu_L
def deltaMu_L(T : float ,P : float, structure, components, bips: dict):
    print('first, deltaMuL')
    deltaMu0 = structure.deltaMu0
    deltah = deltah_L(T, structure)
    deltaV0 = structure.deltaV0
    aw = a_w(T, P, components, bips)
    # print('P:' , P)

    termMu = T * deltaMu0 / T_ZERO

    def integrandH(Temp):
        return deltah / (Temp**2)
    termH = T * integrate.quad(integrandH, T_ZERO, T)[0]

    termV = deltaV0 * (P*1E6 - P_ZERO)
    # print('P, V : ', P, deltaV0, termV)

    termAw = R_IDEAL_GAS * T * log(aw)

    print(P, 'deltaL :', termMu - termH + termV - termAw)
    return termMu - termH + termV - termAw
    # return termMu - termH + termV


## Calculation of deltaMu_H using statistical thermodynamics
# Cij: Langmuir constant for component j in cavity i
def langmuir_ij(T : float, structure_cavities, components, i: int, component_keyj) -> float:

    # determination of R, z, and Kihara parameters for a component j in a type i cavity
    R = structure_cavities[i].r * 1E-10
    z = structure_cavities[i].z
    epsilonk = components[component_keyj].epsilon
    sigma = components[component_keyj].sigma * 1E-10
    a = components[component_keyj].a * 1E-10
    # print(component_keyj, R, z, epsilonk, sigma, a)

    # p = ( (sigma/R)**6 ) * ( (1 - a/R)**(-6) )
    # q = ( (sigma/R)**6 ) * ( (1 - a/R)**(-7) ) * a/R
    # r = -1
    # s = - ( (1 - a/R)**(-1) ) * a/R
    # k = 1 / (R-a)
    # l = 2 * ( (sigma**6) / (R**5)) * ( (1 - a/R)**(-4) ) * k
    # A = 2 * z * epsilonk * K_BOLTZMANN * l * (p + q + r + s)
    # B = 2 * z * epsilonk * K_BOLTZMANN * l * (44*p + 52*q + 15*r + 21*s)
    # C = np.exp(- A / (K_BOLTZMANN*T))
    # D = - B / (K_BOLTZMANN*T)
    # print('D : ', D)
    # print(1E-30 * 4 * pi * (1 / (K_BOLTZMANN * T)) * (C / (2*D)) * (2*D - 1) * np.exp(D * (R**2)))
    # return 1E-30 * 4 * pi * (1 / (K_BOLTZMANN * T)) * (C / (2*D)) * (2*D - 1) * np.exp(D * (R**2))

    # print(T)

    # definition of the function inside the integral
    def integrand(r):

        # definition of w(r)
        def w_r(r):
            # definition of delta_N
            def delta_N(N):
                return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

            # print("sigma12 : ", (sigma**12)/(r * R**11) , "sigma5 : ", (sigma** 6 / (r * R** 5)))
            return ( 2 * z * epsilonk
                    * ( ((sigma**12) / (r * (R**11))) * (delta_N(10) + delta_N(11) * a/R )
                    -   ((sigma** 6) / (r * (R** 5))) * (delta_N(4) + delta_N(5) * a/R ) )
            )
        return r**2 * np.exp(-w_r(r)/T)
    #     print('r : ', r , 'w/kT : ', w_r(r)/ T)
    #     # print('w/T : ', w_r(r)/T)
    #     # print('exp : ' , sy.exp( - w_r(r) / T))
    #     # print( 'puissance : ', ((np.exp( - w_r(r) / T))**(1/K_BOLTZMANN)) * (r**2 / K_BOLTZMANN))
    #     # print( 'dans exp :', np.exp( - w_r(r) / K_BOLTZMANN*T) * (r**2))
    #     # return np.exp( - w_r(r) / (T)) * (r**2)
    #     # print('r : ', r, '\n', 1 - w_r(r) / (K_BOLTZMANN *T) + (w_r(r) / (K_BOLTZMANN *T))**2 )
    #     # return (1 - w_r(r) / (K_BOLTZMANN *T) + (1/2)*(w_r(r) / (K_BOLTZMANN *T))**2 ) * (r**2)
    #     # return pi
    #     print((r**2)* ((np.exp(- w_r(r) / (T)))**(1/K_BOLTZMANN**2)))
    #     # return (r**2)* ((np.exp(- w_r(r) / (T)))**(1/K_BOLTZMANN**1))
    #     return (r**2)* ((1 - w_r(r) / (T) + (1/2)*(w_r(r)**2 / (T)**2))**(1/K_BOLTZMANN**2))

    #     # return ((np.exp( - w_r(r) / T))**(1/K_BOLTZMANN)) * (r**2 / K_BOLTZMANN)
    #     # TODO pb : k is 1E-23 so it makes the inside of exp really big
    #     # print( 'dans exp :', np.exp( - w_r(r) / T) * (r**2))
    #     # return np.exp( - w_r(r) / T) * (r**2)

    # # r = sy.Symbol("r")
    # # print('encore bloque ici')
    # # integral = sy.integrate(integrand(r), (r, 0, R))
    integral = scipy.integrate.quad(integrand, 0, R-a)[0]
    # # integral = scipy.integrate.fixed_quad(integrand, 0, R)[0]
    # # print('1E-30 :', integrand(2E-30), 'R :', integrand(R), R)
    # # print('arrive ici')

    # print('int : ' , integral)
    # # return 4 * pi * 1  / (K_BOLTZMANN * T)
    # # return 4 * pi * integrate.quad(integrand, 0, R)[0]  / T
    return 4 * pi * integral  / (K_BOLTZMANN * T)
    # # return 4 * pi * integral  / T


    ### calculs for now pour N2
    # if i== 0:
    #     # return (1.617E-3/101325*T)*np.exp(2905/T)
    #     return 4.44E-7
    # elif i == 1:
    #     # return (6.078E-3/101325*T)*np.exp(2431/T)
    #     return 3.05E-6

    # return 3.5E-7

# a_m
def a_j(T : float, components, component_keyj):
    Pc = components[component_keyj].pc
    Tc = components[component_keyj].tc
    omegaj = components[component_keyj].omega
    m = 0.48 + 1.1574 * omegaj - 0.176 * (omegaj**2)
    alphaj = ( 1 + m * (1- sqrt(T/Tc)) ) ** 2
    return 0.42747 * (R_IDEAL_GAS**2) * (Tc**2) * sqrt(T) * alphaj / Pc

def a_jk(T : float, components, bips: dict, component_keyj, component_keyk):
    aj = a_j(T, components, component_keyj)
    ak = a_j(T, components, component_keyk)
    idj = components[component_keyj].formula
    idk = components[component_keyk].formula
    kjk = bips[tuple(sorted((idj,idk)))]
    return sqrt(aj * ak) * (1 - kjk)

def a_m(T : float, components, bips: dict):
    sumj = 0
    for component_keyj in components:
        yj = components[component_keyj].y
        sumk = 0
        for component_keyk in components:
            yk = components[component_keyk].y
            ajk = a_jk(T, components, bips, component_keyj, component_keyk)
            sumk += yk * ajk
        sumj += yj * sumk
    return sumj

# b_m
def b_j(components, component_keyj):
    Tc = components[component_keyj].tc
    Pc = components[component_keyj].pc
    # print(Tc, Pc, R_IDEAL_GAS)
    return 0.0866 * R_IDEAL_GAS * Tc / Pc

def b_m(components):
    sumj = 0
    for component_key in components:
        yj = components[component_key].y
        bj = b_j(components, component_key)
        sumj += yj * bj
        # print(yj, bj, sumj)
    return sumj

# phij
def lnphi_j(T : float, Pres : float, components, bips: dict, component_keyj):
    bj = b_j(components, component_keyj)
    bm = b_m(components)
    aj = a_j(T, components, component_keyj)
    am = a_m(T, components, bips)
    # print('lnphi ' , component_keyj)
    # print('bj etc : ', bj, bm, aj, am)
    # # from equation of state, Z = PV/RT is solution of the cubic equation
    # Z**3 - Z**2 + (A - B**2 - B)*Z - AB = 0 , where
    # print('P :' , Pres)
    A = am * Pres / ((R_IDEAL_GAS**2) * (T**2) * sqrt(T))
    B = bm * Pres / (R_IDEAL_GAS * T)
    # a1 = 1
    # b1 = -1
    # c1 = A - B**2 - B
    # d1 = -A*B
    # c1 = 8E-7
    # d1 = -1E-14
    # print(c1, d1)

    # # for better readibility, let's define
    # p = (3*a*c - b**2) / (3*a**2)
    # q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
    # delta = (q**2 / 4 + p**3 / 27)
    # print('delta : ', delta)
    # if delta > 0:  # one real and two complex roots
    #     u = (-q/2 + cmath.sqrt(delta))**(1/3)
    #     v = (-q/2 - cmath.sqrt(delta))**(1/3)
    #     x1 = u + v - b / (3*a)
    #     x2 = -(u + v)/2 - b / (3*a) + (u - v)*cmath.sqrt(3)/2j
    #     x3 = -(u + v)/2 - b / (3*a) - (u - v)*cmath.sqrt(3)/2j
    #     print("One real root and two complex roots:")
    #     print("x1 = ", x1.real)
    #     print("x2 = ", x2)
    #     print("x3 = ", x3)
    # elif delta == 0:  # three real roots, two are equal
    #     u = (-q/2)**(1/3)
    #     v = u
    #     x1 = 2*u - b / (3*a)
    #     x2 = -u - b / (3*a)
    #     print("Three real roots, two are equal:")
    #     print("x1 = ", x1.real)
    #     print("x2 = ", x2.real)
    #     print("x3 = ", x3.real)
    # else:  # three distinct real roots
    #     u = (-q/2 + cmath.sqrt(delta))**(1/3)
    #     v = (-q/2 - cmath.sqrt(delta))**(1/3)
    #     x1 = u + v - b / (3*a)
    #     x2 = -(u + v)/2 - b / (3*a) + (u - v)*cmath.sqrt(3)/2j
    #     x3 = -(u + v)/2 - b / (3*a) - (u - v)*cmath.sqrt(3)/2j
    #     print("Three distinct real roots:")
    #     print("x1 = ", x1.real)
    #     print("x2 = ", x2.real)
    #     print("x3 = ", x3.real)


    # return ( (bj / bm) * (x1 - 1)
    #         - log(x1.real - B)
    #         - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(1 + B/x1.real)
    # )

    # for better readibility, let's define
    p = -1
    q = (A - B**2 - B)
    r = - A * B
    m = q - (p**2) / 3
    n = r + (2 * p**3 - 9*p*q) / 27
    delt = (n**2)/4 + (m**3) / 27
    # print('delt : ', delt)
    if delt >= 0:
        Zj = (- p / 3
             + cbrt(sqrt(delt) - n/2)
             + cbrt(- sqrt(delt) - n/2)
        )
        # print('sup : ', Zj)

    elif delt < 0:
        cosangle = - (n / abs(n)) * sqrt( (- 27 * n**2) / (4 * m**3) )
        # print(cosangle)
        angle = math.acos(cosangle)
        # print(angle)
        Zj = ( - p / 3
            + 2 * sqrt(-m / 3) * cos(angle/3)
        )
        # print('inf :' , Zj)
    # else:
    #     Zj = (- p / 3
    #           + cbrt(- n/2)
    #     )
    #     print('Error: no known value of Z for delta = 0')
    #     # TODO raise error here

    # print('Zj : ', Zj, 'P : ', Pres)
    # print(Zj, B, B/Zj)
    # print(( (bj / bm) * (Zj - 1)
    #         - log(Zj - B)
    #         - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(abs(1 + B/Zj))
    # ))
    # print('tot', ( (bj / bm) * (Zj - 1)
    #         - log(Zj - B)
    #         - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(1 + B/Zj)
    # ))


    # x = np.linspace(-5, 5,1000)

    # def cubic_eq(x,a,b,c,d):
    #     return a*x**3 +b*x**2 + c*x + d

    # y = cubic_eq(x,a1,b1,c1,d1)

    # fig, ax = plt.subplots()
    # ax.plot(x, y)

    # ax.spines['left'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['top'].set_color('none')

    # plt.show()

    return ( (bj / bm) * (Zj - 1)
            - log(Zj - B)
            - (A / B) * (2 * sqrt(aj/am) - bj/bm ) * log(1 + B/Zj)
    )

# fj
def fugacity_j(T : float, P : float, components: dict, bips : dict, component_keyj):
    # phij = exp(-lnphi_j(T, P, components, bips, component_keyj))
    if P == 0:
        return 0
    phij = exp(lnphi_j(T, P*1E6, components, bips, component_keyj))
    # print("PROBLEME ICI, P EST NEG, MEME PB QUE AVANT, LE P PART DANS TOUS LES SENS : ")
    yj = components[component_keyj].y
    # print(phij * yj * P)
    return phij * yj * P*1E6

# Injecting in deltaMu_H
def deltaMu_H(T, P, structure_cavities, components, bips: dict):
    # print('deltaH')
    sumi = 0
    print('Pc : ', P, T)
    cij = {component_key : (langmuir_ij(T, structure_cavities, components, i=0, component_keyj=component_key),
                            langmuir_ij(T, structure_cavities, components, i=1, component_keyj=component_key)) for component_key in sorted(components)}
    print('Pf :', P)
    fj = {component_key : fugacity_j(T, P, components, bips, component_keyj=component_key) for component_key in sorted(components)}

    # print(cij, fj)
    for i in range(len(structure_cavities)):

        sumthetai = 0
        for component_keyj in components:
            sumthetai += theta_ij(cij, fj, structure_cavities, i, component_keyj)

        # TODO should it be nu or nu/nutot that is multiplied here ?
        nui = structure_cavities[i].nu
        sumi += nui * log(1 - sumthetai)
    print(sumi)
    print(P, 'deltaH :', - R_IDEAL_GAS * T * sumi)

    # sumi = 0
    # for i in range(len(structure_cavities)):
    #     sumj = 0
    #     for component_key in components:
    #         cij = langmuir_ij(T, structure_cavities, components, i, component_key)
    #         fj = fugacity_j(T, P, components, bips, component_key)
    #         sumj += cij * fj
    #     nui = structure_cavities[i].nu
    #     sumi += nui * log(1 - sumj)

    # print(sumi, R_IDEAL_GAS * T * sumi)
    return - R_IDEAL_GAS * T * sumi


# thetaij
def theta_ij(cij : dict[str, tuple], fj : dict[str, float], structure_cavities, i : int, component_keyj) -> float:
    sumtheta = 0
    for component_keyk in cij:
        sumtheta += cij[component_keyk][i] * fj[component_keyk]
    return cij[component_keyj][i] * fj[component_keyj] / (1 + sumtheta)

def thetas_iall(thetas : dict[str, tuple], components):
    sumthetaS = 0
    sumthetaL = 0
    for component_key in components:
        sumthetaS += thetas[components[component_key].formula][0]
        sumthetaL += thetas[components[component_key].formula][1]
    return sumthetaS, sumthetaL


## Optimisation (Kihara parameters)
# interpolation from (P,T) data
def PfromT_f(list_temp, list_pres):
    return interp1d(x= list_temp, y= list_pres, fill_value='extrapolate', bounds_error=False)
def TfromP_f(list_pres, list_temp):
    # print(list_pres ,list_temp)
    return interp1d(x= list_pres, y= list_temp, fill_value='extrapolate', bounds_error=False)


from scipy.optimize import brentq

# calculates the P that verifies the deltaMus equality, i.e Peq, for a certain model i from literature
def calculatePi(Ti, Pexpi, structurei, componentsi, bipsi):
    print('start Pi with ', Pexpi, Ti)
    def f(P, T = Ti, structure = structurei, components=componentsi,bips= bipsi):
        muH = deltaMu_H(T, P, structure.cavities, components, bips)
        muL = deltaMu_L(T, P, structure, components, bips)
        print(muH, muL, P)
        return muL - muH
        # return deltaMu_H(T, P, structure.cavities, components, bips) - deltaMu_L(T, P, structure, components, bips)
    # P_i = newton(func= f, x0=Pexpi)
    P_i = brentq(f, a = 0, b = 500)
    print('calculatePi : ' , P_i, f(P_i), deltaMu_L(Ti, P_i, structurei, componentsi, bipsi))
    return (P_i, deltaMu_L(Ti, P_i, structurei, componentsi, bipsi))

# TODO change newton to brentq
# calculates the T that verifies the deltaMus equality, i.e Peq, for a certain model i from literature
def calculateTi(Pi, Texpi, structurei, componentsi, bipsi):
    def f(T, P = Pi, structure = structurei, components=componentsi,bips= bipsi):
        return deltaMu_H(T, P, structure.cavities, components, bips) - deltaMu_L(T, P, structure, components, bips)
    T_i = newton(func= f, x0=Texpi)
    return (T_i, deltaMu_L(P = Pi, T= T_i, structure=structurei, components=componentsi, bips=bipsi))


# TODO CHANGE WITH ACUTAL ARGUMENTS
# (note: the optimization is done for that component as sole component of the gas)
def optimisationKiharafromT(T1, T2, calculate_unknownPfromT, calculateTfromP, allPTinterpol, component_pure, structure, bips, list_models, n_T = 10):
    sigma, epsilon = component_pure.sigma, component_pure.epsilon
    PfromT = allPTinterpol[component_pure.formula][0][0]
    TfromP = allPTinterpol[component_pure.formula][1][0]
    xy_P = []
    i = 0
    if structure.id == 'II':
        i = 1
    for j in range(0, n_T + 1):
        for model_values in list_models.values():
            print(model_values)
            prev_deltaMu0, prev_deltaH0 = structure.deltaMu0, structure.deltaH0
            structure.deltaMu0, structure.deltaH0 = float(model_values[i]['deltaMu0']), float(model_values[i]['deltaH0'])
            # NOTE : calculate Pi returns TWO values, one is delta TODO
            # PB : changing the values of the models makes P go negative....
            xy_P += [ calculatePi(Ti = T1 + j*(T2 - T1)/n_T, Pexpi= calculate_unknownPfromT(T1 + j*(T2 - T1)/n_T, PfromT), structurei=structure, componentsi={component_pure.formula : component_pure}, bipsi=bips) ]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of temp
            print(xy_P)
    structure.deltaMu0, structure.deltaH0 = prev_deltaMu0, prev_deltaH0

    # note: unlike deltaMu_L, deltaMu_H does not depend on the macroscopic values, so it is independent from the models that were used to get Pi
    def f_Ki(P_values, eps, sig):
        # component_pure.epsilon = eps
        # component_pure.sigma = sig
        # print('pb : ici, P est un array, pas une valeur unique')
        # return deltaMu_H(T=calculateTfromP(P, TfromP), P=P, components={component_pure.formula : component_pure}, structure_cavities=structure.cavities, bips=bips)
        component_pure.epsilon = eps
        component_pure.sigma = sig
        delta_values = np.array([deltaMu_H(T=calculateTfromP(P, TfromP), P=P, components={component_pure.formula : component_pure}, structure_cavities=structure.cavities, bips=bips)
                                 for P in P_values])
        print(P_values, delta_values)
        return delta_values

    print('here starts optimization')
    # TODO solve pb with curve_fit returning zeros for delta when going through f_Ki
    print('final value detlaH : ',deltaMu_H(T=calculateTfromP(8.11, TfromP), P=8.11, components={component_pure.formula : component_pure}, structure_cavities=structure.cavities, bips=bips))
    popt, pcov = curve_fit(f_Ki, xdata=[item[0] for item in xy_P], ydata=[item[1] for item in xy_P], p0=[sigma, epsilon])
    print('popt etc : ', popt)
    return popt

# TODO see above
def optimisationKiharafromP(P1, P2, calculate_unknownTfromP, calculatePfromT, allPTinterpol, component_pure, structure, bips, list_models, n_P = 10):
    sigma, epsilon = component_pure.sigma, component_pure.epsilon
    PfromT = allPTinterpol[component_pure.formula][0][0]
    TfromP = allPTinterpol[component_pure.formula][1][0]
    xy_T = []
    i = 0
    if structure.id == 'II':
        i = 1
    for j in range(0, n_P + 1):
        for model_values in list_models.values():
            prev_deltaMu0, prev_deltaH0 = structure.deltaMu0, structure.deltaH0
            structure.deltaMu0, structure.deltaH0 = float(model_values[i]['deltaMu0']), float(model_values[i]['deltaH0'])
            xy_T += [ calculateTi(Pi = P1 + j*(P2 - P1)/n_P, Texpi= calculate_unknownTfromP(P1 + j*(P2 - P1)/n_P, TfromP), structurei=structure, componentsi={component_pure.formula : component_pure}, bipsi=bips) ]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of pres
    structure.deltaMu0, structure.deltaH0 = prev_deltaMu0, prev_deltaH0

    def f(T, eps, sig):
        component_pure.epsilon = eps
        component_pure.sigma = sig
        return deltaMu_H(P=calculatePfromT(T, PfromT), T=T, components={component_pure.formula : component_pure}, structure_cavities=structure.cavities, bips=bips)
    popt, pcov = curve_fit(f, xdata=[item[0] for item in xy_T], ydata=[item[1] for item in xy_T], p0=[sigma, epsilon])
    return popt

## Determination of hydrate composition xj_H
def xj_H(structure_cavities, components, thetas: dict, component_keyj: int):      # theta = dict[tuple] or dict[list]
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


### TESTS


T_test = 280
P_test = 5.2E6
y0_test = 0.4
y1_test = 0.6
results_test = {'Components' : [], 'Composition' : [], 'Temperature' : -1, 'Pressure' : -1, 'Structure' : '', 'Thetas' : []}
# cavities_test = [hydinter.Cavity('6^1',3.91* 1E-10, 20, 16), hydinter.Cavity('6^2',4.33E-10, 24, 8)]
# structure_test = hydinter.Structure('I', 1299, 1861, -38.12, 0.141, 3, cavities_test[0], cavities_test[1])
# components_test = {'CH4' : hydinter.Component('CH4', 0, y0_test, 190.56, 4.599E6, 127.426, 3.135E-10, 0.3526E-10, 32, 15.826277, -1559.0631, 0.0115),
#                    'CO2' : hydinter.Component('CO2', 1, y1_test, 304.19, 7.342E6, 168.77, 2.9818E-10, 0.6805E-10, 32, 14.283146, -2050.3269, 0.2276)
#                    }

# print(components_test)
# print(structure_test)
# main(T=T_test, P=P_test, structure=structure_test, components=components_test, bips=bips_test, results=results_test)
# print(results_test)

# print('langmuir : ', langmuir_ij(T_test, structure_test.cavities, components_test, 0, 'CH4'))

# print(deltaMu_L(T_test, P_test, structure_test, components_test, bips_test))

# R en metres
# k en m kg s K
# P et fugacite en Pa

T_test = 280
a = 0.3834E-10
sigma = 3.165E-10
epsilonk = 154.54       # = epsilon / k
R = 3.911E-10
z = 20

# def integrand(r):
#     def w_r_test(r):
#         # definition of delta_N
#         def delta_N_test(N):
#             # print('delta : ', ((1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
#             # print ('delta : ', ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
#             # print("sigma12 : ", (sigma**12)/(r * R**11) , "sigma5 : ", (sigma** 6 / (r * R** 5)))
#             return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

#         return ( 2 * z * epsilonk * K_BOLTZMANN**0
#                 * ( (sigma**12 / (r * R**11)) * (delta_N_test(10) + delta_N_test(11) * a/R )
#                 -   (sigma** 6 / (r * R** 5)) * (delta_N_test(4) + delta_N_test(5) * a/R ) )
#         )
#     print(w_r_test(r) / (T_test))
#     print((r**2)* ((1 - w_r_test(r) / (T_test) + (1/2)*(w_r_test(r)**2 / (T_test)**2))**(1/K_BOLTZMANN**0)), 1 - w_r_test(r) / (T_test) + (1/2)*(w_r_test(r)**2 / (T_test)**2), r**2)
#     print((r**2)* ((np.exp(- w_r_test(r) / (T_test)))**(1/K_BOLTZMANN**0)), np.exp(- w_r_test(r) / (T_test)), r**2)
#     return (r**2)* ((np.exp(- w_r_test(r) / (T_test)))**(1/K_BOLTZMANN**0))

# print(4 * pi * integrate.quad(integrand, 0, R)[0]  / K_BOLTZMANN* T_test)


def w_r_test(r):
        # definition of delta_N
        def delta_N_test(N):
            # print('delta : ', ((1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
            # print ('delta : ', ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
            # print("sigma12 : ", (sigma**12)/(r * R**11) , "sigma5 : ", (sigma** 6 / (r * R** 5)))
            return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

        return ( 2 * z * epsilonk * K_BOLTZMANN**1
                * ( (sigma**12 / (r * R**11)) * (delta_N_test(10) + delta_N_test(11) * a/R )
                -   (sigma** 6 / (r * R** 5)) * (delta_N_test(4) + delta_N_test(5) * a/R ) )
        )

p = ( (sigma/R)**6 ) * ( (1 - a/R)**(-6) )
q = ( (sigma/R)**6 ) * ( (1 - a/R)**(-7) ) * a/R
r1 = -1
s = - ( (1 - a/R)**(-1) ) * a/R
k = 1 / (R-a)
l = 2 * ( (sigma**6) / (R**5)) * ( (1 - a/R)**(-4) ) * k
A = 2 * z * epsilonk * K_BOLTZMANN * l * (p + q + r1 + s)
B = 2 * z * epsilonk * K_BOLTZMANN * l * (44*p + 52*q + 15*r1 + 21*s)
C = np.exp(- A / (K_BOLTZMANN*T_test))
D = - B / (K_BOLTZMANN*T_test)
def w2_test(r):
    return A + B/(r**2)

# x = np.linspace(1E-13, R, 5000)
# # plt.plot(x, np.exp(w_r_test(x)))
# # plt.plot(x, (np.exp(w_r_test(x)))**(1))
# fig, (ax1, ax2) = plt.subplots(1, 2)
# # ax1.plot(x, -w_r_test(x)/(T_test*K_BOLTZMANN**1))
# ax1.plot(x, w_r_test(x))
# ax2.plot(x, w2_test(x))
# # ax2.plot(x, (np.exp(-w_r_test(x)))**(1/(T_test*K_BOLTZMANN**1)))
# # ax3.plot(x, (np.exp(-w_r_test(x)/(T_test*K_BOLTZMANN**1))))
# # ax3.plot(x, (1 - w_r_test(x) + (w_r_test(x)**2)/2)**(1/(K_BOLTZMANN**2)))
# plt.show()

N = 5
def delta_N_test(r, N):
    # print('delta : ', ((1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
    # print ('delta : ', ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N)
    # print("sigma12 : ", (sigma**12)/(r * R**11) , "sigma5 : ", (sigma** 6 / (r * R** 5)))
    return ( (1 - r/R - a/R)**(-N) - (1 + r/R - a/R)**(-N) ) / N

def delta_test2(r, N):
    return 2 * ( (1 - a/R)**(-N)) * (1 / (R - a)) * r

# x = np.linspace(1E-13, R, 5000)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# # ax1.plot(x, -w_r_test(x)/(T_test*K_BOLTZMANN**1))
# ax1.plot(x, delta_N_test(x, 5))
# ax2.plot(x, delta_test2(x, 5))

# x = np.linspace(0, 1E-9, 100)
# plt.plot(x, delta_N_test(x))
# plt.show()