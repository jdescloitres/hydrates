# Hydrate Composition Calculation Algorithm with Interface


import hydrate_algorithm_no_interface as calc
import interface
from scipy.optimize import newton, curve_fit
from scipy.interpolate import interp1d

max_eps = 1E-5

class Cavity:

    def __init__(self, ray, coord_z, pop_nu) -> None:
        self.r = ray
        self.z = coord_z
        self.nu = pop_nu

class Structure:

    def __init__(self, struct_id : str, deltaMu0, deltaH0, deltaCp0, b, deltaV0, petite_cavite: Cavity, grande_cavite: Cavity) -> None:
        self.id = struct_id
        self.deltaMu0 = deltaMu0
        self.deltaH0 = deltaH0
        self.deltaCp0 = deltaCp0
        self.b = b
        self.deltaV0 = deltaV0
        self.cavities = [petite_cavite, grande_cavite]

class Component:

    def __init__(self, name: str, component_id: int, y, Tc, Pc, epsilon, sigma, a, Vinf, k1, k2, omega) -> None:
        self.name = name
        self.id = component_id
        self.y = y
        self.tc = Tc
        self.pc = Pc
        self.epsilon = epsilon
        self.sigma = sigma
        self.a = a
        self.vinf = Vinf
        self.k1 = k1
        self.k2 = k2
        self.omega = omega

components = []
list_temp = []
list_pres = []

def PfromT_f(list_temp, list_pres):
    return interp1d(x= list_temp, y= list_pres)
PfromT = PfromT_f(list_temp = list_temp, list_pres=list_pres)
def calculatePfromT(T):
    return PfromT._call(T)

def TfromP_f(list_pres, list_temp):
    return interp1d(x= list_pres, y= list_temp)
TfromP = TfromP_f(list_pres=list_pres, list_temp = list_temp)
def calculateTfromP(P):
    return PfromT._call(P)


def calculatePi(Ti, Pexpi, structurei, componentsi, coefficientsi):
    # determine le Pi qui verifie l'egalite des deltaMu
    def f(P, T = Ti, structure = structurei, components=componentsi,coefficients= coefficientsi):
        return calc.deltaMu_H(T, P, structure[calc.KEY_FOR_CAVITIES], components, coefficients) - calc.deltaMu_L(T, P, structure, components, coefficients)
    P_i = newton(func= f, x0=Pexpi)
    return (P_i, calc.deltaMu_L(Ti, P_i, structurei, componentsi, coefficientsi))


# TODO if it is the first time said component is being used (note: the optimization is done for that component as sole component of the gas)
def optimisationKihara(T1, T2, component_pure, structure, coefficients, kw, file, n_models= 3, n_T = 10):
    sigma, epsilon, a = file['parameters']

    xy = []
    for j in range(1, n_T + 1):
        xy += xy + [(calculatePi(T1 + j*(T2 - T1)/n_T, calculatePfromT(T1 + j*(T2 - T1)/n_T), kw[i])) for i in range(n_models)]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of temp

    # note: unlike deltaMu_L, deltaMu_H does not depend on the macroscopic values, so it is independent from the models that were used to get Pi
    def f(P, eps, sig):
        component_pure[calc.KEY_FOR_EPSILON_OVER_K] = eps
        component_pure[calc.KEY_FOR_SIGMA] = sig
        return calc.deltaMu_H(T=calculateTfromP(P), P=P, components=component_pure, structure_cavities=structure[calc.KEY_FOR_CAVITIES], coefficients=coefficients)

    popt, pcov = curve_fit(f, xdata=[item[0] for item in xy], ydata=[item[1] for item in xy], p0=[sigma, epsilon])

    # on peut faire la meme chose en faisant varier les P:
    # xy_T = []
    # for j in range(1, n_P + 1):
    #     xy_T += xy_T + [(calculateTi(P1 + j*(P2 - P1)/n_P, calculateTfromP(P1 + j*(P2 - P1)/n_P), kw[i])) for i in range(n_models)]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of pres
    # def f(T, eps, sig):
    #     component_pure[calc.KEY_FOR_EPSILON_OVER_K] = eps
    #     component_pure[calc.KEY_FOR_SIGMA] = sig
    #     return calc.deltaMu_H(P=calculatePfromT(T), T=T, components=component_pure, structure_cavities=structure[calc.KEY_FOR_CAVITIES], coefficients=coefficients)
    # potp_T = curve_fit(f, xdata=[item[0] for item in xy_T], ydata=[item[1] for item in xy_T], p0=[sigma, epsilon])

    return popt

# to put inside a class maybe
def main(f):
    # get data from files

    # initialize Structure objects
    structures = [ Structure(petite_cavite=Cavity(), grande_cavite=Cavity()) , Structure(petite_cavite=Cavity(), grande_cavite=Cavity()) ]

    # initialize Component objects
    file = f        # explore all lines of file
    components = [Component(line['name'], line['etc'], y=1) for line in range(len(file))]
    for component in components:
        sig, eps = optimisationKihara(component_pure=component)
        component.epsilon = eps
        component.sigma = sig
        component.y = 0.0

    # creation of the interface with the new run function that is created in this file
    # ==> all actions are now answered by in the interface methods
    inter = interface.Hydrate_interface(componentsList=components, structuresList=structures)

    pass

# inside new interface class inheriting interface
def run(self):
    ### differenciate depending on components
    # TODO
    #

    ### fill in results array
    update_results()

    ### usr results to fill the resultsTree
    # if new tree, add it to list of existing trees



def update_results(self):

    # TODO replace arguments with actual function arguments
    ### fill in results array for the given data (which will in turn be used to fill the results tree)
    T = self.T
    components = self.components

    self.results['Components'] = [component.name for component in components]
    self.results['Composition'] = [ component.y for component in components ]
    self.results['Temperature'] = T

    structures = self.structures
    coefficients = self.coefficients

    ### calculate the rest of the values we need
    ## estimate P and suppose Structure is I
    PeqI = calculatePi(T)
    structureI = structures['I']
    cijI = {component.name : calc.langmuir_ij() for component in components}
    fjI = {component.name : calc.fugacity_j() for component in components}

    # deltaMuHI = calc.deltaMu_H()
    # deltaMuLI = calc.deltaMu_L()
    # change P until the difference is small enough
    # while abs(PeqI - calculatePi(T)) > max_eps:
    #     PeqI = calculatePi()
    #     cijI = {component.name : calc.langmuir_ij() for component in components}
    #     fjI = {component.name : calc.fugacity_j() for component in components}
        # deltaMuHI = calc.deltaMu_H()
        # deltaMuLI = calc.deltaMu_L()

    ## repeat for Str II and determine which corresponds
    # suppose Structure is II
    PeqII = calculatePi()
    structureII = structures['II']
    cijII = {component.name : (calc.langmuir_ij(0), calc.langmuir_ij(1)) for component in components}
    fjII = {component.name : calc.fugacity_j() for component in components}

    # determine which structure
    if PeqI <= PeqII:
        Peq = PeqI
        struct = structureI
        cij = cijI
        fj = fjI
    else:
        Peq = PeqII
        struct = structureII
        cij = cijII
        fj = fjII

    # calculate thetasj and xj

    # thetas = {(calc.theta_ij(T, Peq, struct[calc.KEY_FOR_CAVITIES], components, coefficients, 0, component.id),
    #           calc.theta_ij(T, Peq, struct[calc.KEY_FOR_CAVITIES], components, coefficients, 1, component.id)) for component in components}

    # def theta_ji(cij_list, fj_list, cavity, component_name)
    thetas = {component.name : (calc.theta_ij(cij, fj, struct[calc.KEY_FOR_CAVITIES][0], component.name),
              calc.theta_ij(cij, fj, struct[calc.KEY_FOR_CAVITIES][1], component.name)) for component in components}
    xj = {component.name : calc.xj_H(structure_cavities=struct[calc.KEY_FOR_CAVITIES], components=components, thetas=thetas, j=component.name) for component in components}

    # fill in results array for the calculated data
    self.results['Hydrate Composition'] = [ xj[component.name] for component in components ]
    self.results['Pressure'] = Peq
    self.results['Structure'] = struct.id
    self.results['Thetas'] = [ thetas[component.name] for component in components]
    thetas_Sall = thetas_iall(thetas=thetas, components=components, structure_cavities=struct[calc.KEY_FOR_CAVITIES])[0]
    thetas_Lall = thetas_iall(thetas=thetas, components=components, structure_cavities=struct[calc.KEY_FOR_CAVITIES])[1]
    theta_tot =( (thetas_Sall * struct[calc.KEY_FOR_CAVITIES][0].nu + thetas_Lall * struct[calc.KEY_FOR_CAVITIES][1].nu)
                        / (struct[calc.KEY_FOR_CAVITIES][0].nu + struct[calc.KEY_FOR_CAVITIES][1].nu)
    )


def thetas_iall(thetas, components):
    sumthetaS = 0
    sumthetaL = 0
    for component in components:
        sumthetaS += thetas[component.name][0]
        sumthetaL += thetas[component.name][1]
    return sumthetaS, sumthetaL




# f_x = calc.deltaMu_L(T = calc.T_test, P = calc.P_test, structure=calc.structure_test, components=calc.components_test,coefficients= calc.coefficients_test)

# def f(P, T = calc.T_test, structure=calc.structure_test, components=calc.components_test,coefficients= calc.coefficients_test):
#     return calc.deltaMu_L(P=P, T=T, structure=structure, components=components,coefficients= coefficients) - f_x
# x = newton(func= f, x0=0.09)

# print(f_x, calc.P_test, x)

# n_T = 3
# T1 = 250
# T2 = 300
# xy = []
# for j in range(n_T + 1):
#     xy += [(T1 + j*(T2 - T1)/n_T, 2* (T1 + j*(T2 - T1)/n_T)) for i in range(1, 2)]
# print([x[0] for x in xy], [y[1] for y in xy])