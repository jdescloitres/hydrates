# Hydrate Composition Calculation Algorithm with Interface


import csv
import re
import tkinter as tk
from tkinter import messagebox
import hydrate_algorithm_no_interface as calc
import interface
from scipy.optimize import newton, curve_fit
from scipy.interpolate import interp1d

MAX_EPS = 1E-5

class Cavity:

    def __init__(self, name, ray, coord_z, pop_nu) -> None:
        self.name = name
        self.r = ray
        self.z = coord_z
        self.nu = pop_nu

class Structure:

    def __init__(self, struct_id : str, deltaMu0, deltaH0, deltaCp0, b, deltaV0, small_cavity: Cavity, big_cavity: Cavity) -> None:
        self.id = struct_id
        self.deltaMu0 = deltaMu0
        self.deltaH0 = deltaH0
        self.deltaCp0 = deltaCp0
        self.b = b
        self.deltaV0 = deltaV0
        self.cavities = [small_cavity, big_cavity]

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


def calculatePfromT(T, PfromT):
    return PfromT._call(self=PfromT, x_new= T)

def calculateTfromP(P, TfromP):
    return TfromP._call(self = TfromP, x_new=P)

def getInterpolFuncs(compo_foldername):
    all_rep_str = compo_foldername + 'all_repertories.csv'
    with open(all_rep_str, mode='r') as infile:
        reader = csv.reader(infile)
        all_rep_list = [rows[0] for rows in reader]

    list_temp=[]
    list_pres=[]
    for paper in all_rep_list:
        filename = compo_foldername + paper + '.csv'
        with open(filename, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            properties = [(float(rows[0]), float(rows[1])) for rows in reader]
            list_temp += [item[0] for item in properties]
            list_pres += [item[1] for item in properties]

    PfromT = calc.PfromT_f(list_temp = list_temp, list_pres=list_pres)
    TfromP = calc.TfromP_f(list_pres = list_pres, list_temp = list_temp)
    return (PfromT, TfromP)

def main():
    # TODO get data from files
    # first, get PT methods for pure gases
    allInterpolPT = {}

    folder_name = DATA_FOLDER + PT_FOLDER + '1CondensableElements/'
    all_repertories_str = folder_name + 'all_repertories.txt'
    with open(all_repertories_str, mode = 'r') as infile:
        all_repertories_list = [line for line in infile.read().splitlines()]

    for compo in all_repertories_list:
        compo_foldername = folder_name + compo + '/'
        allInterpolPT[compo] = getInterpolFuncs(compo_foldername)

    # get all coefficients
    bip = {}
    bip_file = DATA_FOLDER + BIP_FOLDER + 'BIP.csv'
    with open(bip_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        all_lines = [row for row in reader]
        for row in all_lines:
            for col in list(all_lines[0].keys())[1:]:
                if col != 'ref':
                    bip[tuple(sorted((row['formula'],col)))] = float(row[col])

    # initialize Cavity objects
    cavitiesI_file = DATA_FOLDER + CAVITIES_FOLDER + 'structureI.csv'
    with open(cavitiesI_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        cavitiesI_properties = [row for row in reader]
    cavitiesI = [Cavity(name=cavitiesI_properties[0]['name'],
                        ray=cavitiesI_properties[0]['r'],
                        coord_z=cavitiesI_properties[0]['z'],
                        pop_nu=cavitiesI_properties[0]['nu']),
                Cavity(name=cavitiesI_properties[1]['name'],
                        ray=cavitiesI_properties[1]['r'],
                        coord_z=cavitiesI_properties[1]['z'],
                        pop_nu=cavitiesI_properties[1]['nu'])
                ]

    cavitiesII_file = DATA_FOLDER + CAVITIES_FOLDER + 'structureII.csv'
    with open(cavitiesII_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        cavitiesII_properties = [row for row in reader]
    cavitiesII = [Cavity(name=cavitiesII_properties[0]['name'],
                        ray=cavitiesII_properties[0]['r'],
                        coord_z=cavitiesII_properties[0]['z'],
                        pop_nu=cavitiesII_properties[0]['nu']),
                Cavity(name=cavitiesII_properties[1]['name'],
                        ray=cavitiesII_properties[1]['r'],
                        coord_z=cavitiesII_properties[1]['z'],
                        pop_nu=cavitiesII_properties[1]['nu'])
                ]
    print(cavitiesI, cavitiesII)

    # get all models for Structure objects
    all_rep_str = DATA_FOLDER + STRUCTURES_FOLDER + 'all_repertories.csv'
    with open(all_rep_str, mode='r') as infile:
        reader = csv.reader(infile)
        all_rep_list = [rows[0] for rows in reader]
    all_models = {}
    for model in all_rep_list:
        filename = DATA_FOLDER + STRUCTURES_FOLDER + model + '.csv'
        with open(filename, mode='r') as infile:
            reader = csv.DictReader(infile, skipinitialspace=True)
            model_param = [row for row in reader]
        all_models[model] = model_param


    # initialize Structure objects

    # structure_file = DATA_FOLDER + STRUCTURES_FOLDER + 'Handa_and_Tse.csv'
    # with open(structure_file, mode='r') as infile:
    #     reader = csv.DictReader(infile, skipinitialspace=True)
    #     struct_properties = [row for row in reader]
    struct_properties = all_models[CHOSEN_MODEL]
    structures = {struct_properties[0]['id'] : Structure(struct_id=struct_properties[0]['id'],
                                                deltaMu0=float(struct_properties[0]['deltaMu0']),
                                                deltaH0=float(struct_properties[0]['deltaH0']),
                                                deltaCp0=float(struct_properties[0]['deltaCp0']),
                                                b=float(struct_properties[0]['b']),
                                                deltaV0=float(struct_properties[0]['deltaV0']),
                                                small_cavity=cavitiesI[0],
                                                big_cavity=cavitiesI[1]),
                struct_properties[1]['id'] : Structure(struct_id=struct_properties[1]['id'],
                                                deltaMu0=float(struct_properties[1]['deltaMu0']),
                                                deltaH0=float(struct_properties[1]['deltaH0']),
                                                deltaCp0=float(struct_properties[1]['deltaCp0']),
                                                b=float(struct_properties[1]['b']),
                                                deltaV0=float(struct_properties[1]['deltaV0']),
                                                small_cavity=cavitiesII[0],
                                                big_cavity=cavitiesII[1])
                    }


    # initialize Component objects
    components_file = DATA_FOLDER + COMPONENTS_FOLDER + 'Sloan.csv'
    with open(components_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        compo_properties = [row for row in reader]
    all_components = {compo_properties[row]['name'] : Component(name=compo_properties[row]['name'],
                                                            component_id=compo_properties[row]['id'],
                                                            y=0.0,
                                                            Tc=compo_properties[row]['Tc'],
                                                            Pc=compo_properties[row]['Pc'],
                                                            epsilon=compo_properties[row]['epsilon'],
                                                            sigma=compo_properties[row]['sigma'],
                                                            a=compo_properties[row]['a'],
                                                            Vinf=compo_properties[row]['Vinf'],
                                                            k1=compo_properties[row]['k1'],
                                                            k2=compo_properties[row]['k2'],
                                                            omega=compo_properties[row]['omega'])
                for row in range(len(compo_properties))}

    # first optimization of Kihara parameters
    # for component in all_components.values():
    #     epsilon, sigma = calc.optimisationKiharafromP(component_pure=component, all_models)
    #     component.epsilon = epsilon
    #     component.sigma = sigma

    # creation of the interface with the new run function that is created in this file
    inter = Hydrate_interface(componentsDict=all_components, structuresDict=structures, interpolPT = allInterpolPT, bip = bip, all_models = all_models)

class Hydrate_interface(interface.Hydrate_interface_squelette):
    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self):
        ### check if all provided information is correct
        # if no components have been selected, returns error
        if len(self.tree.get_children()) == 0:
            messagebox.showerror(title='Error', message = 'Please select the gas components')
            return

        # if no temperature and no pressure has been set, returns error
        if (self.results['Temperature'] < 0 and self.results['Pressure'] < 0
            or self.results['Temperature'] < 0 and self.checkPres_var.get() == 0
            or self.checkTemp_var.get() == 0 and self.results['Pressure'] < 0
            or self.checkTemp_var.get() == 0 and self.checkPres_var.get() == 0
        ):
            messagebox.showerror(title='Error', message = 'Please set the temperature or the pressure')
            return

        # if the sum of all mole fractions of the gas isn't 1, returns error
        sumx = 0
        for item in self.tree.get_children():
            sumx += float(self.tree.item(item)['values'][0])
        if sumx != 1:
            messagebox.showerror(title='Error', message = 'Total mole fraction should be 1. \n Please check composition.')
            return

        ### fill in results array
        TisCalculated = False
        PisCalculated = False
        StrisCalculated = False
        if self.checkTemp_var.get() == 0 or self.results['Temperature'] < 0:
            TisCalculated = True
        if self.checkPres_var.get() == 0 or self.results['Pressure'] < 0:
            PisCalculated = True
        if self.checkStruct_var.get() == 0 or self.results['Structure'] == '':
            StrisCalculated = True

        theta_tot, thetas_Sall, thetas_Lall = self.update_results(PisCalculated, TisCalculated, StrisCalculated)
        self.results['Thetas_tots'] = [theta_tot, thetas_Sall, thetas_Lall]
        small_cavity, big_cavity = self.structuresDict[self.results['Structure']].cavities

        tree_components = tuple(self.results['Components'])

        # create here the new interpol functions for gas with multiple components
        # may not be useful bc we only need it when optimizing parameters ==> done with "pure" gas
        # if tree_components not in self.PTinterpolDict:
        #     parent_folder = DATA_FOLDER + PT_FOLDER + f'{len(tree_components)}CondensableElements/'
        #     compo_name = tree_components[0]
        #     if len(tree_components) >0 :
        #         for i in range(1, len(tree_components)):
        #             compo_name += '=' + tree_components[i]
        #             print(compo_name)
        #     compo_folder_name = parent_folder + compo_name +'/'
        #     print(compo_folder_name)
        #     self.PTinterpolDict[tree_components]=getInterpolFuncs(compo_foldername = compo_folder_name)
        # print(calculatePfromT(275, self.PTinterpolDict[tree_components][0]))

        ### differenciate how the tree is build depending on components
        # if this is the first round, i.e. if the results tree was empty before, the first tree is used
        if len(self.all_trees) == 1 and len(list(self.all_trees.values())[0][0].get_children()) == 0:
            columnsTemp = self.column_names_ini
            treeTemp, buttonTemp, scrollbarxTemp, scrollbaryTemp = list(self.all_trees.values())[0]
            frameTemp = list(self.all_treeFrames.values())[0]
            tree_components = tuple(self.results['Components'])
            self.all_trees.clear()
            self.all_treeFrames.clear()
            self.all_trees[tree_components] = [treeTemp, buttonTemp, scrollbarxTemp, scrollbaryTemp]
            self.all_treeFrames[tree_components] = frameTemp

        # if the combination of components has been used previously, we use that tree, and no need to change columns
        elif tree_components in self.all_trees:
            treeTemp = self.all_trees[tree_components][0]
            self.addRowtoTree(treeTemp)
            # reset data that were calculated, not given
            if TisCalculated:
                self.results['Temperature'] = -1
            if PisCalculated:
                self.results['Pressure'] = -1
            if StrisCalculated:
                self.results['Structure'] = ''
            return
        # and if not, a new tree is created
        elif tree_components not in self.all_trees:
            columnsTemp = Hydrate_interface.column_names_ini.copy()
            treeTemp, buttonTemp, scrollbarx, scrollbary, frameTemp = self.makeNewResultsTree(self.frameR, columnsTemp)
            self.all_trees[tree_components] = [treeTemp, buttonTemp, scrollbarx, scrollbary]
            self.all_treeFrames[tree_components] = frameTemp


        ### use results to fill the resultsTree
        # initialize the column names
        columnsTemp = Hydrate_interface.column_names_ini.copy()

        # update the lists of column and values
        values_item = [self.results['Temperature'], self.results['Pressure'], self.results['Structure']]
        l = len(self.tree.get_children())
        values_tocc = [ [''] * (l*2 + 2) for i in range(len(self.tree.get_children())) ]

        for component_index in range(len(self.results['Components'])):
            if component_index > 0:
                columnsTemp.insert(component_index-1, f'y{component_index}')
                columnsTemp.insert(-3, f'compo{component_index}')

                values_item.insert(component_index-1, self.results['Composition'][component_index])
            values_item += [self.results['Hydrate Composition'][component_index]]

            values_tocc[component_index] += [self.results['Components'][component_index], self.results['Thetas'][component_index][0], self.results['Thetas'][component_index][1]]

        values_item += self.results['Thetas_tots']

        # configure and fill the tree
        treeTemp.configure(columns=columnsTemp)

        for component_index in range(len(self.results['Components'])):
            name = self.results['Components'][component_index]
            if component_index == 0:
                treeTemp.heading('#0', text = f'y ({name})')
                treeTemp.column('#0', width = 75)
            else:
                treeTemp.heading(f'y{component_index}', text = f'y ({name})')
                treeTemp.column(f'y{component_index}', width = 75)
            treeTemp.heading(f'compo{component_index}', text = f'x_H ({name})')
            treeTemp.column(f'compo{component_index}', width = 75)

        treeTemp.heading("temp", text = "T (K)")
        treeTemp.heading("pres", text = "Peq (Pa)")
        treeTemp.heading("struct", text = "Structure")
        treeTemp.heading("tocc_tot", text = "Total Occ")
        treeTemp.heading("toccS_tot", text = f"{small_cavity.name}")
        treeTemp.heading("toccL_tot", text = f"{big_cavity.name}")
        treeTemp.column("temp", width = 35)
        treeTemp.column("pres", width = 75)
        treeTemp.column("struct", width = 60)
        treeTemp.column("tocc_tot", width = 75)
        treeTemp.column("toccS_tot", width = 75)
        treeTemp.column("toccL_tot", width = 75)

        treeTemp.configure(displaycolumns=columnsTemp)

        treeTemp.insert(parent = "", index= tk.END, iid=f'{len(treeTemp.get_children())}', text = self.results['Composition'][0], values= values_item)
        for component_index in range(len(self.results['Components'])):
            treeTemp.insert(parent = f'{len(treeTemp.get_children())-1}', index= tk.END, text = '', values= values_tocc[component_index])

        # reset data that were calculated, not given
        if TisCalculated:
            self.results['Temperature'] = -1
        if PisCalculated:
            self.results['Pressure'] = -1
        if StrisCalculated:
            self.results['Structure'] = ''


    def update_results(self, PisCalculated : bool, TisCalculated: bool, StrisCalculated: bool):

        # TODO replace arguments with actual function arguments
        components = {}
        for item in self.tree.get_children():
            components[self.tree.item(item)['text']] = self.allComponents[self.tree.item(item)['text']]
            components[self.tree.item(item)['text']].y = self.tree.item(item)['values'][0]
        structures = self.structuresDict
        bip = self.bipDict

        self.results['Components'] = [components[component_key].name for component_key in sorted(components)]
        self.results['Composition'] = [ components[component_key].y for component_key in sorted(components)]

        ### calculate the rest of the values we need
        # if P is not given
        if PisCalculated:
            Peq = 'Pcalc'
            # Teq = self.results['Temperature']
            # # if the structure used is known, we can directly get the corresponding values
            # if not StrisCalculated:
            #     Peq = calc.calculatePi(Teq)
            #     struct = self.results['Structure']
            #     cijI = {component_key : calc.langmuir_ij() for component_key in sorted(components)}
            #     fjI = {component_key : calc.fugacity_j() for component_key in sorted(components)}
            #     self.results['Pressure'] = Peq

            # # if not, we need to determine the more adequate structure
            # else:
            #     # estimate P suppose Structure is I
            #     PeqI = calc.calculatePi(Teq)
            #     structureI = structures['I'].id
            #     cijI = {component_key : calc.langmuir_ij() for component_key in sorted(components)}
            #     fjI = {component_key : calc.fugacity_j() for component_key in sorted(components)}

            #     # repeat for Str II
            #     PeqII = calc.calculatePi(Teq)
            #     structureII = structures['II'].id
            #     cijII = {component_key : (calc.langmuir_ij(0), calc.langmuir_ij(1)) for component_key in sorted(components)}
            #     fjII = {component_key : calc.fugacity_j() for component_key in sorted(components)}

            #     # determine which structure corresponds best to equilibrium
            #     if PeqI <= PeqII:
            #         Peq = PeqI
            #         struct = structureI
            #         cij = cijI
            #         fj = fjI
            #     else:
            #         Peq = PeqII
            #         struct = structureII
            #         cij = cijII
            #         fj = fjII
            self.results['Pressure'] = Peq
            #     self.results['Structure'] = struct


        # if T is not given
        elif TisCalculated:
            Teq = 'Tcalc'
            # Peq = self.results['Pressure']
            # # if the structure used is known, we can directly get the corresponding values
            # if not StrisCalculated:
            #     Teq = calc.calculateTi(Peq)
            #     struct = self.results['Structure']
            #     cijI = {component_key : calc.langmuir_ij() for component_key in sorted(components)}
            #     fjI = {component_key : calc.fugacity_j() for component_key in sorted(components)}

            # # if not, we need to determine the more adequate structure
            # else:
            #     # estimate T suppose Structure is I
            #     TeqI = calc.calculateTi(Peq)
            #     structureI = structures['I'].id
            #     cijI = {component_key : calc.langmuir_ij() for component_key in sorted(components)}
            #     fjI = {component_key : calc.fugacity_j() for component_key in sorted(components)}

            #     # repeat for Str II
            #     TeqII = calc.calculateTi(Peq)
            #     structureII = structures['II'].id
            #     cijII = {component_key : (calc.langmuir_ij(0), calc.langmuir_ij(1)) for component_key in sorted(components)}
            #     fjII = {component_key : calc.fugacity_j() for component_key in sorted(components)}

            #     # determine which structure corresponds best to equilibrium
            #     if TeqI <= TeqII:
            #         Teq = TeqI
            #         struct = structureI
            #         cij = cijI
            #         fj = fjI
            #     else:
            #         Teq = TeqII
            #         struct = structureII
            #         cij = cijII
            #         fj = fjII
            self.results['Temperature'] = Teq
            #     self.results['Structure'] = struct

        else :
            # Peq = self.P
            # Teq = self.T
            Peq = self.results['Pressure']
            Teq = self.results['Temperature']
            # TODO how do we determine which structure to use in this case ?
        struct = 'I'

        # calculate thetasj and xj
        # thetas = {component_key : (calc.theta_ij(cij, fj, struct.cavities, 0, component_key),
        #         calc.theta_ij(cij, fj, struct.cavities, 1, component_key)) for component_key in sorted(components)}
        # xj = {component_key : calc.xj_H(structure_cavities=struct.cavities, components=components, thetas=thetas, component_keyj=component_key) for component_key in sorted(components)}

        # fill in results array for the calculated data
        self.results['Hydrate Composition'] = [ 'xj' + str(component_key) for component_key in sorted(components) ]
        # self.results['Hydrate Composition'] = [ xj[component_key] for component_key in sorted(components) ]
        self.results['Structure'] = struct
        self.results['Thetas'] = [('small', 'large') for component_key in sorted(components)]
        # self.results['Thetas'] = [ thetas[component_key] for component_key in sorted(components)]
        # thetas_Sall = calc.thetas_iall(thetas=thetas, components=components, structure_cavities=struct.cavities)[0]
        # thetas_Lall = calc.thetas_iall(thetas=thetas, components=components, structure_cavities=struct.cavities)[1]
        # theta_tot =( (thetas_Sall * struct.cavities[0].nu + thetas_Lall * struct.cavities[1].nu)
        #                     / (struct.cavities[0].nu + struct.cavities[1].nu)
        # )


        # return theta_tot, thetas_Sall, thetas_Lall
        return 'theta_tot', 'thetas_Sall', 'thetas_Lall'


    def addRowtoTree(self, tree):
        values_item = [self.results['Temperature'], self.results['Pressure'], self.results['Structure']]
        l = len(self.tree.get_children())
        values_tocc = [ [''] * (l*2 + 2) for i in range(len(self.tree.get_children())) ]

        for component_index in range(len(self.results['Components'])):
            if component_index > 0:
                values_item.insert(component_index-1, self.results['Composition'][component_index])
            values_item += [self.results['Hydrate Composition'][component_index]]
            values_tocc[component_index] += [self.results['Components'][component_index], self.results['Thetas'][component_index][0], self.results['Thetas'][component_index][1]]

        values_item += self.results['Thetas_tots']

        tree.insert(parent = "", index= tk.END, iid=f'{len(tree.get_children())}', text = self.results['Composition'][0], values= values_item)
        for j in range(len(self.results['Components'])):
            tree.insert(parent = f'{len(tree.get_children())-1}', index= tk.END, text = '', values= values_tocc[j])


    def optimizeKihara(self):
        list_models = self.list_models
        for item in self.tree.get_children():
            component = self.all_components[self.tree.item(item)['text']]
            # TODO how to choose P or T, and which bounds P1 P2 to choose ==> window comes on ?
            calc.optimisationKiharafromP(component_pure=component, kw=list_models)
            calc.optimisationKiharafromT(component_pure=component)
            # if self.PisCalculated:
            #     pass
        # return super().optimizeKihara()


##### ALL THESE METHODS ARE NOW IN CALC
# def PfromT_f(list_temp, list_pres):
#     return interp1d(x= list_temp, y= list_pres)
# def TfromP_f(list_pres, list_temp):
#     return interp1d(x= list_pres, y= list_temp)
# def calculatePi(Ti, Pexpi, structurei, componentsi, coefficientsi):
#     # determine le Pi qui verifie l'egalite des deltaMu
#     def f(P, T = Ti, structure = structurei, components=componentsi,coefficients= coefficientsi):
#         return calc.deltaMu_H(T, P, structure.cavities, components, coefficients) - calc.deltaMu_L(T, P, structure, components, coefficients)
#     P_i = newton(func= f, x0=Pexpi)
#     return (P_i, calc.deltaMu_L(Ti, P_i, structurei, componentsi, coefficientsi))
# def optimisationKihara(T1, T2, component_pure : Component, structure, coefficients, kw, file, n_models= 3, n_T = 10):
#     sigma, epsilon, a = file['parameters']
#     xy = []
#     for j in range(1, n_T + 1):
#         xy = xy + [(calculatePi(T1 + j*(T2 - T1)/n_T, calculatePfromT(T1 + j*(T2 - T1)/n_T), kw[i])) for i in range(n_models)]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of temp
#     # note: unlike deltaMu_L, deltaMu_H does not depend on the macroscopic values, so it is independent from the models that were used to get Pi
#     def f(P, eps, sig):
#         component_pure.epsilon = eps
#         component_pure.sigma = sig
#         return calc.deltaMu_H(T=calculateTfromP(P), P=P, components={component_pure.name : component_pure}, structure_cavities=structure.cavities, coefficients=coefficients)
#     popt, pcov = curve_fit(f, xdata=[item[0] for item in xy], ydata=[item[1] for item in xy], p0=[sigma, epsilon])
#     # on peut faire la meme chose en faisant varier les P:
#     # xy_T = []
#     # for j in range(1, n_P + 1):
#     #     xy_T = xy_T + [(calculateTi(P1 + j*(P2 - P1)/n_P, calculateTfromP(P1 + j*(P2 - P1)/n_P), kw[i])) for i in range(n_models)]           # return list of points [ (x1, y1), (x2, y2), (x1, y1), (x2, y2) ] for all models for a range of pres
#     # def f(T, eps, sig):
#     #     component_pure.epsilon = eps
#     #     component_pure.sigma = sig
#     #     return calc.deltaMu_H(P=calculatePfromT(T), T=T, components={component_pure.name : component_pure}, structure_cavities=structure.cavities, coefficients=coefficients)
#     # potp_T = curve_fit(f, xdata=[item[0] for item in xy_T], ydata=[item[1] for item in xy_T], p0=[sigma, epsilon])
#     return popt
# def thetas_iall(thetas, components):
#     sumthetaS = 0
#     sumthetaL = 0
#     for component_key in sorted(components):
#         sumthetaS += thetas[components[component_key].name][0]
#         sumthetaL += thetas[components[component_key].name][1]
#     return sumthetaS, sumthetaL



# f_x = calc.deltaMu_L(T = calc.T_test, P = calc.P_test, structure=calc.structure_test, components=calc.components_test,coefficients= calc.coefficients_test)

# def f(P, T = calc.T_test, structure=calc.structure_test, components=calc.components_test,coefficients= calc.coefficients_test):
#     return calc.deltaMu_L(P=P, T=T, structure=structure, components=components,coefficients= coefficients) - f_x
# x = newton(func= f, x0=0.09)

# print(f_x, calc.P_test, x)

DATA_FOLDER = 'DataFiles/'
PT_FOLDER = 'EquilibriumDataPT/'
COMPONENTS_FOLDER = 'CondensableElementsProperties/'
BIP_FOLDER = 'CondensableElementsProperties/BIP/'
STRUCTURES_FOLDER = 'StructureReferenceProperties/'
CHOSEN_MODEL = 'Handa_and_Tse'
CAVITIES_FOLDER = 'StructureReferenceProperties/CavitiesInternalParameters/'

# componentsli={'H20' : Component('H20', 0, 0.0, 120, 2, 143, 0.3, 0.3, 200, 1220, -4000, 23),
#               'CH4' : Component('CH4', 1, 0.0, 120, 2, 143, 0.3, 0.3, 200, 1220, -4000, 23),
#               'C2H6' : Component('C2H6', 2, 0.0, 120, 2, 143, 0.3, 0.3, 200, 1220, -4000, 23)}

# inter = Hydrate_interface(componentsDict = componentsli,
#         structuresDict={'I' : Structure('I', 123, 143, 231, 2, 12, Cavity('5^12', 1e-10, 32, 20), Cavity('5^(12)6^2', 2*1e-10, 30, 24)),
#                         'II' : Structure('II', 123, 143, 231, 2, 12, Cavity('5^12', 1e-10, 32, 20), Cavity('5^(12)6^4', 2*1e-10, 30, 24))},
#         interpolPT = {})


main()