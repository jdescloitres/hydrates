# Hydrate Composition Calculation Algorithm with Interface


import csv
import re
import tkinter as tk
from tkinter import messagebox

from matplotlib import pyplot as plt
import hydrate_algorithm_no_interface as calc
import interface
import numpy as np
# from scipy.optimize import newton, curve_fit
# from scipy.interpolate import interp1d

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
    def __init__(self, formula: str, component_id: int, y, Tc, Pc, epsilon, sigma, a, Vinf, k1, k2, omega) -> None:
        self.formula = formula
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
    """Returns the value of Pressure corresponding to a given value of Temperature, using a function previously obtained through 1D interpolation of literature equilibrium data"""
    # P = PfromT._call(self=PfromT, x_new= T)
    return PfromT(T)
    # return PfromT._call(self=PfromT, x_new= T)

def calculateTfromP(P, TfromP):
    """Returns the value of Temperature corresponding to a given value of Pressure, using a function previously obtained through 1D interpolation of literature equilibrium data"""
    # return TfromP._call(self = TfromP, x_new=P)
    return TfromP(P)

def getInterpolFuncs(compo_foldername):
    """Returns interpolation functions of literature equilibrium data for Pressure and Temperature, for a given component"""
    # this gets the names of all files containing equilibrium data for the given component (each file corresponding to data from one reference)
    all_rep_str = compo_foldername + 'all_repertories.csv'
    with open(all_rep_str, mode='r') as infile:
        reader = csv.reader(infile)
        all_rep_list = [rows[0] for rows in reader]

    list_temp=[]
    list_pres=[]
    # this will retrieve all values of Pressure and Temperature, for all the references files for this component
    for paper in all_rep_list:
        filename = compo_foldername + paper + '.csv'
        with open(filename, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            properties = [(float(rows[0]), float(rows[1])) for rows in reader]
            list_temp += [item[0] for item in properties]
            # TODO ADD 1E6 HERE BUT TAKE IT OFF ELSEWHERE ?
            list_pres += [item[1] for item in properties]

    # the interpolation functions are created using all references data
    PfromT = calc.PfromT_f(list_temp = list_temp, list_pres=list_pres)
    xTmin = min(list_temp)
    xTmax = max(list_temp)
    TfromP = calc.TfromP_f(list_pres = list_pres, list_temp = list_temp)
    xPmin = min(list_pres)
    xPmax = max(list_pres)


    # PfromT = id
    # TfromP = id
    return ((PfromT, xTmin, xTmax), (TfromP, xPmin, xPmax))

def main():
    """Initializes the interface with data from literature"""
    # first, get PT methods for pure gases
    allInterpolPT = {}
    # get all the names of available pure gases
    folder_name = DATA_FOLDER + PT_FOLDER + '1CondensableElements/'
    all_repertories_str = folder_name + 'all_repertories.txt'
    with open(all_repertories_str, mode = 'r') as infile:
        all_repertories_list = [line for line in infile.read().splitlines()]

    # if isfile(self.components_file):
    #     with open(self.components_file) as f :
    #         for line in f:
    #             self.componentsList.append()
    # else:
    #     print(f'Error: data file {COMPONENTS_FILENAME} is not in location {self.folder}')

    # for each of these pure gases, the interpolation functions for T and P of the component are determined
    for compo in all_repertories_list:
        compo_foldername = folder_name + compo + '/'
        allInterpolPT[compo] = getInterpolFuncs(compo_foldername)

    # get all binary interaction parameters data from literature
    bip = {}
    bip_file = DATA_FOLDER + BIP_FOLDER + 'BIP.csv'
    with open(bip_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        all_lines = [row for row in reader]
        for row in all_lines:
            for col in list(all_lines[0].keys())[1:]:
                if col != 'ref':
                    bip[tuple(sorted((row['formula'],col)))] = float(row[col])              # to each alphabetically sorted couple of components is associated their bip

    # initialize Cavity objects
    cavitiesI_file = DATA_FOLDER + CAVITIES_FOLDER + 'structureI.csv'
    with open(cavitiesI_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        cavitiesI_properties = [row for row in reader]
    cavitiesI = [Cavity(name=cavitiesI_properties[0]['name'],
                        ray=float(cavitiesI_properties[0]['r']),
                        coord_z=int(cavitiesI_properties[0]['z']),
                        pop_nu=float(cavitiesI_properties[0]['nu'])),
                Cavity(name=cavitiesI_properties[1]['name'],
                        ray=float(cavitiesI_properties[1]['r']),
                        coord_z=int(cavitiesI_properties[1]['z']),
                        pop_nu=float(cavitiesI_properties[1]['nu']))
                ]

    cavitiesII_file = DATA_FOLDER + CAVITIES_FOLDER + 'structureII.csv'
    with open(cavitiesII_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        cavitiesII_properties = [row for row in reader]
    cavitiesII = [Cavity(name=cavitiesII_properties[0]['name'],
                        ray=float(cavitiesII_properties[0]['r']),
                        coord_z=int(cavitiesII_properties[0]['z']),
                        pop_nu=float(cavitiesII_properties[0]['nu'])),
                Cavity(name=cavitiesII_properties[1]['name'],
                        ray=float(cavitiesII_properties[1]['r']),
                        coord_z=int(cavitiesII_properties[1]['z']),
                        pop_nu=float(cavitiesII_properties[1]['nu']))
                ]

    # get all parameters data from all available models (i.e. from different references in literature) for Structure objects; this will be useful when optimizing Kihara Parameters
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
        # NOTE: the values in this dictionnary are strings, and float() needs to be called when using those values


    # initialize Structure objects
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
    components_file = DATA_FOLDER + COMPONENTS_FOLDER + CHOSEN_MODEL_COMPONENTS
    with open(components_file, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        compo_properties = [row for row in reader]
    all_components = {compo_properties[row]['formula'] : Component(formula=compo_properties[row]['formula'],
                                                            component_id=int(compo_properties[row]['id']),
                                                            y=1.0,                                              # the molar fraction is initialized at 0, as it is in the components tree
                                                            Tc=float(compo_properties[row]['Tc']),
                                                            Pc=float(compo_properties[row]['Pc'])*1E6,
                                                            epsilon=float(compo_properties[row]['epsilon']),
                                                            sigma=float(compo_properties[row]['sigma']),
                                                            a=float(compo_properties[row]['a']),
                                                            Vinf=float(compo_properties[row]['Vinf']),
                                                            k1=float(compo_properties[row]['k1']),
                                                            k2=float(compo_properties[row]['k2']),
                                                            omega=float(compo_properties[row]['omega']))
                for row in range(len(compo_properties))}

    # first optimization of Kihara parameters
    # for component in all_components.values():
    # for component in {'N2': all_components['N2']}.values():
    #     # print([(att, getattr(structures['I'],att)) for att in dir(structures['I'])])
    #     T1 = allInterpolPT[component.formula][0][1]
    #     T2 = allInterpolPT[component.formula][0][2]
    #     epsilon, sigma = calc.optimisationKiharafromT(T1 = T1, T2=T2, calculate_unknownPfromT=calculatePfromT, calculateTfromP=calculateTfromP,allPTinterpol=allInterpolPT, component_pure=component, structure=structures['I'], bips=bip, list_models= all_models, n_T=10)
    #     component.epsilon = epsilon
    #     component.sigma = sigma
    #     # print(epsilon, sigma)
    #     # TODO finish pb with curve_fit
    #     print("FOR NOW STOP BUT AT SOME POINT PICK IT UP TO MAKE IT WORK")


    # ## test for Pfromt extrapolation etc
    # inter_Test = allInterpolPT['N2'][0][0]
    # xT = np.linspace(250, 300, 1000)
    # plt.plot(xT, inter_Test(xT),color = 'red')
    # plt.plot(xT, inter_Test(xT))
    # print(inter_Test(300))
    # plt.show()

    # ### TESTS for calc
    # # print(calc.fugacity_j(T=300, P=40E6, components={'N2': all_components['N2']},bips=bip, component_keyj= 'N2'))
    # # print('langmuir : ', calc.langmuir_ij(T=273, structure_cavities=[Cavity('6^1',3.91, 20, 16), Cavity('6^2',4.735, 28, 8)], components={'N2': all_components['N2']}, i = 1, component_keyj= 'N2'))
    # xP = np.linspace(0, 50, 1000)
    # # fig, (ax1, ax2) = plt.subplots(1, 2)
    # cav = [Cavity('5^1',3.91, 20, 0.0434), Cavity('5^2',4.33, 24, 0.1304)]
    # def yPL(x):
    #     return [calc.deltaMu_L(T=273, P=x_unit, structure=Structure('I', 1714, 1400, -38.12, 0.141, 0.00000499644, cav[0], cav[1]),components={'N2': all_components['N2']},bips=bip) for x_unit in x]
    # def yPH(x):
    #     return [calc.deltaMu_H(T=273, P=x_unit,structure_cavities=cav, components={'N2': all_components['N2']},bips=bip) for x_unit in x]
    # # for i in range(len(xP)):
    # #     print(xP[i], yPL(xP)[i], yPH(xP)[i])
    # # ax1.plot(xP, yPL(xP),color = 'red')
    # # ax1.plot(xP, yPH(xP))
    # plt.plot(xP, yPL(xP),color = 'red')
    # plt.plot(xP, yPH(xP))
    # plt.show()

    # creation of the interface with the new run function that is created in this file
    inter = Hydrate_interface(componentsDict=all_components, structuresDict=structures, interpolPT = allInterpolPT, bip = bip, all_models = all_models)

class Hydrate_interface(interface.Hydrate_interface_squelette):
    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self):
        """Calculates all unknown parameters from given values and displays them in a results tree when the Run button is clicked"""
        ## check if all provided information is correct and within range
        # if no components have been selected, returns error
        if len(self.tree.get_children()) == 0:
            messagebox.showerror(title='Error', message = 'Please select the gas components')
            return

        # if no temperature AND no pressure has been set, returns error (calculations are not possible without at least one)
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

        ## check whether or not the Pressure, Temperature and/or Structure need to be determined through calculations, or if their are given
        TisCalculated = False
        PisCalculated = False
        StrisCalculated = False
        if self.checkTemp_var.get() == 0 or self.results['Temperature'] < 0:
            TisCalculated = True
        if self.checkPres_var.get() == 0 or self.results['Pressure'] < 0:
            PisCalculated = True
        if self.checkStruct_var.get() == 0 or self.results['Structure'] == '':
            StrisCalculated = True

        ### update the results dictionnary
        theta_tot, thetas_Sall, thetas_Lall = self.update_results(PisCalculated, TisCalculated, StrisCalculated)
        self.results['Thetas_tots'] = [theta_tot, thetas_Sall, thetas_Lall]
        small_cavity, big_cavity = self.structuresDict[self.results['Structure']].cavities

        # get all the components that were selected for this round of calculations
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
        # print(calculatePfromT(275, self.PTinterpolDict[tree_components][0][0]))

        ### differenciate how the tree is to be built depending on components
        # if this is the first round, i.e. if the initial results tree was empty before, the initial tree is used
        if len(self.all_trees) == 1 and len(list(self.all_trees.values())[0][0].get_children()) == 0:
            columnsTemp = self.column_names_ini
            treeTemp, buttonTemp, scrollbarxTemp, scrollbaryTemp = list(self.all_trees.values())[0]                 # this gets the widgets of the initial tree
            frameTemp = list(self.all_treeFrames.values())[0]                                                       # this gets the frame containing the initial tree and the widgets
            tree_components = tuple(self.results['Components'])
            self.all_trees.clear()                                                                                  # the default key that was used for the initial tree is deleted from the dictionnary of all trees
            self.all_treeFrames.clear()                                                                             # so is the default key for the initial tree frame
            self.all_trees[tree_components] = [treeTemp, buttonTemp, scrollbarxTemp, scrollbaryTemp]                # only for the key to be replaced by a sorted tuple of the selected components for this round
            self.all_treeFrames[tree_components] = frameTemp                                                        # that way, it is easier to determine if a certain combination of components already has its own tree

        # if the combination of components has been used previously, we use that tree, and no need to change columns
        elif tree_components in self.all_trees:                                                                     # if the sorted tuple of the selected components is already a key in the dict, i.e. already has its own tree
            treeTemp = self.all_trees[tree_components][0]                                                           # the used tree is that already existing tree
            self.addRowtoTree(treeTemp)                                                                             # and only a new row needs to be added
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

        # update the lists of column and associated values
        values_item = [self.results['Temperature'], self.results['Pressure'], self.results['Structure']]
        l = len(self.tree.get_children())
        # this will be the list for the secondary rows (when details about the occupancy rates)
        values_tocc = [ [''] * (l*2 + 2) for i in range(len(self.tree.get_children())) ]                        # l*2 blanks because 2 columns (y and x) for each component,
                                                                                                                # -1 for the first column y0 which is considered 'text' and not 'value',
                                                                                                                # +3 for Temperature, Pressure and Structure columns ==> total of (l*2 + 2) columns before tocc that need blanks
        for component_index in range(len(self.results['Components'])):
            if component_index > 0:                                                                             # the columns y0 and x0 already exist
                columnsTemp.insert(component_index-1, f'y{component_index}')                                    # this will insert the columns at the beginning of the list, in index order
                columnsTemp.insert(-3, f'compo{component_index}')                                               # this will insert the columns right before the occupancy rates 3 columns, in index order

                values_item.insert(component_index-1, self.results['Composition'][component_index])             # note: y0 is excluded because it is not entered in 'values' but as 'text'

            values_item += [self.results['Hydrate Composition'][component_index]]                               # all x (even x0) values are added, in index order
            values_tocc[component_index] += [self.results['Components'][component_index],                       # all occupancy rates are added to the occupancy list of values
                                             self.results['Thetas'][component_index][0],
                                             self.results['Thetas'][component_index][1]]

        values_item += self.results['Thetas_tots']                                                              # the total occupancy rates are added to the end of the values list

        # configure and fill the tree with the columns and values lists
        treeTemp.configure(columns=columnsTemp)

        for component_index in range(len(self.results['Components'])):
            formula = self.results['Components'][component_index]
            if component_index == 0:                                                                            # the first column of a tree (here, y0) needs to be treated seperately
                treeTemp.heading('#0', text = f'y ({formula})')
                treeTemp.column('#0', width = 75)
                print(formula)
            else:
                treeTemp.heading(f'y{component_index}', text = f'y ({formula})')
                treeTemp.column(f'y{component_index}', width = 75)
            treeTemp.heading(f'compo{component_index}', text = f'x_H ({formula})')
            treeTemp.column(f'compo{component_index}', width = 75)

        treeTemp.heading("temp", text = "T (K)")
        treeTemp.heading("pres", text = "Peq (Pa)")
        treeTemp.heading("struct", text = "Structure")
        treeTemp.heading("tocc_tot", text = "Total Occ")
        # treeTemp.heading("toccS_tot", text = f"{small_cavity.name}")
        # treeTemp.heading("toccL_tot", text = f"{big_cavity.name}")
        treeTemp.heading("toccS_tot", text = "Small cavity")
        treeTemp.heading("toccL_tot", text = "Large cavity")
        treeTemp.column("temp", width = 35)
        treeTemp.column("pres", width = 75)
        treeTemp.column("struct", width = 60)
        treeTemp.column("tocc_tot", width = 75)
        treeTemp.column("toccS_tot", width = 75)
        treeTemp.column("toccL_tot", width = 75)

        treeTemp.configure(displaycolumns=columnsTemp)

        # the row for the main parameters is added
        treeTemp.insert(parent = "", index= tk.END, iid=f'{len(treeTemp.get_children())}', text = self.results['Composition'][0], values= values_item)
        # for each of the components, a row of the detailed occupancy rates is added
        for component_index in range(len(self.results['Components'])):
            treeTemp.insert(parent = f'{len(treeTemp.get_children())-1}', index= tk.END, text = '', values= values_tocc[component_index])

        # reset data that were calculated and not given
        if TisCalculated:
            self.results['Temperature'] = -1
        if PisCalculated:
            self.results['Pressure'] = -1
        if StrisCalculated:
            self.results['Structure'] = ''


    def update_results(self, PisCalculated : bool, TisCalculated: bool, StrisCalculated: bool):
        """Calculates missing parameters values and updates the results dictionnary accordingly. Returns total occupancy rates"""
        # TODO replace arguments with actual function arguments
        # creates a dictionnary containing all Component objects involved in this round, and sets their mole fraction to the one entered in the tree
        components = {}
        for item in self.tree.get_children():
            components[self.tree.item(item)['text']] = self.allComponents[self.tree.item(item)['text']]
            components[self.tree.item(item)['text']].y = float(self.tree.item(item)['values'][0])
        structures = self.structuresDict
        bip = self.bipDict

        # creation of the PfromT etc interpol functions for mixed gas, if they don't exist already
        gas_formula = '='.join(sorted(components))
        components_formulas = tuple(sorted(components))
        num = len(components)
        if num > 1 and components_formulas not in self.PTinterpolDict.keys():
            folder_name = DATA_FOLDER + PT_FOLDER + f'{num}CondensableElements/'
            compo_foldername = folder_name + gas_formula + '/'
            self.PTinterpolDict[components_formulas] = getInterpolFuncs(compo_foldername)
        interpol = self.PTinterpolDict


        # lists of components and composition are updated in the results dictionnary
        self.results['Components'] = [components[component_key].formula for component_key in sorted(components)]
        self.results['Composition'] = [ components[component_key].y for component_key in sorted(components)]

        ### calculate the rest of the values we need
        # if P is not given
        if PisCalculated:
            # Peq = 'Pcalc'
            Teq = float(self.results['Temperature'])
            # # if the structure used is known, we can directly get the corresponding values
            if not StrisCalculated:
                # Peq = calc.calculatePi(T = Teq, Pexpi = calculatePfromT(T, calc.))
                Peq = 5E6
                struct = self.results['Structure']
                cijI = {component_key : calc.langmuir_ij() for component_key in sorted(components)}
                fjI = {component_key : calc.fugacity_j() for component_key in sorted(components)}
                self.results['Pressure'] = Peq

            # if not, we need to determine the more adequate structure
            else:
                if len(components) > 1:
                    sorted_compo_keys = tuple(sorted(components.keys()))
                else:
                    sorted_compo_keys = tuple(components.keys())[0]
                # estimate P suppose Structure is I
                structureI = structures['I']
                PeqI = calc.calculatePi(Ti = Teq, Pexpi = calculatePfromT(Teq, interpol[sorted_compo_keys][0][0]), structurei=structureI, componentsi=components, bipsi=bip)[0]
                print('here1')
                # PeqI = calculatePfromT(Teq, interpol[sorted_compo_keys][0][0]) * 1E6
                # TODO here: calculate Pi gives a very bad estimate, check langmuir...
                print('PeqI :' , PeqI, Teq, sorted_compo_keys)
                cijI = {component_key : (calc.langmuir_ij(T=Teq, structure_cavities=structureI.cavities, components=components, i=0, component_keyj=component_key),
                                        calc.langmuir_ij(T=Teq, structure_cavities=structureI.cavities, components=components, i=1, component_keyj=component_key)) for component_key in sorted(components)}
                fjI = {component_key : calc.fugacity_j(T = Teq, P=PeqI, components=components, bips=bip, component_keyj=component_key) for component_key in sorted(components)}
                # print(cijI, fjI)

                # repeat for Str II
                # PeqII = calc.calculatePi(Ti = Teq, Pexpi = calculatePfromT(Teq, interpol[sorted_compo_keys][0][0]))
                structureII = structures['II']
                # PeqII = calc.calculatePi(Ti = Teq, Pexpi = calculatePfromT(Teq, interpol[sorted_compo_keys][0][0])*1E6, structurei=structureII, componentsi=components, bipsi=bip)[0]
                # print('I : ', PeqI, 'II : ', PeqII)
                # PeqII = 17E6
                PeqII = PeqI
                # cijII = {component_key : (calc.langmuir_ij(0), calc.langmuir_ij(1)) for component_key in sorted(components)}
                cijII = {component_key : (calc.langmuir_ij(T=Teq, structure_cavities=structureII.cavities, components=components, i=0, component_keyj=component_key),
                                        calc.langmuir_ij(T=Teq, structure_cavities=structureII.cavities, components=components, i=1, component_keyj=component_key)) for component_key in sorted(components)}
                fjII = {component_key : calc.fugacity_j(T = Teq, P=PeqII, components=components, bips=bip, component_keyj=component_key) for component_key in sorted(components)}
                # print(cijII, fjII)

                # determine which structure corresponds best to equilibrium
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
                self.results['Pressure'] = Peq
                self.results['Structure'] = struct.id
                print(cij, fj)

        # TODO
        # if T is not given
        elif TisCalculated:
            Peq = float(self.results['Pressure'])
            if len(components) > 1:
                    sorted_compo_keys = tuple(sorted(components.keys()))
            else:
                sorted_compo_keys = tuple(components.keys())[0]
            print(interpol['N2'])
            # estimate P suppose Structure is I
            # Teq = 'Tcalc'
            print(Peq)
            Teq = calculateTfromP(Peq, interpol[sorted_compo_keys][1][0])
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
        # struct = 'I'

        # calculate thetasj and xj
        thetas = {component_key : (calc.theta_ij(cij, fj, struct.cavities, 0, component_key),
                calc.theta_ij(cij, fj, struct.cavities, 1, component_key)) for component_key in sorted(components)}
        xj = {component_key : calc.xj_H(structure_cavities=struct.cavities, components=components, thetas=thetas, component_keyj=component_key) for component_key in sorted(components)}

        # fill in results array for the rest of the calculated data
        # self.results['Hydrate Composition'] = [ 'xj' + str(component_key) for component_key in sorted(components) ]
        self.results['Hydrate Composition'] = [ xj[component_key] for component_key in sorted(components) ]
        # self.results['Structure'] = struct
        # self.results['Thetas'] = [('small', 'large') for component_key in sorted(components)]
        self.results['Thetas'] = [ thetas[component_key] for component_key in sorted(components)]
        thetas_Sall, thetas_Lall = calc.thetas_iall(thetas=thetas, components=components)
        #  = calc.thetas_iall(thetas=thetas, components=components, structure_cavities=struct.cavities)[1]
        theta_tot =( (thetas_Sall * struct.cavities[0].nu + thetas_Lall * struct.cavities[1].nu)
                            / (struct.cavities[0].nu + struct.cavities[1].nu)
        )
        # print(calc.fugacity_j(Teq, Peq, components, bip, component_keyj='N2'))

        return theta_tot, thetas_Sall, thetas_Lall
        # return 'theta_tot', 'thetas_Sall', 'thetas_Lall'


    def addRowtoTree(self, tree):
        """Skips to creating values lists, and adds a row to an already existing given tree"""
        values_item = [self.results['Temperature'], self.results['Pressure'], self.results['Structure']]
        l = len(self.tree.get_children())
        values_tocc = [ [''] * (l*2 + 2) for i in range(len(self.tree.get_children())) ]

        for component_index in range(len(self.results['Components'])):
            if component_index > 0:
                values_item.insert(component_index-1, self.results['Composition'][component_index])
            values_item += [self.results['Hydrate Composition'][component_index]]
            values_tocc[component_index] += [self.results['Components'][component_index], self.results['Thetas'][component_index][0], self.results['Thetas'][component_index][1]]
        values_item += self.results['Thetas_tots']

        # the row for the main parameters is added
        tree.insert(parent = "", index= tk.END, iid=f'{len(tree.get_children())}', text = self.results['Composition'][0], values= values_item)
        # for each of the components, a row of the detailed occupancy rates is added
        for j in range(len(self.results['Components'])):
            tree.insert(parent = f'{len(tree.get_children())-1}', index= tk.END, text = '', values= values_tocc[j])


    def optimizeKihara(self):
        """Opitmizes the Kihara Parameters using models from literature, when the Optimize button is clicked"""

        # list_models = self.modelsDict
        for item in self.tree.get_children():
            component, compo_key = self.allComponents[self.tree.item(item)['text']], self.tree.item(item)['text']
            print(component, compo_key)
            # TODO how to choose P or T, and which bounds P1 P2 to choose ==> window comes on ?
            T1 = self.PTinterpolDict[component.formula][0][1]
            T2 = self.PTinterpolDict[component.formula][0][2]
            # calc.optimisationKiharafromP(component_pure=component, kw=list_models)
            # calc.optimisationKiharafromT(component_pure=component)
            epsilon, sigma = calc.optimisationKiharafromT(T1 = T1, T2=T2,
                                                          calculate_unknownPfromT=calculatePfromT,
                                                          calculateTfromP=calculateTfromP,
                                                          allPTinterpol=self.PTinterpolDict,
                                                          component_pure=component,
                                                          structure=self.structuresDict['I'],
                                                          bips=self.bipDict,
                                                          list_models= self.modelsDict, n_T=10)
            P1 = self.PTinterpolDict[component.formula][1][1]
            P2 = self.PTinterpolDict[component.formula][1][2]
            print('howdy here', epsilon, sigma)
            # if self.PisCalculated:
            #     pass
        # return super().optimizeKihara()


DATA_FOLDER = 'DataFiles/'
PT_FOLDER = 'EquilibriumDataPT/'
COMPONENTS_FOLDER = 'CondensableElementsProperties/'
BIP_FOLDER = 'CondensableElementsProperties/BIP/'
STRUCTURES_FOLDER = 'StructureReferenceProperties/'
CHOSEN_MODEL = 'Handa_and_Tse'
CHOSEN_MODEL_COMPONENTS = 'Sloan.csv'
CAVITIES_FOLDER = 'StructureReferenceProperties/CavitiesInternalParameters/'


if __name__ == '__main__':
    main()