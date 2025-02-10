# Interface file for the hydrate algorithm

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from os.path import isfile
import re
from tabulate import tabulate
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class TreeviewEdit(ttk.Treeview):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.bind("<Double-1>", self.on_double_click)

    # this method will allow for the treeview object's cells to be modified (here, so that the gas composition can be modified)
    def on_double_click(self, event):

        # identifying which cell was selected, so we can add an Entry widget corresponding to it
        region_clicked = self.identify_region(event.x, event.y)
        # we're only interested if the region clicked is a cell
        if region_clicked != "cell":
            return
        column = self.identify_column(event.x)
        # we're only interested if the region clicked is a cell in the second column (composition values)
        if column != "#1":
            return

        # caracteristics of selected cell
        column_index = int(column[1:]) - 1                              # column of selected cell (int) (starting at -1 for the tree, 0 for first value column)
        selected_iid = self.focus()                                     # row of selected cell (e.g. 'I001')
        selected_values = self.item(selected_iid)
        selected_text = selected_values.get('values')[column_index]     # value of selected cell (str)
        column_box = self.bbox(selected_iid, column)                    # position and dimensions of selected cell (x, y, w, h)

        # initialize Entry widget where the selected cell is, with the current value of the selected cell
        entry_edit = ttk.Entry(self.master, width = column_box[2])
        entry_edit.insert(0, selected_text)
        entry_edit.select_range(0, tk.END)
        entry_edit.focus()
        # record id of cell that is being modified (column and row of tree)
        entry_edit.editing_column_index = column_index
        entry_edit.editing_item_iid = selected_iid

        entry_edit.bind("<FocusOut>", self.on_focus_out)
        entry_edit.bind("<Return>", self.on_enter)

        # placing the Entry widget over the selected cell
        entry_edit.place(x=column_box[0], y=column_box[1], w=column_box[2], h=column_box[3])

    # the Entry object that was used to change the treeview value is deleted
    def on_focus_out(self, event):
        event.widget.destroy()

    # the Entry object is deleted here too, but after assigning the new entered value to the treeview cell corresponding to it
    def on_enter(self, event):
        new_text = event.widget.get()
        # first we need to check that the entered value is in range
        try :
            float(new_text)

            if float(new_text) > 1 or float(new_text) < 0:
                messagebox.showerror(title='Error', message = 'Not in range, please enter a number between 0 and 1')
                return
        except ValueError:
            messagebox.showerror(title='Error', message = 'Not a number, please enter a number between 0 and 1')
            return

        selected_iid = event.widget.editing_item_iid
        column_index = event.widget.editing_column_index
        # the corresponding cell is changed to the new entered value
        if column_index == 0:
            current_values = self.item(selected_iid).get("values")
            current_values[column_index] = new_text
            self.item(selected_iid, values = current_values)

        event.widget.destroy()


class EntryFocusOut(tk.Entry):
    def __init__(self, master, interface, **kw):
        super().__init__(master, **kw)
        self.interface = interface

        self.bind("<Return>", self.on_enter)
        self.bind("<FocusOut>", self.on_focus_out)

    # these will make sure that the entered temperature or pressure are within range, and then change the previous value to the new entered one

    def on_enter(self, event):
        new_text = event.widget.get()
        if new_text != '':
            try :
                float(new_text)

                if float(new_text) < 0:
                    messagebox.showerror(title='Error', message = 'Not in range, please enter a number above 0')
                    event.widget.delete(0, 'end')
                    return
            except ValueError:
                messagebox.showerror(title='Error', message = 'Not a number, please enter a number above 0')
                event.widget.delete(0, 'end')
                return

            # if the entered value is within range, the modified value is changed to the new entered value in the results dictionnary
            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = float(new_text)
            if event.widget == self.interface.presEnt:
                factor = 1
                if self.interface.unit_var.get() == 'MPa':
                    factor = 1E6
                elif self.interface.unit_var.get() == 'bar':
                    factor = 1E5
                self.interface.results['Pressure'] = float(new_text) * factor
        else:
            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = -1
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = -1

        self.master.tk_focusNext().focus()

    def on_focus_out(self, event):
        new_text = event.widget.get()
        if new_text != '':
            try :
                float(new_text)

                if float(new_text) < 0:
                    messagebox.showerror(title='Error', message = 'Not in range, please enter a number above 0')
                    event.widget.delete(0, 'end')
                    return
            except ValueError:
                messagebox.showerror(title='Error', message = 'Not a number, please enter a number above 0')
                event.widget.delete(0, 'end')
                return

            # if the entered value is within range, the modified value is changed to the new entered value in the results dictionnary
            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = float(new_text)
            if event.widget == self.interface.presEnt:
                factor = 1
                if self.interface.unit_var.get() == 'MPa':
                    factor = 1E6
                elif self.interface.unit_var.get() == 'bar':
                    factor = 1E5
                self.interface.results['Pressure'] = float(new_text) * factor
        else:
            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = float(-1)
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = -1


class ComboboxFocusOut(ttk.Combobox):
    def __init__(self, master, interface, **kw):
        super().__init__(master, **kw)
        self.interface = interface
        self.bind("<FocusOut>", self.on_focus_out)

    # this will change the previous value for structure in results to the new chosen structure
    def on_focus_out(self, event):
        new_text = event.widget.get()
        self.interface.results['Structure'] = new_text


class ButtonEnter(tk.Button):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.bind("<Return>", self.on_enter)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, event):
        self.invoke()

    def on_click(self, event):
        self.focus()


class Hydrate_interface_squelette:

    column_names_ini = ['temp', 'pres', 'struct', 'compo0', 'tocc_tot', 'toccS_tot', 'toccL_tot']
    column_names_main_ini = ['temp', 'pres', 'struct', 'compo0', 'tocc_tot']

    def __init__(self, componentsDict: dict, structuresDict: dict, interpolPT: dict, bip : dict, all_models : dict) -> None:
        """Main window initialization"""
        # attributes creation
        self.allComponents = componentsDict
        self.componentsList = list(componentsDict.keys())
        self.structuresDict = structuresDict
        self.PTinterpolDict =interpolPT
        self.bipDict = bip
        self.modelsDict = all_models

        self.results = {'Components' : [], 'Composition' : [], 'Temperature' : -1, 'Pressure' : -1, 'Structure' : '', 'Thetas' : [], 'Hydrate Composition' : [], 'Thetas_tots' : []}
        self.detailsareShown = False
        self.all_trees = {}
        self.all_treeFrames = {}

        # window initialization
        self.root = tk.Tk()
        self.root.maxsize(width=1520, height=1000)
        self.root.title('Hydrate Composition')
        self.root.config(relief=tk.GROOVE, bd = 3, width=1000, height=600)
        self.makeWidgets()
        self.componentsChoice.focus()
        self.root.mainloop()


    def makeWidgets(self):
        """Initialization of secondary widgets in main window"""
        ## interactive frame initialization
        frameL = tk.LabelFrame(self.root, relief = tk.GROOVE, bd = 2, text='Choice of parameters')
        frameL.pack(side=tk.LEFT)
        # subframe for choices
        frameLU = tk.Frame(frameL, relief = tk.FLAT, bd = 2, width = 600, height = 400)
        frameLU.pack()

        ## Gas compoisition
        # tk.Label(frameLU, text="Components").grid(row = 0, column = 0, columnspan=3, pady = 5, padx = 5)
        current_var = tk.StringVar()
        self.componentsChoice = ttk.Combobox(frameLU, width=35, textvariable = current_var, state = 'readonly')
        self.componentsChoice.bind("<<ComboboxSelected>>",lambda e: addCompo.focus())
        self.componentsChoice['values'] = self.componentsList
        # TODO make it work so that there is an indication as to what the combobox is for
        self.componentsChoice.set('--- Select component ---')
        self.componentsChoice.grid(row = 1, column = 0, columnspan=2, padx = 10, pady = 5, sticky=tk.E)

        # adding selected component to the tree of gas composition
        addCompo = ButtonEnter(frameLU, text = "Add", command = self.addCompo, width=7)
        addCompo.grid(row = 1, column = 2, sticky= tk.W, pady = 10, padx = 10)

        # Autochanging tree in subsubframe with chosen components and their modifiable compositions
        frameLUM = tk.Frame(frameLU, relief = tk.FLAT, bd =2)
        frameLUM.grid(row=3, column = 0, columnspan = 4)

        self.tree = TreeviewEdit(frameLUM, columns= 'y_values')
        self.tree.heading('#0', text = 'Gas Components')
        self.tree.heading('y_values', text = 'Gas Composition')
        self.tree.column("#0", minwidth=0, width=200, stretch=False)
        self.tree.column('y_values', minwidth=0, width=200, stretch=False)

        self.tree.grid(row=3, column = 0, columnspan=4)

        # Deleting components from table: adding buttons to remove the component entirely, and to reset all compositions to 0
        remCompo = ButtonEnter(frameLU, text = "Remove Component", command = self.removeCompo)
        remCompo.grid(row = 5, column=0, columnspan=2, pady = 3, padx = 20)
        clearComposition = ButtonEnter(frameLU, text = "Clear All Values", command = self.clearComposi)
        clearComposition.grid(row = 5, column=1, columnspan=2, pady = 3, padx = 20, sticky=tk.E)

        ## Other parameters
        # Widgets for the setting of Temperature: checkbox to enable entry of value
        tk.Label(frameLU, text= "Temperature (K) :").grid(sticky= tk.E, row = 7, column = 1, pady = 5, padx = 10)
        self.tempEnt = EntryFocusOut(frameLU, width=25, interface=self)
        self.tempEnt.grid(row= 7, column =2, padx = 5, pady=10, columnspan=2)

        # disable the entry of a pressure value is the checkbox is unchecked
        self.checkTemp_var = tk.IntVar(value = 1)
        tk.Checkbutton(frameLU, text= "impose equilibrium temperature", state='active', variable=self.checkTemp_var, command=lambda : self.activateCheck(self.checkTemp_var, self.tempEnt)).grid(sticky= tk.W, row = 7, column=0, pady = 5, padx = 10)

        # Widgets for the setting of Pressure: checkbox to enable entry of value
        tk.Label(frameLU, text= "Pressure :").grid(sticky= tk.E, row = 8, column = 1, pady = 5, padx = 10)
        self.presEnt = EntryFocusOut(frameLU, width=12, state='disabled', interface=self)
        self.presEnt.grid(row= 8, column =2, padx = 5, pady=5, columnspan=1)
        # Combobox to choose the unit for the pressure
        self.unit_var = tk.StringVar()
        self.unitCombo = ttk.Combobox(frameLU, width=8, textvariable = self.unit_var, state = 'disabled')
        self.unitCombo['values'] = ['Pa', 'MPa', 'bar']
        self.unitCombo.current(0)
        self.unitCombo.grid(row=8, column = 3)
        self.unitCombo.bind('<<ComboboxSelected>>', self.select_unit)

        # disable the entry of a pressure value is the checkbox is unchecked
        self.checkPres_var = tk.IntVar()
        tk.Checkbutton(frameLU, text= "impose equilibrium pressure", variable=self.checkPres_var, command=lambda : self.activateCheck(self.checkPres_var, self.presEnt)).grid(sticky= tk.W, row = 8, column=0, pady = 5, padx = 10)

        # Widgets for the choice of Structure: checkbox to enable choice
        tk.Label(frameLU, text= "Structure :").grid(sticky= tk.E, row = 9, column = 1, pady = 5, padx = 10)
        structure_current_var = tk.StringVar()
        self.structureChoice = ComboboxFocusOut(frameLU, width=25, textvariable = structure_current_var, state = 'disabled', interface = self)
        self.structureChoice.set('Choose structure')
        self.structureChoice['values'] = list(self.structuresDict.keys())
        self.structureChoice.grid(row= 9, column = 2, padx = 5, pady=5, columnspan=2)

        # disable the choice of structure if checkbox is unchecked
        self.checkStruct_var = tk.IntVar()
        tk.Checkbutton(frameLU, variable = self.checkStruct_var, text= "impose choice of structure", command = lambda : self.activateCheck(self.checkStruct_var, self.structureChoice)).grid(sticky= tk.W, row = 9, column=0, pady = 5, padx = 10)

        ## General buttons subframe
        frameLD = tk.Frame(frameL, relief = tk.FLAT, bd = 2)
        frameLD.pack(side = tk.BOTTOM)

        RunBut = ButtonEnter(frameLD, text = "Run", command = self.run, width=7)
        RunBut.grid(row = 0, column = 0, sticky= tk.W, pady = 10, padx = 10)
        optimizeBut = ButtonEnter(frameLD, text = "Optimize Kihara Parameters", command = self.optimizeOrChooseKihara)
        optimizeBut.grid(row = 0, column=1, pady = 10, padx = 10)
        resetBut = ButtonEnter(frameLD, text = "Reset", command = self.reset, width = 7)
        resetBut.grid(row = 0, column=2, sticky=tk.E, pady = 10, padx = 10)

        ## Results frame
        self.frameR = tk.LabelFrame(self.root, relief=tk.GROOVE, bd = 2, text='Results')
        self.frameR.pack(side=tk.RIGHT, pady = 10)

        self.frameRUTree = tk.Frame(self.frameR, relief=tk.FLAT, bd = 2)
        self.frameRUTree.pack(side='top', pady = 5)

        # results tree initialization
        column_names = Hydrate_interface_squelette.column_names_ini.copy()
        self.resultsTree = ttk.Treeview(self.frameRUTree,height=6, columns= column_names, displaycolumns=column_names)

        self.resultsTree.heading("#0", text = "y")
        self.resultsTree.heading("temp", text = "T (K)")
        self.resultsTree.heading("pres", text = "Peq (Pa)")
        self.resultsTree.heading("struct", text = "Structure")
        self.resultsTree.heading("compo0", text = "x_H")
        self.resultsTree.heading("tocc_tot", text = "Total Occ")
        self.resultsTree.heading("toccS_tot", text = "")
        self.resultsTree.heading("toccL_tot", text = "")
        self.resultsTree.column("#0", width = 75)
        self.resultsTree.column("temp", width = 35)
        self.resultsTree.column("pres", width = 75)
        self.resultsTree.column("struct", width = 60)
        self.resultsTree.column("compo0", width = 75)
        self.resultsTree.column("tocc_tot", width = 75)
        self.resultsTree.column("toccS_tot", width = 75)
        self.resultsTree.column("toccL_tot", width = 75)

        scrollbarx = ttk.Scrollbar(self.frameRUTree, orient = "horizontal", command=self.resultsTree.xview)
        scrollbary = ttk.Scrollbar(self.frameRUTree, orient = "vertical", command=self.resultsTree.yview)
        scrollbarx.pack(side='bottom', fill = "x")
        scrollbary.pack(side='right', fill = "y")
        self.resultsTree.configure(xscrollcommand=scrollbarx.set, yscrollcommand=scrollbary.set)
        self.resultsTree.pack(padx = 5)

        # export button enabling the export of the values in this tree
        self.exportBut = ButtonEnter(self.frameRUTree, text = "Export table", command = lambda : self.exportTree(self.resultsTree), width = 10)
        self.exportBut.pack(side='bottom', pady = 5)

        # add the initial tree to the list of existing results trees to keep track
        self.all_trees[(0,0)] = [self.resultsTree, self.exportBut, scrollbarx, scrollbary]
        self.all_treeFrames[(0,0)] = self.frameRUTree

        # myscrollbar= tk.Scrollbar(self.frameR, orient="vertical")
        # myscrollbar.pack(side="right",fill="y")


    def addCompo(self):
        """Adds the selected component to the components tree when the Add button is clicked"""
        selected_compo = self.componentsChoice.get()
        if selected_compo == '':
            return
        for item in self.tree.get_children():
            if self.tree.item(item)['text'] == selected_compo:
                return
        self.tree.insert(parent = "", index= tk.END, text = selected_compo, values= 0.0)


    def removeCompo(self):
        """Removes the selected component from the components tree when the Remove button is clicked"""
        if self.tree.selection() == ():
            return
        selected_item = self.tree.selection()[0]
        self.tree.delete(selected_item)


    #resets all the compositions values to 0
    def clearComposi(self):
        """Resets all the components' composition in the components tree to 0.0 when the Clear button is clicked"""
        for item in self.tree.get_children():
            self.tree.item(item, values = 0.0)

    def select_unit(self, event):
        entry_content = event.widget.get()
        self.unit_var.set(entry_content)
        new_text = self.presEnt.get()
        if new_text != '':
            factor = 1
            if self.unit_var.get() == 'MPa':
                factor = 1E6
            elif self.unit_var.get() == 'bar':
                factor = 1E5
            self.results['Pressure'] = float(new_text) * factor

    def activateCheck(self, var, widget):
        """Makes sure that Temperature, Pressure or Structure choice is enabled/disabled according to their corresponding checkbutton"""
        if var.get() == 1:          # checked
            if isinstance(widget, ttk.Combobox):
                widget.config(state='readonly')
            else:
                widget.config(state='normal')
                if widget == self.presEnt:
                    self.unitCombo.config(state='readonly')
            widget.focus()
        elif var.get() == 0:        # unchecked
            widget.config(state='disabled')
            if widget == self.presEnt:
                self.unitCombo.config(state='disabled')
            # at least one of Pressure or Temperature has to be entered, otherwise the calculations will not work
            if widget == self.tempEnt and self.checkPres_var.get() == 0:
                self.checkPres_var.set(1)
                self.activateCheck(self.checkPres_var, self.presEnt)
            elif widget == self.presEnt and self.checkTemp_var.get() == 0:
                self.checkTemp_var.set(1)
                self.activateCheck(self.checkTemp_var, self.tempEnt)


    def run(self):
        pass

    def optimizeOrChooseKihara(self):
        top = tk.Toplevel(self.root)
        top.geometry("650x150")

        def select_compo(event):
            entry_content = event.widget.get()
            popup_var.set(entry_content)
        def select_eps(event):
            entry_content = event.widget.get()
            eps_var.set(entry_content)
        def select_sig(event):
            entry_content = event.widget.get()
            sig_var.set(entry_content)
        def select_a(event):
            entry_content = event.widget.get()
            a_var.set(entry_content)

        popup_var = tk.StringVar()
        tk.Label(top, text= "Component to optimize :").grid(row = 0, column = 0, padx = 5)
        Kihara_combo = ttk.Combobox(top, width=20, textvariable = popup_var, state = 'readonly')
        Kihara_combo['values'] = [self.tree.item(child)['text'] for child in self.tree.get_children()]
        # Kihara_combo['values'] = self.componentsList
        Kihara_combo.grid(row=0, column = 2)
        Kihara_combo.bind('<<ComboboxSelected>>', select_compo)
        # print(component)

        tk.Button(top, text="Calculate optimized Kihara parameters", command=lambda: self.optimizeKiharaAndClose(top)).grid(row = 0, column = 3, padx = 10)

        sig_var = tk.StringVar()
        eps_var = tk.StringVar()
        a_var = tk.StringVar()
        tk.Label(top, text= "epsilon / k :").grid(row = 3, column = 0, padx = 10)
        entry_eps = tk.Entry(top, width= 15, textvariable=eps_var)
        entry_eps.grid(row = 4, column=0)
        tk.Label(top, text= "sigma :").grid(row = 3, column = 1, padx = 10)
        entry_sig = tk.Entry(top, width= 15, textvariable=sig_var)
        entry_sig.grid(row = 4, column=1)
        tk.Label(top, text= "a :").grid(row = 3, column = 2, padx = 10)
        entry_a = tk.Entry(top, width= 15, textvariable=a_var)
        entry_a.grid(row = 4, column=2)

        entry_eps.bind("<FocusOut>", select_eps)
        entry_eps.bind("<Return>", select_eps)

        entry_sig.bind("<FocusOut>", select_sig)
        entry_sig.bind("<Return>", select_sig)

        entry_a.bind("<FocusOut>", select_a)
        entry_a.bind("<Return>", select_a)


        # component = popup_var.get()
        # print("compo is : ", component, popup_var.get(), chosen_eps)

        tk.Label(top, text= "OR").grid(row = 2, column = 3, pady = 5, padx = 10)
        tk.Button(top,text= "Choose Kihara parameters", command= lambda:self.chooseKihara(top, popup_var.get(), eps_var.get(), sig_var.get(), a_var.get())).grid(row = 4, column = 3, pady=10, padx = 10)


    # def chooseKiharaAndClose(self, top):
    #     top.destroy()
    #     self.chooseKihara()

    def chooseKihara(self, top, component, chosen_eps, chosen_sig, chosen_a):
        self.allComponents[component].epsilon = chosen_eps
        self.allComponents[component].sigma = chosen_sig
        self.allComponents[component].a = chosen_a
        top.destroy()

    def optimizeKiharaAndClose(self, top):
        top.destroy()
        self.optimizeKihara()

    def optimizeKihara(self):
        pass


    def reset(self):
        """Resets the results tree(s) when the Reset button is clicked"""
        if len(self.all_trees) == 1 and len(list(self.all_trees.values())[0][0].get_children()) == 0:           # i.e. if only the initial blank tree is there, no need for a reset
            return
        if messagebox.askokcancel(title = "WARNING: Data will be lost",
                message = "This will reset the data in all results tables. \n Do you wish to proceed?"):
            for tree_id in self.all_treeFrames:
                self.all_treeFrames[tree_id].destroy()                                                          # destroying the tree frame will destroy the tree and all widgets connected to it
            # the lists of existing trees is then reset
            self.all_trees = {}
            self.all_treeFrames = {}

            # a new initial blank tree is created
            column_names = Hydrate_interface_squelette.column_names_ini.copy()
            resultsTree, exportBut, scrollbarx, scrollbary, frameTree = self.makeNewResultsTree(self.frameR, column_names)

            resultsTree.heading("#0", text = "y")
            resultsTree.heading("temp", text = "T (K)")
            resultsTree.heading("pres", text = "Peq (Pa)")
            resultsTree.heading("struct", text = "Structure")
            resultsTree.heading("compo0", text = "x_H")
            resultsTree.heading("tocc_tot", text = "Total Occ")
            resultsTree.heading("toccS_tot", text = "")
            resultsTree.heading("toccL_tot", text = "")
            resultsTree.column("#0", width = 75)
            resultsTree.column("temp", width = 35)
            resultsTree.column("pres", width = 75)
            resultsTree.column("struct", width = 60)
            resultsTree.column("compo0", width = 75)
            resultsTree.column("tocc_tot", width = 75)
            resultsTree.column("toccS_tot", width = 75)
            resultsTree.column("toccL_tot", width = 75)

            resultsTree.configure(displaycolumns=column_names)

            self.all_trees[(0,0)] = resultsTree, exportBut, scrollbarx, scrollbary
            self.all_treeFrames[(0,0)] = frameTree


    def makeNewResultsTree(self, master, columns) -> ttk.Treeview:
        """Creates and returns an additional results tree with the given columns"""
        frameRDTree = tk.Frame(master, relief=tk.FLAT, bd = 2)
        frameRDTree.pack(side='top', pady = 5)

        resultsTreeTemp = ttk.Treeview(frameRDTree,height=6, columns= columns, displaycolumns=columns)

        scrollbarxTemp = ttk.Scrollbar(frameRDTree, orient = "horizontal", command=resultsTreeTemp.xview)
        scrollbaryTemp = ttk.Scrollbar(frameRDTree, orient = "vertical", command=resultsTreeTemp.yview)
        scrollbarxTemp.pack(side='bottom', fill = "x")
        scrollbaryTemp.pack(side='right', fill = "y")
        resultsTreeTemp.configure(xscrollcommand=scrollbarxTemp.set, yscrollcommand=scrollbaryTemp.set)
        resultsTreeTemp.pack(padx = 2)

        exportButTemp = ButtonEnter(frameRDTree, text = "Export table", command = lambda : self.exportTree(resultsTreeTemp), width = 10)
        exportButTemp.pack(side='bottom', pady = 5)

        return resultsTreeTemp, exportButTemp, scrollbarxTemp, scrollbaryTemp, frameRDTree


    def exportTree(self, tree: ttk.Treeview, filename = ""):
        """Opens a dialog window to chose/create a file in which to save the data from the results tree corresponding to the Export button that was clicked"""
        filename = filedialog.asksaveasfilename(parent = self.root,                             # prevents the dialogbox from being hidden behind the main window
                                                initialdir=dir_path,                            # opens the file explorer to current location
                                                filetypes=[("Text files (*.txt)",".txt")],      # authorized file types in which to save data
                                                defaultextension=".txt",
                                                confirmoverwrite=False)
        # turns the data in the treeview into a reader friendly string
        head = [tree.heading('#0')['text']] + [tree.heading(column_id)['text'] for column_id in tree['columns']]
        table1 = []
        for child in tree.get_children():
            table1 += ([[tree.item(child).get('text')] + tree.item(child).get('values')]
                    + [[tree.item(sub_child).get('text')] + tree.item(sub_child).get('values') for sub_child in tree.get_children(child)] )
        n = len(tree.get_children(tree.get_children()[0])) + 1
        for i in range(int(len(table1)/n)):
            table1[n*i] = [item for item in table1[n*i] if item != '']
        table = tabulate( table1,
                          headers= head)
        if filename != "":
            with open(filename, 'a') as f:
                f.write('\n' * 3)
                f.write(table)

### TEST ###

# if __name__ == '__main__':
#     app = Hydrate_interface_squelette(molecules, structure)