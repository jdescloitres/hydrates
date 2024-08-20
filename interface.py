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

    def on_focus_out(self, event):
        event.widget.destroy()

    def on_enter(self, event):
        new_text = event.widget.get()
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

            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = float(new_text)
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = float(new_text)
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

            if event.widget == self.interface.tempEnt:
                self.interface.results['Temperature'] = float(new_text)
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = float(new_text)
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
    # def __init__(self, data_folder: str) -> None:
        """Main window initialization"""
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

        # self.componentsDict = {}
        # self.structuresDict = {}
        # self.folder = data_folder

        # # TODO complete attributes for files depending on location and file content
        # self.components_file = data_folder + COMPONENTS_FILENAME
        # self.structures_file = data_folder + STRUCTURES_FILENAME
        # if isfile(self.components_file):
        #     with open(self.components_file) as f :
        #         for line in f:
        #             self.componentsList.append()
        # else:
        #     print(f'Error: data file {COMPONENTS_FILENAME} is not in location {self.folder}')

        # if isfile(self.structures_file):
        #     with open(self.strucutres_file) as f :
        #         for line in f:
        #             self.structuresDict.append()
        # else:
        #     print(f'Error: data file {STRUCTURES_FILENAME} is not in location {self.folder}')

        self.root = tk.Tk()
        self.root.maxsize(width=1520, height=1000)
        self.root.title('Hydrate Composition')
        self.root.config(relief=tk.GROOVE, bd = 3, width=1000, height=600)
        self.makeWidgets()
        self.componentsChoice.focus()
        self.root.mainloop()


    def makeWidgets(self):
        """Secondary widgets in main window"""
        # interactive frame
        frameL = tk.LabelFrame(self.root, relief = tk.GROOVE, bd = 2, text='Choice of parameters')
        frameL.pack(side=tk.LEFT)

        # subframe for choices
        frameLU = tk.Frame(frameL, relief = tk.FLAT, bd = 2, width = 600, height = 400)
        frameLU.pack()

        # Components frame
        # tk.Label(frameLU, text="Components").grid(row = 0, column = 0, columnspan=3, pady = 5, padx = 5)
        current_var = tk.StringVar()
        self.componentsChoice = ttk.Combobox(frameLU, width=35, textvariable = current_var, state = 'readonly')
        self.componentsChoice.bind("<<ComboboxSelected>>",lambda e: addCompo.focus())
        self.componentsChoice['values'] = self.componentsList

        # TODO make it work so that there is an indication as to what the combobox is for
        self.componentsChoice.set('--- Select component ---')
        self.componentsChoice.grid(row = 1, column = 0, columnspan=2, padx = 10, pady = 5, sticky=tk.E)

        # adding selected component
        addCompo = ButtonEnter(frameLU, text = "Add", command = self.addCompo, width=7)
        addCompo.grid(row = 1, column = 2, sticky= tk.W, pady = 10, padx = 10)

        # Autochanging subsubframe with chosen components and their compositions
        frameLUM = tk.Frame(frameLU, relief = tk.FLAT, bd =2)
        frameLUM.grid(row=3, column = 0, columnspan = 4)

        self.tree = TreeviewEdit(frameLUM, columns= 'y_values')
        self.tree.heading('#0', text = 'Gas Components')
        self.tree.heading('y_values', text = 'Gas Composition')
        self.tree.column("#0", minwidth=0, width=200, stretch=False)
        self.tree.column('y_values', minwidth=0, width=200, stretch=False)

        self.tree.grid(row=3, column = 0, columnspan=4)

        # Deleting components from table
        remCompo = ButtonEnter(frameLU, text = "Remove Component", command = self.removeCompo)
        remCompo.grid(row = 5, column=0, columnspan=2, pady = 3, padx = 20)
        clearComposition = ButtonEnter(frameLU, text = "Clear All Values", command = self.clearComposi)
        clearComposition.grid(row = 5, column=1, columnspan=2, pady = 3, padx = 20, sticky=tk.E)

        # Other parameters

        # Widgets for the setting of Temperature: checkbox to enable entry of value
        tk.Label(frameLU, text= "Temperature (K) :").grid(sticky= tk.E, row = 7, column = 1, pady = 5, padx = 10)
        self.tempEnt = EntryFocusOut(frameLU, width=25, interface=self)
        self.tempEnt.grid(row= 7, column =2, padx = 5, pady=10, columnspan=2)

        # disable the entry of a pressure value is the checkbox is unchecked
        self.checkTemp_var = tk.IntVar(value = 1)
        tk.Checkbutton(frameLU, text= "impose equilibrium temperature", state='active', variable=self.checkTemp_var, command=lambda : self.activateCheck(self.checkTemp_var, self.tempEnt)).grid(sticky= tk.W, row = 7, column=0, pady = 5, padx = 10)

        # Widgets for the setting of Pressure: checkbox to enable entry of value
        tk.Label(frameLU, text= "Pressure (Pa) :").grid(sticky= tk.E, row = 8, column = 1, pady = 5, padx = 10)
        self.presEnt = EntryFocusOut(frameLU, width=25, state='disabled', interface=self)
        self.presEnt.grid(row= 8, column =2, padx = 5, pady=5, columnspan=2)

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

        # General buttons subframe
        frameLD = tk.Frame(frameL, relief = tk.FLAT, bd = 2)
        frameLD.pack(side = tk.BOTTOM)

        RunBut = ButtonEnter(frameLD, text = "Run", command = self.run, width=7)
        RunBut.grid(row = 0, column = 0, sticky= tk.W, pady = 10, padx = 10)
        optimizeBut = ButtonEnter(frameLD, text = "Optimize Kihara Parameters", command = self.optimizeKihara)
        optimizeBut.grid(row = 0, column=1, pady = 10, padx = 10)
        resetBut = ButtonEnter(frameLD, text = "Reset", command = self.reset, width = 7)
        resetBut.grid(row = 0, column=2, sticky=tk.E, pady = 10, padx = 10)

        # results frame
        self.frameR = tk.LabelFrame(self.root, relief=tk.GROOVE, bd = 2, text='Results')
        self.frameR.pack(side=tk.RIGHT, pady = 10)

        self.frameRUTree = tk.Frame(self.frameR, relief=tk.FLAT, bd = 2)
        self.frameRUTree.pack(side='top', pady = 5)

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

        self.exportBut = ButtonEnter(self.frameRUTree, text = "Export table", command = lambda : self.exportTree(self.resultsTree), width = 10)
        self.exportBut.pack(side='bottom', pady = 5)

        self.all_trees[(0,0)] = [self.resultsTree, self.exportBut, scrollbarx, scrollbary]
        self.all_treeFrames[(0,0)] = self.frameRUTree

        # myscrollbar= tk.Scrollbar(self.frameR, orient="vertical")
        # myscrollbar.pack(side="right",fill="y")


    def addCompo(self):
        selected_compo = self.componentsChoice.get()
        if selected_compo == '':
            return
        for item in self.tree.get_children():
            if self.tree.item(item)['text'] == selected_compo:
                return
        self.tree.insert(parent = "", index= tk.END, text = selected_compo, values= 0.0)


    def removeCompo(self):
        if self.tree.selection() == ():
            return
        selected_item = self.tree.selection()[0]
        self.tree.delete(selected_item)


    #resets all the compositions values to 0
    def clearComposi(self):
        for item in self.tree.get_children():
            self.tree.item(item, values = 0.0)


    def activateCheck(self, var, widget):
        if var.get() == 1:          # checked
            if isinstance(widget, ttk.Combobox):
                widget.config(state='readonly')
            else:
                widget.config(state='normal')
            widget.focus()
        elif var.get() == 0:        # unchecked
            widget.config(state='disabled')
            # at least one of Pressure or Temperature has to be entered
            if widget == self.tempEnt and self.checkPres_var.get() == 0:
                self.checkPres_var.set(1)
                self.activateCheck(self.checkPres_var, self.presEnt)
            elif widget == self.presEnt and self.checkTemp_var.get() == 0:
                self.checkTemp_var.set(1)
                self.activateCheck(self.checkTemp_var, self.tempEnt)


    def run(self):
        pass


    def optimizeKihara(self):
        pass


    def reset(self):
        if len(self.all_trees) == 1 and len(list(self.all_trees.values())[0][0].get_children()) == 0:
            return
        if messagebox.askokcancel(title = "WARNING: Data will be lost",
                message = "This will reset the data in all results tables. \n Do you wish to proceed?"):
            for tree_id in self.all_treeFrames:
                self.all_treeFrames[tree_id].destroy()
            self.all_trees = {}
            self.all_treeFrames = {}

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
        # returns the new tree
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
        filename = filedialog.asksaveasfilename(parent = self.root, initialdir=dir_path, filetypes=[("Text files (*.txt)",".txt")], defaultextension=".txt", confirmoverwrite=False)
        head = ['y0'] + [tree.heading(column_id)['text'] for column_id in tree['columns']]
        table1 = []
        for child in tree.get_children():
            table1 += ([[tree.item(child).get('text')] + tree.item(child).get('values')]
                    + [[tree.item(sub_child).get('text')] + tree.item(sub_child).get('values') for sub_child in tree.get_children(child)] )
        # print(table1)
        n = len(tree.get_children(tree.get_children()[0])) + 1
        for i in range(int(len(table1)/n)):
            table1[n*i] = [item for item in table1[n*i] if item != '']
        # print(table1)
        table = tabulate( table1,
                          headers= head)
        # print(table1, table, sep='\n')
        if filename != "":
            with open(filename, 'a') as f:
                f.write('\n' * 3)
                f.write(table)

# Example, TODO change to actual file names once they are created
COMPONENTS_FILENAME = 'components.txt'
STRUCTURES_FILENAME = 'structures.txt'

### TEST ###

molecules = ['H20', 'CH4', 'CO2']
structure = ['I', 'II']


# if __name__ == '__main__':
#     app = Hydrate_interface_squelette(molecules, structure)