# Interface file for the hydrate algorithm

import tkinter as tk
from tkinter import ttk, messagebox
from os.path import isfile

# class FrameFocus(tk.LabelFrame):
#     def __init__(self, master, **kw):
#         super().__init__(master, **kw)

#         self.bind("<Button-1>", self.on_click)

#     def on_click(self, event):
#         self.focus()

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
                # print(self.interface.results, 'new temp set')
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = float(new_text)
                # print(self.interface.results, 'new pres set')

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
                # print(self.interface.results, 'new temp set')
            if event.widget == self.interface.presEnt:
                self.interface.results['Pressure'] = float(new_text)
                # print(self.interface.results, 'new pres set')

        # self.master.tk_focusNext().focus()


class ComboboxFocusOut(ttk.Combobox):
    def __init__(self, master, interface, **kw):
        super().__init__(master, **kw)
        self.interface = interface
        self.bind("<FocusOut>", self.on_focus_out)

    def on_focus_out(self, event):
        new_text = event.widget.get()
        self.interface.results['Structure'] = new_text
        # print(self.interface.results, 'new structure set')


class ButtonEnter(tk.Button):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.bind("<Return>", self.on_enter)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, event):
        self.invoke()

    def on_click(self, event):
        self.focus()


class Hydrate_interface:

    column_names_ini = ['temp', 'pres', 'struct', 'compo0', 'tocc_tot', 'toccS_tot', 'toccL_tot']
    column_names_main_ini = ['temp', 'pres', 'struct', 'compo0', 'tocc_tot']

    def __init__(self, componentsList: list, structuresList: list) -> None:
    # def __init__(self, data_folder: str) -> None:
        """Main window initialization"""
        self.componentsList = componentsList
        self.structuresList = structuresList
        self.results = {'Components' : [], 'Composition' : [], 'Temperature' : -1, 'Pressure' : -1, 'Structure' : '', 'Thetas' : []}
        self.detailsareShown = False
        self.column_names_all = ['temp', 'pres', 'struct', 'compo0', 'tocc_tot', 'toccS_tot', 'toccL_tot']
        # self.componentsList = []
        # self.structuresList = []
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
        #             self.structuresList.append()
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
        # frameL.pack_propagate(0)
        frameL.pack(side=tk.LEFT)

        # subframe for choices
        frameLU = tk.Frame(frameL, relief = tk.FLAT, bd = 2, width = 600, height = 400)
        # frameLU.pack_propagate(0)
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

        # Temperature entry
        tk.Label(frameLU, text= "Temperature (K) :").grid(sticky= tk.E, row = 7, column = 1, pady = 5, padx = 10)
        self.tempEnt = EntryFocusOut(frameLU, width=25, interface=self)
        self.tempEnt.grid(row= 7, column = 2, padx = 5, pady = 10, columnspan=2)

        # Widgets for the setting of Pressure: checkbox to enable entry of value
        tk.Label(frameLU, text= "Pressure (bar) :").grid(sticky= tk.E, row = 8, column = 1, pady = 5, padx = 10)
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
        self.structureChoice['values'] = self.structuresList
        self.structureChoice.grid(row= 9, column = 2, padx = 5, pady=5, columnspan=2)
        # disable the choice of structure if checkbox is unchecked
        self.checkStruct_var = tk.IntVar()
        tk.Checkbutton(frameLU, variable = self.checkStruct_var, text= "impose choice of structure", command = lambda : self.activateCheck(self.checkStruct_var, self.structureChoice)).grid(sticky= tk.W, row = 9, column=0, pady = 5, padx = 10)

        # General buttons subframe
        frameLD = tk.Frame(frameL, relief = tk.FLAT, bd = 2)
        frameLD.pack(side = tk.BOTTOM)

        RunBut = ButtonEnter(frameLD, text = "Run", command = self.run, width=7)
        RunBut.grid(row = 0, column = 0, sticky= tk.W, pady = 10, padx = 10)
        updateBut = ButtonEnter(frameLD, text = "Update", command = self.update, width = 7)
        updateBut.grid(row = 0, column=1, pady = 10, padx = 10)
        resetBut = ButtonEnter(frameLD, text = "Reset", command = self.reset, width = 7, foreground = 'red')
        resetBut.grid(row = 0, column=2, sticky=tk.E, pady = 10, padx = 10)

        # results frame
        self.frameR = tk.LabelFrame(self.root, relief=tk.GROOVE, bd = 2, text='Results')
        self.frameR.pack(side=tk.RIGHT, pady = 10)

        frameRU = tk.Frame(self.frameR, relief=tk.FLAT, bd = 2)
        frameRU.pack(side=tk.TOP, pady = 10)
        frameRUTree = tk.Frame(frameRU, relief=tk.FLAT, bd = 2)
        frameRUTree.pack(side='top', pady = 10)

        # self.column_names = ['ymol2', 'ymol3', 'temp', 'pres', 'struct', 'tocc_tot', 'toccS_tot', 'toccS1', 'toccS2', 'toccS3', 'toccL_tot' , 'toccL1', 'toccL2', 'toccL3']
        self.column_names = Hydrate_interface.column_names_ini.copy()
        self.column_names_main = Hydrate_interface.column_names_main_ini.copy()
        self.resultsTree = ttk.Treeview(frameRUTree, height=10, columns= self.column_names_all, displaycolumns=self.column_names_main)

        self.resultsTree.heading("#0", text = "y")
        self.resultsTree.heading("temp", text = "T (K)")
        self.resultsTree.heading("pres", text = "Peq (bar)")
        self.resultsTree.heading("struct", text = "Structure")
        self.resultsTree.heading("compo0", text = "x_H")
        self.resultsTree.heading("tocc_tot", text = "Total Occ")
        self.resultsTree.heading("toccS_tot", text = "Occ Small")
        self.resultsTree.heading("toccL_tot", text = "Occ Large")
        self.resultsTree.column("#0", width = 75)
        self.resultsTree.column("temp", width = 35)
        self.resultsTree.column("pres", width = 75)
        self.resultsTree.column("struct", width = 60)
        self.resultsTree.column("compo0", width = 75)
        self.resultsTree.column("tocc_tot", width = 75)
        self.resultsTree.column("toccS_tot", width = 75)
        self.resultsTree.column("toccL_tot", width = 75)


        scrollbarx = ttk.Scrollbar(frameRUTree, orient = "horizontal", command=self.resultsTree.xview)
        scrollbary = ttk.Scrollbar(frameRUTree, orient = "vertical", command=self.resultsTree.yview)
        scrollbarx.pack(side='bottom', fill = "x")
        scrollbary.pack(side='right', fill = "y")
        self.resultsTree.configure(xscrollcommand=scrollbarx.set, yscrollcommand=scrollbary.set)
        self.resultsTree.pack(padx = 10)

        self.detailsBut = ButtonEnter(frameRU, text = "Show details", command = lambda : self.showDetails(self.resultsTree, buttonShow=self.detailsBut, buttonHide=self.hideDetailsBut), width = 10)
        self.detailsBut.pack(side='bottom', pady = 5)
        self.hideDetailsBut = ButtonEnter(frameRU, text = "Hide details", command = lambda : self.hideDetails(self.resultsTree, buttonShow=self.detailsBut, buttonHide=self.hideDetailsBut), width = 10)


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


    def run(self):
        # TODO check if same components first? or before calling method?

        # if no components have been selected, returns error
        if len(self.tree.get_children()) == 0:
            messagebox.showerror(title='Error', message = 'Please select the gas components')
            return

        # if no temperature has been set, returns error
        if self.results['Temperature'] < 0:
            messagebox.showerror(title='Error', message = 'Please set the temperature')
            return

        # if the sum of all mole fractions of the gas isn't 1, returns error
        sumx = 0
        for item in self.tree.get_children():
            sumx += float(self.tree.item(item)['values'][0])
        if sumx != 1:
            messagebox.showerror(title='Error', message = 'Total mole fraction should be 1. \n Please check composition.')
            return

        self.column_names = Hydrate_interface.column_names_ini.copy()
        self.column_names_main = Hydrate_interface.column_names_main_ini.copy()
        # TODO once done replace it with the entered / calculated value
        if self.checkPres_var.get() == 0 or self.results['Pressure'] < 0:
            # TODO calculate the pressure with the algorithm
            self.results['Pressure'] = 'calcPeq'
            pass
        if self.checkStruct_var.get() == 0 or self.results['Structure'] == '':
            # TODO determine which structure with the algorithm
            self.results['Structure'] = 'calcStr'
            pass
        # complete the rest of results with the calculated compositions and thetas
        # TODO

        # if the new number of gas components is different from the previous one, the tree needs to be reset first
        if self.tree.get_children() != len(self.results['Components']) and len(self.resultsTree.get_children()) != 0:
            self.reset()

        # reset the values depending on components selection
        self.results['Components'] = []
        self.results['Composition'] = []
        self.results['Thetas'] = []

        values_item = [self.results['Temperature'], self.results['Pressure'], self.results['Structure']]
        # theta_S = thetajS + thetaj2S + ...
        # theta_L = thetajL + thetaj2L + ...
        # theta_tot = (theta_S * nuS + theta_L * nuL) / (nuS + nuL)


        j = 0
        for item in self.tree.get_children():

            name = self.tree.item(item)['text']
            print(self.tree.item(item)['values'][0])
            comp = self.tree.item(item)['values'][0]
            self.results['Components'].append(name)
            self.results['Composition'].append(float(comp))
            # TODO give the value of the calculated theta
            # xj = xj_H(all_thetas : list, j)
            self.results['Thetas'].append((f'ToccS{j}', f'ToccL{j}'))
            print(self.results)

            if j == 0:
                if len(self.tree.get_children()) > 1:
                    self.column_names.insert(-1, 'toccS0')
                    self.column_names.append('toccL0')
                    # TODO gives the values of the calculated total thetas
                    # tocc tot
                    values_item.append('Total')
                    # tocc small tot
                    values_item.append('Total Small')
                    values_item.append(self.results['Thetas'][0][0])
                    # tocc large tot
                    values_item.append('Total Large')
                    values_item.append(self.results['Thetas'][0][1])
                else:
                    # TODO tocc tot
                    values_item.append('Total')
                    values_item.append(self.results['Thetas'][0][0])
                    values_item.append(self.results['Thetas'][0][1])
                values_item.insert(3, f'compo0')


            elif j > 0:

                # self.column_names.insert(3+j, f"compo{j}")

                self.column_names.insert(j-1, f'y{j}')
                self.column_names_main.insert(j-1, f'y{j}')
                values_item.insert(j-1, self.results['Composition'][j])
                self.column_names.insert(-j-1, f'toccS{j}')
                self.column_names.append(f'toccL{j}')
                values_item.insert(-j-1, self.results['Thetas'][j][0])
                values_item.append(self.results['Thetas'][j][1])
                self.column_names.insert(4+j, f'compo{j}')
                self.column_names_main.insert(-1, f'compo{j}')
                values_item.insert(4+j, f'compo{j}')

            j += 1

            # add the new columns to the list of all existing columns
            for item in self.column_names:
                if item not in self.column_names_all:
                    self.column_names_all.append(item)


        print("all columns :", self.column_names_all, "values :", values_item, self.results, sep = "\n")
        self.resultsTree.configure(columns=self.column_names_all)

        for j in range(len(self.tree.get_children())):
            name = self.results['Components'][j]
            if j == 0:
                self.resultsTree.heading('#0', text = f'y ({name})')
                self.resultsTree.column('#0', width = 75)
            else:
                self.resultsTree.heading(f'y{j}', text = f'y ({name})')
                self.resultsTree.column(f'y{j}', width = 75)
            self.resultsTree.heading(f'compo{j}', text = f'x_H ({name})')
            self.resultsTree.column(f'compo{j}', width = 75)
            if len(self.tree.get_children())>1:
                self.resultsTree.heading(f'toccS{j}', text = f'Occ_S ({name})')
                self.resultsTree.column(f'toccS{j}', width = 75)
                self.resultsTree.heading(f'toccL{j}', text = f'Occ_L ({name})')
                self.resultsTree.column(f'toccL{j}', width = 75)

        self.resultsTree.heading("temp", text = "T (K)")
        self.resultsTree.heading("pres", text = "Peq (bar)")
        self.resultsTree.heading("struct", text = "Structure")
        self.resultsTree.heading("tocc_tot", text = "Total Occ")
        self.resultsTree.heading("toccS_tot", text = "Occ Small")
        self.resultsTree.heading("toccL_tot", text = "Occ Large")
        self.resultsTree.column("temp", width = 35)
        self.resultsTree.column("pres", width = 75)
        self.resultsTree.column("struct", width = 60)
        self.resultsTree.column("tocc_tot", width = 75)
        self.resultsTree.column("toccS_tot", width = 75)
        self.resultsTree.column("toccL_tot", width = 75)

        if self.detailsareShown == False:
            self.resultsTree.configure(displaycolumns=self.column_names_main)
        if self.detailsareShown == True:
            self.resultsTree.configure(displaycolumns=self.column_names)

        print(values_item)
        self.resultsTree.insert(parent = "", index= tk.END, text = self.results['Composition'][0], values= values_item)



        # TODO update column_names by dropping the non existing components
        # and rebuilding the tree accordingly
        # isEmpty = False
        # if self.resultsTree.item('#O')['text'] == 'x':
        #     isEmpty = True

        # updating columns
        # TODO write an override method for run in the algo with interface

        # l = len(self.results['Components'])
        # if l >= 1:
        #     # let's delete what is there and not needed
        #     for i in range(l+1, 4):
        #         if f'xmol{i}' in self.column_names:
        #             self.column_names.remove(f'xmol{i}')
        #         if f'toccS{i}' in self.column_names:
        #             self.column_names.remove(f'toccS{i}')
        #             self.column_names.remove(f'toccL{i}')
        #     if l == 1 :
        #         if f'toccS1' in self.column_names:
        #             self.column_names.remove('toccS1')
        #             self.column_names.remove('toccL1')

        #     # let's add what is needed and not there)
        #     for i in range(1, l+1):
        #         if f'xmol{i}' not in self.column_names:
        #             self.column_names.insert()
        #         if f'toccS{i}' not in self.column_names:
        #             self.column_names.remove(f'toccS{i}')
        #             self.column_names.remove(f'toccL{i}')


        # self.resultsTree.configure(displaycolumns=column_names[:-l])

        # yes:
            # overwrite the empty tree with the new one
        # no: # is it the same components as previously?
            # yes:
                # just add a row underneath
            # no:
                # create a whole new tree

        # for i in len(self.results['Components'][0]):
        #     column_names.insert()

    def update(self):
        # TODO this is like run but if the molecules remain the same: only a row is added with the new values
        # override this method in algo with interface
        # update the compositions, temp, Peq (go through calculations again), tocc
        pass

    def reset(self):
        if len(self.resultsTree.get_children()) == 0:
            return
        if messagebox.askokcancel(title = "WARNING: Data will be lost",
                message = "This will reset the data in the results tables. \n Do you wish to proceed?"):
            for item in self.resultsTree.get_children():
                self.resultsTree.delete(item)

            self.column_names = Hydrate_interface.column_names_ini.copy()
            self.column_names_main = Hydrate_interface.column_names_main_ini.copy()

            # if new tree needed
            # resultsTreeTemp = self.makeNewResultsTree(self.frameR)
            # print(self.column_names, self.column_names_main)
            # resultsTreeTemp.configure(columns= self.column_names_all)
            # # newresultsTreeTemp = self.makeNewResultsTree(new_columns=self.column_names, new_columns_main=self.column_names_main)
            # resultsTreeTemp.configure(displaycolumns=self.column_names_main)


            print(self.column_names, self.column_names_main)
            # TODO pb: cannot delete a column ==> need to do a whole new tree and replace the old one with it
            self.resultsTree.configure(columns= self.column_names_all)
            # newresultsTreeTemp = self.makeNewResultsTree(new_columns=self.column_names, new_columns_main=self.column_names_main)

            self.resultsTree.heading("#0", text = "y")
            self.resultsTree.heading("temp", text = "T (K)")
            self.resultsTree.heading("pres", text = "Peq (bar)")
            self.resultsTree.heading("struct", text = "Structure")
            self.resultsTree.heading("compo0", text = "x_H")
            self.resultsTree.heading("tocc_tot", text = "Total Occ")
            self.resultsTree.heading("toccS_tot", text = "Occ Small")
            self.resultsTree.heading("toccL_tot", text = "Occ Large")
            self.resultsTree.column("#0", width = 75)
            self.resultsTree.column("temp", width = 35)
            self.resultsTree.column("pres", width = 75)
            self.resultsTree.column("struct", width = 60)
            self.resultsTree.column("compo0", width = 75)
            self.resultsTree.column("tocc_tot", width = 75)
            self.resultsTree.column("toccS_tot", width = 75)
            self.resultsTree.column("toccL_tot", width = 75)

            self.resultsTree.configure(displaycolumns=self.column_names_main)

            # Hide details if they are currently being shown
            if self.detailsareShown == True:
                self.hideDetails(widget=self.resultsTree, buttonShow=self.detailsBut, buttonHide=self.hideDetailsBut)

    # def makeNewResultsTree(self, master) -> ttk.Treeview:
    #     # creates a new results tree with the columns in argument
    #     # does not do the configuration for the headings etc though, which is done is run or reset
    #     # return the new tree
    #     frameRD = tk.Frame(master, relief=tk.FLAT, bd = 2)
    #     frameRD.pack(side=tk.BOTTOM, pady = 10)
    #     frameRDTree = tk.Frame(frameRD, relief=tk.FLAT, bd = 2)
    #     frameRDTree.grid(row = 0, pady = 10)
    #     resultsTreeTemp = ttk.Treeview(frameRDTree, height=10, columns= self.column_names)

    #     scrollbarx = ttk.Scrollbar(frameRDTree, orient = "horizontal", command=resultsTreeTemp.xview)
    #     scrollbary = ttk.Scrollbar(frameRDTree, orient = "vertical", command=resultsTreeTemp.yview)
    #     scrollbarx.pack(side='bottom', fill = "x")
    #     scrollbary.pack(side='right', fill = "y")
    #     resultsTreeTemp.configure(xscrollcommand=scrollbarx.set, yscrollcommand=scrollbary.set)
    #     resultsTreeTemp.pack(padx = 10, side=tk.BOTTOM)

    #     self.detailsButTemp = ButtonEnter(frameRD, text = "Show details", command = lambda : self.showDetails(resultsTreeTemp, buttonShow=self.detailsButTemp, buttonHide=self.hideDetailsButTemp), width = 10)
    #     self.detailsButTemp.grid(row = 1, pady=5)
    #     self.hideDetailsButTemp = ButtonEnter(frameRD, text = "Hide details", command = lambda : self.hideDetails(resultsTreeTemp, buttonShow=self.detailsButTemp, buttonHide=self.hideDetailsButTemp), width = 10)

    #     return resultsTreeTemp

    def hideDetails(self, widget: ttk.Treeview, buttonShow: tk.Button, buttonHide: tk.Button):
        widget.configure(displaycolumns=self.column_names_main)
        buttonHide.pack_forget()
        buttonShow.pack(side='bottom', pady = 5)
        self.detailsareShown = False

    def showDetails(self, widget: ttk.Treeview, buttonShow: tk.Button, buttonHide: tk.Button):
        widget.configure( displaycolumns=self.column_names)
        buttonShow.pack_forget()
        self.detailsareShown = True
        buttonHide.pack(side='bottom', pady = 5)

# Example, TODO change to actual file names once they are created
COMPONENTS_FILENAME = 'components.txt'
STRUCTURES_FILENAME = 'structures.txt'

### TEST ###

molecules = ['H20', 'CH4', 'CO2']
structure = ['I', 'II']

# def select(event):
#     i = lb.curselection()[0]
#     item.set(items[i])

# def update(event):
#     i = lb.curselection()[0]
#     items[i] = item.get()
#     var.set(items)

# root = tk.Tk()
# items = dir(tk)
# var = tk.StringVar(value=items)

# lb = tk.Listbox(root, listvariable=var)
# lb.grid()
# lb.bind('<<ListboxSelect>>', select)

# item = tk.StringVar()
# entry = tk.Entry(root, textvariable=item, width=20)
# entry.grid()
# entry.bind('<Return>', update)


if __name__ == '__main__':
    app = Hydrate_interface(molecules, structure)


# tab = ['xmol2', 'xmol3', 'temp', 'pres', 'struct', 'tocc_tot', 'toccS_tot']
# # tab.remove('xmol4')

# tab.insert(-1, 'test')
# print(tab)