# Interface file for the hydrate algorithm

import tkinter as tk
from tkinter import ttk
from os.path import isfile


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
                # TODO make a pop up window appear saying please select within range
                print("Not in range, please enter a number between 0 and 1")
                return
        except ValueError:
            # TODO make a pop up window appear
            print("Please enter a number between 0 and 1")
            return

        selected_iid = event.widget.editing_item_iid
        column_index = event.widget.editing_column_index

        if column_index == 0:
            current_values = self.item(selected_iid).get("values")
            current_values[column_index] = new_text
            self.item(selected_iid, values = current_values)

        event.widget.destroy()

class Hydrate_interface:
    def __init__(self, componentsList: list, structuresList: list) -> None:
    # def __init__(self, data_folder: str) -> None:
        """Main window initialization"""
        self.componentsList = componentsList
        self.structuresList = structuresList
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
        #     print('Error: data file {COMPONENTS_FILENAME} is not in location {self.folder}')

        # if isfile(self.structures_file):
        #     with open(self.strucutres_file) as f :
        #         for line in f:
        #             self.structuresList.append()
        # else:
        #     print('Error: data file {STRUCTURES_FILENAME} is not in location {self.folder}')

        self.root = tk.Tk()
        self.root.title('Hydrate Composition')
        self.root.config(relief=tk.GROOVE, bd = 3, width=500, height=500)
        self.makeWidgets()
        self.componentsChoice.focus()
        self.root.mainloop()


    def makeWidgets(self):
        """Secondary widgets in main window"""
        # interactive frame
        frameL = tk.Frame(self.root, relief = tk.FLAT, bd = 2, width=1200, height = 600)
        # frameL.pack_propagate(0)
        frameL.pack()

        # subframe for choices
        frameLU = tk.LabelFrame(frameL, relief = tk.GROOVE, bd = 2, width = 600, height = 400, text='Choice of parameters')
        # frameLU.pack_propagate(0)
        frameLU.pack()

        # Components frame
        # frameLUL = tk.Frame(frameLU, relief = tk.FLAT)
        # frameLUL.grid()
        # tk.Label(frameLUL, text="Components").grid(row = 0, column = 0, sticky = tk.W)
        # self.componentsEnt = tk.Entry(frameLUL)
        # self.componentsEnt.grid(row = 1, column = 0, sticky = tk.W, padx = 5, pady = 10)
        # addCompo = tk.Button(frameLUL, text = "Add", command = self.addCompo)
        # addCompo.grid(row = 2, column = 0)
        # remCompo = tk.Button(frameLUL, text = "Remove", command = self.removeCompo)
        # remCompo.grid()

        # tk.Label(frameLU, text="Components").grid(row = 0, column = 0, columnspan=3, pady = 5, padx = 5)
        current_var = tk.StringVar()
        self.componentsChoice = ttk.Combobox(frameLU, width=35, textvariable = current_var, state = 'readonly')
        self.componentsChoice.bind("<<ComboboxSelected>>",lambda e: addCompo.focus())
        self.componentsChoice['values'] = self.componentsList

        # TODO make it work so that there is an indication as to what the combobox is for
        self.componentsChoice.set('Choose component')
        self.componentsChoice.grid(row = 1, column = 0, columnspan=2, padx = 10, pady = 5, sticky=tk.E)

        # adding selected component

        addCompo = tk.Button(frameLU, text = "Add", command = self.addCompo, width=7)
        addCompo.grid(row = 1, column = 2, sticky= tk.W, pady = 10, padx = 10)
        remCompo = tk.Button(frameLU, text = "Remove Component", command = self.removeCompo)
        remCompo.grid(row = 5, column=0, columnspan=2, pady = 3, padx = 20)
        clearComposition = tk.Button(frameLU, text = "Clear All Values", command = self.clearComposi)
        clearComposition.grid(row = 5, column=1, columnspan=2, pady = 3, padx = 20, sticky=tk.E)

        # Autochanging subsubframe with chosen components and their compositions

        frameLUM = tk.Frame(frameLU, relief = tk.FLAT, bd =2)
        frameLUM.grid(row=3, column = 0, columnspan = 4)

        self.tree = TreeviewEdit(frameLUM, columns= 'x_values')
        self.tree.heading('#0', text = 'Gas Components')
        self.tree.heading('x_values', text = 'Composition')
        self.tree.column("#0", minwidth=0, width=200, stretch=False)
        self.tree.column('x_values', minwidth=0, width=200, stretch=False)

        self.tree.grid(row=3, column = 0, columnspan=4)

        # Updating the composition values
        # https://www.youtube.com/watch?v=n5gItcGgIkk

        # detect which cell is clicked on



        # def addToTree():
        #     tree.insert("", text = )


        # frameLUM = tk.Frame(frameLU, relief = tk.FLAT, bd = 2)
        # frameLUM.grid(row=9, column=0, columnspan=3)
        # frameLUMList = tk.Frame(frameLUM, relief = tk.FLAT, bd =2)
        # frameLUMList.grid(row=4, column=0, columnspan=2, pady = 5, padx = 10)
        # self.scroll = tk.Scrollbar(frameLUMList)
        # self.chosenList = tk.Listbox(frameLUMList, yscrollcommand=self.scroll.set, height=6, width=30)
        # self.scroll.config(command=self.chosenList.yview)
        # self.scroll.pack(side = tk.RIGHT, fill= tk.Y, pady=5)
        # self.chosenList.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, pady = 5)

        # tk.Label(frameLUM, text= "Composition").grid(row = 3, column = 2, pady = 5, padx = 10)
        # self.compositionEnt = tk.Entry(frameLUM, width=30)
        # self.compositionEnt.grid(row= 4, column = 2, sticky = tk.N, padx = 5, pady = 5)

        # Other parameters

        # Temperature entry
        tk.Label(frameLU, text= "Temperature (K) :").grid(sticky= tk.E, row = 7, column = 1, pady = 5, padx = 10)
        self.tempEnt = tk.Entry(frameLU, width=25)
        self.tempEnt.grid(row= 7, column = 2, padx = 5, pady = 10, columnspan=2)

        # Widgets for the setting of Pressure: checkbox to enable entry of value
        tk.Label(frameLU, text= "Pressure (bar) :").grid(sticky= tk.E, row = 8, column = 1, pady = 5, padx = 10)
        self.presEnt = tk.Entry(frameLU, width=25, state='disabled')
        self.presEnt.grid(row= 8, column =2, padx = 5, pady=5, columnspan=2)
        # disable the entry of a pressure value is the checkbox is unchecked

        checkPres_var = tk.IntVar()
        tk.Checkbutton(frameLU, text= "impose equilibrium pressure", variable=checkPres_var, command=lambda : self.activateCheck(checkPres_var, self.presEnt)).grid(sticky= tk.W, row = 8, column=0, pady = 5, padx = 10)

        # Widgets for the choice of Structure: checkbox to enable choice
        tk.Label(frameLU, text= "Structure :").grid(sticky= tk.E, row = 9, column = 1, pady = 5, padx = 10)
        structure_current_var = tk.StringVar()
        self.structureChoice = ttk.Combobox(frameLU, width=25, textvariable = structure_current_var, state = 'disabled')
        self.structureChoice.set('Choose structure')
        self.structureChoice['values'] = self.structuresList
        self.structureChoice.grid(row= 9, column = 2, padx = 5, pady=5, columnspan=2)
        # disable the choice of structure if checkbox is unchecked
        checkStruct_var = tk.IntVar()
        # def activateCheck_Struct():
        #     if checkStruct_var.get() == 1:          # checked
        #         self.structureChoice.config(state='readonly')
        #         self.structureChoice.focus()
        #     elif checkStruct_var.get() == 0:        # unchecked
        tk.Checkbutton(frameLU, variable = checkStruct_var, text= "impose choice of structure", command = lambda : self.activateCheck(checkStruct_var, self.structureChoice)).grid(sticky= tk.W, row = 9, column=0, pady = 5, padx = 10)

        # General buttons subframe
        frameLD = tk.Frame(frameL, relief = tk.FLAT, bd = 2)
        frameLD.pack(side = tk.BOTTOM)

        RunBut = tk.Button(frameLD, text = "Run", command = self.run, width=7)
        RunBut.grid(row = 0, column = 0, sticky= tk.W, pady = 10, padx = 10)
        updateBut = tk.Button(frameLD, text = "Update", command = self.update, width = 7)
        updateBut.grid(row = 0, column=1, pady = 10, padx = 10)
        resetBut = tk.Button(frameLD, text = "Reset", command = self.reset, width = 7, foreground = 'red')
        resetBut.grid(row = 0, column=2, sticky=tk.E, pady = 10, padx = 10)

        # results frame
        frameR = tk.Frame(frameL, relief=tk.GROOVE, bd = 2)
        frameR.pack(pady = 10)


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
        pass
    def update(self):
        pass
    def reset(self):
        # TODO make sure that it is not a mistake => pop up box with warnign message
        pass

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

# TODO if in an entry, the wrong type (e.g. str instead of float) is entered ==> action = erase and pop up error message, new input please

if __name__ == '__main__':
    app = Hydrate_interface(molecules, structure)