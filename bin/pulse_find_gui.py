import tkinter as tk
import matplotlib as mpl 
mpl.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import median_absolute_deviation as mad
from pulse_find import Candidate


class Application(tk.Frame):

    def __init__(self, master, bursts, burst_metadata, outfilename, data, dtype):
        self.master = master
        self.bursts = bursts
        self.burst_metadata = burst_metadata
        self.outfilename = outfilename
        self.data = data
        self.dtype=dtype

        master.geometry("1000x800")
        super().__init__(self.master)

        self.frames = []
        self.selected = tk.IntVar()

        self.chk_btn_vars = []

        self.create_widgets()

    def show_frame(self, frame, value):
        frame = PlotWindow(self.master, int(value), self.bursts, self.burst_metadata, self.outfilename, self.data, self.dtype)

        frame.tkraise()



    def create_widgets(self):
        self.create_frames()
        self.create_labels_per_frame()


    def create_frames(self):
        self.master.rowconfigure(0, weight=4)
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.top_frame = tk.Frame(self.master)
        self.top_frame.grid(row=0, column=0, sticky="nsew")

        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")

        self.add_buttons(self.bottom_frame)

        self.top_frame.columnconfigure(0, weight = 1)
        self.top_frame.columnconfigure(1, weight = 1)
        self.top_frame.columnconfigure(2, weight = 1)
        self.top_frame.columnconfigure(3, weight = 1)
        self.top_frame.columnconfigure(4, weight=1)
        self.top_frame.rowconfigure(0, weight=1)

        self.frame1 = tk.Frame(self.top_frame)
        self.frame1.grid(row=0, column=0, sticky='nsew')

        self.frames.append(self.frame1)

        self.frame2 = tk.Frame(self.top_frame)
        self.frame2.grid(row=0, column=1, sticky='nsew')

        self.frames.append(self.frame2)

        self.frame3 = tk.Frame(self.top_frame)
        self.frame3.grid(row=0, column=2, sticky='nsew')

        self.frames.append(self.frame3)

        self.frame4 = tk.Frame(self.top_frame)
        self.frame4.grid(row=0, column=3, sticky='nsew')

        self.frames.append(self.frame4)

        self.frame5 = tk.Frame(self.top_frame)
        self.frame5.grid(row=0, column=4, sticky='nsew')

        self.frames.append(self.frame5)



    def create_labels_per_frame(self):

        for frame in self.frames:
            self.create_labels(frame)


    def create_labels(self, parent):
        column_index = int(parent.grid_info()['column'])


        if column_index==0:
            parent.rowconfigure(0, weight=1)
            parent.columnconfigure(0, weight=1)

            label=tk.Label(parent, text="Location from obs beginning")
            label.grid(row=0, column=0)#, sticky='nsew')
        elif column_index==1:
            parent.rowconfigure(0, weight=1)
            parent.columnconfigure(0, weight=1)

            label=tk.Label(parent, text="Signal to noise ratio")
            label.grid(row=0, column=0)#, sticky='nsew')

        elif column_index==2:
            parent.rowconfigure(0, weight=1)
            parent.columnconfigure(0, weight=1)

            label=tk.Label(parent, text="Total fluence")
            label.grid(row=0, column=0)#, sticky='nsew')       
        elif column_index==3:
            parent.rowconfigure(0, weight=1)
            parent.columnconfigure(0, weight=1)

            label=tk.Label(parent, text="Select burst to plot")
            label.grid(row=0, column=0)
        else:
            parent.rowconfigure(0, weight=1)
            parent.columnconfigure(0, weight=1)

            label=tk.Label(parent, text="Delete selected bursts")
            label.grid(row=0, column=0)



        for index, burst in enumerate(self.bursts):


            if column_index==0:
                parent.rowconfigure(index + 1, weight=2)
                #parent.columnconfigure(0, weight=1)

                label = tk.Label(parent, text=str(np.around(burst.location, decimals=2)) + " s")
                label.grid(row=index + 1, column=0, sticky='ew')


            elif column_index==1:
                parent.rowconfigure(index + 1, weight=2)
                #parent.columnconfigure(0, weight=1)

                label = tk.Label(parent, text=str(int(np.around(burst.sigma, decimals=0))))
                label.grid(row=index + 1, column=0, sticky='ew')

            elif column_index==2:
                parent.rowconfigure(index + 1, weight=2)
                #parent.columnconfigure(0, weight=1)

                label = tk.Label(parent, text=str(burst.fluence) + " Jyms")
                label.grid(row=index + 1, column=0, sticky='ew')

            elif column_index==3:
                parent.rowconfigure(index + 1, weight=2)
                #parent.columnconfigure(0, weight=1)

                radbtn = tk.Radiobutton(parent, value=int(index), variable=self.selected)
                radbtn.grid(row=index + 1, column=0, sticky='ew')
            else:
                parent.rowconfigure(index + 1, weight=2)
                self.chk_variable = tk.IntVar()

                chk_btn = tk.Checkbutton(parent, variable=self.chk_variable, onvalue=1, offvalue=0)

                self.chk_btn_vars.append(self.chk_variable)

                chk_btn.grid(row=index + 1, column=0, sticky='ew')

    def add_buttons(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)

        parent.rowconfigure(0, weight=1)

        plot_btn = tk.Button(parent, text="Plot selected burst", command=lambda: self.show_frame(PlotWindow, self.selected.get()))
        plot_btn.grid(row=0, column=0, sticky='ns')

        delete_btn = tk.Button(parent, text="Delete selected bursts", command=self.delete_bursts)
        delete_btn.grid(row=0, column=1, sticky='ns')

        finish_btn = tk.Button(parent, text="Finish", command=self.save_and_quit)
        finish_btn.grid(row=0, column=2, sticky="ns")



    def delete_bursts(self):
        new_bursts = []
        for index, var in enumerate(self.chk_btn_vars):

            if int(var.get())==0:

                new_bursts.append(self.bursts[index])



        frame = Application(self.master, new_bursts, self.burst_metadata, self.outfilename, self.data, self.dtype)
        frame.tkraise()

    def save_and_quit(self):

        for index, burst in enumerate(self.bursts):

            with open(str(self.outfilename) + "_detected_bursts.txt", "a") as f:
                if index==0:
                    f.write("Location from observation beginning (s), Signal to noise, Burst fluence (Jyms)" +"\n")

                f.write(str(burst.location) + "," + str(burst.sigma) + "," + str(burst.fluence) + "\n")

            np.save(str(self.outfilename) + "_acf_" + str(np.around(burst.location), decimals=2) + "s_" + "burst", burst.acf, allow_pickle=False)
            np.save(str(self.outfilename) + "_image_" + str(np.around(burst.location), decimals=2) + "s_" + "burst", burst.image, allow_pickle=False)

        np.save(self.outfilename + "_bursts", self.bursts, allow_pickle=True)

        self.master.destroy()


           

class PlotWindow(tk.Frame):

    def __init__(self, master, value, bursts, burst_metadata, outfilename, data, dtype):
        self.master = master
        self.value = value
        self.bursts = bursts
        self.burst = self.bursts[value]
        self.burst_metadata = burst_metadata
        self.outfilename = outfilename
        self.data = data

        self.text1 = tk.StringVar()
        self.text2 = tk.StringVar()
        self.dtype = dtype

        super().__init__(self.master)

        self.create_frames()


    def create_frames(self):
        self.master.rowconfigure(0, weight=4)
        self.master.rowconfigure(1, weight=1)

        self.master.columnconfigure(0, weight=1)

        self.frame1 = tk.Frame(self.master)
        self.frame1.grid(row=0, column=0, sticky="nsew")
        self.create_canvas(self.frame1)

        self.frame2 = tk.Frame(self.master)
        self.frame2.grid(row=1, column=0, sticky='nsew')
        self.create_buttons(self.frame2)


    def create_canvas(self, parent):

        fig = Figure(figsize=(10, 8), dpi=100)

        extent = [self.burst.location, self.burst.location + (2048*self.burst_metadata[0]), self.burst_metadata[2] - (self.burst_metadata[1]*self.burst_metadata[3]/2) + self.burst_metadata[1], self.burst_metadata[2] + (self.burst_metadata[1]*self.burst_metadata[3]/2)] #x, endx, y, endy

        time_array = np.linspace(self.burst.location, self.burst.location + (2048*self.burst_metadata[0]), np.shape(self.burst.image)[1])

        ax1 = fig.add_subplot(111)

        ax1.imshow(self.burst.image, extent=extent, aspect='auto')#, extent=extent)

        ax1.set_ylabel("Frequency (MHz)", fontsize=15)


        divider = make_axes_locatable(ax1)
        axbottom = divider.append_axes("bottom", size=1.2, pad=0.3, sharex=ax1)

        median = np.median(np.mean(self.burst.image, axis=0))
        med_dev = mad(np.mean(self.burst.image, axis=0))


        axbottom.plot(time_array, (np.mean(self.burst.image, axis=0)-median)/med_dev, marker="o", mfc="k", ms=1, mec="k") 

        axbottom.margins(x=0)

        axbottom.set_xlabel("Time from beginning of observation (s)", fontsize=15)
        axbottom.set_ylabel("Signal-to-noise ratio", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().grid(row=0, column=0)


    def create_buttons(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)

        parent.rowconfigure(0, weight=1)

        back_btn = tk.Button(parent, text="Back to main page", command=self.go_back)
        back_btn.grid(row=0, column=0, sticky='n')

        plot_range_btn = tk.Button(parent, text="Plot desired range", command=self.plot_range_frames)
        plot_range_btn.grid(row=0, column=1, sticky='n')

        plot_next_btn = tk.Button(parent, text="Plot next burst", command=lambda: self.plot_next_burst(self.value + 1))
        plot_next_btn.grid(row=0, column=2, sticky='n')



    def go_back(self):
        frame = Application(self.master, self.bursts, self.burst_metadata, self.outfilename, self.data, self.dtype)
        frame.tkraise()


    def plot_range_frames(self):

        self.top = tk.Toplevel()
        self.top.rowconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)
        self.top.columnconfigure(0, weight=1)

        self.entry_frame1 = tk.Frame(self.top)
        self.entry_frame1.grid(row=0, column=0)

        self.entry_frame2 = tk.Frame(self.top)
        self.entry_frame2.grid(row=1, column=0)

        self.entry_frame1.columnconfigure(0, weight=1)
        self.entry_frame1.columnconfigure(1, weight=1)
        self.entry_frame1.rowconfigure(0, weight=1)
        self.entry_frame1.rowconfigure(1, weight=1)

        l1 = tk.Label(self.entry_frame1, text="Enter beginning time")
        l1.grid(row=0, column=0)

        box1 = tk.Entry(self.entry_frame1, textvariable=self.text1)
        box1.grid(row=1, column=0)

        l2 = tk.Label(self.entry_frame1, text="Enter ending time")
        l2.grid(row=0, column=1)

        box2 = tk.Entry(self.entry_frame1, textvariable=self.text2)
        box2.grid(row=1, column=1)

        btn = tk.Button(self.entry_frame2, text="Plot selected range", command= self.plot_range)
        btn.pack()

    def plot_range(self):
        self.top.destroy()

        new_top = tk.Toplevel()

        new_top.geometry("1000x800")


        begin_value = float(self.text1.get())
        end_value = float(self.text2.get())

        begin_index = int(np.floor(begin_value/(2048*self.burst_metadata[0])))
        end_index = int(np.floor(end_value/(2048*self.burst_metadata[0]))+1)

        fig = Figure(figsize=(10, 8), dpi=100)

        extent = [begin_index*self.burst_metadata[0]*2048, end_index*self.burst_metadata[0]*2048, self.burst_metadata[2] - (self.burst_metadata[1]*self.burst_metadata[3]/2) + self.burst_metadata[1], self.burst_metadata[2] + (self.burst_metadata[1]*self.burst_metadata[3]/2)] #x, endx, y, endy

        time_array = np.linspace(begin_index*self.burst_metadata[0]*2048, end_index*self.burst_metadata[0]*2048, 2048*(end_index-begin_index))

        data = self.data[begin_index:end_index, :, :]
        new_data = np.zeros((self.burst_metadata[3], (end_index - begin_index)*2048), dtype=self.dtype)

        for index, record in enumerate(data):
            record = np.transpose(record)

            new_data[:,2048*index: (index+1)*2048] = record






        ax1 = fig.add_subplot(111)

        ax1.imshow(new_data, extent=extent, aspect='auto')#, extent=extent)

        ax1.set_ylabel("Frequency (MHz)", fontsize=15)


        divider = make_axes_locatable(ax1)
        axbottom = divider.append_axes("bottom", size=1.2, pad=0.3, sharex=ax1)

        median = np.median(np.mean(new_data, axis=0))
        med_dev = mad(np.mean(new_data, axis=0))


        axbottom.plot(time_array, (np.mean(new_data, axis=0)-median)/med_dev, marker="o", mfc="k", ms=1, mec="k") 

        axbottom.margins(x=0)

        axbottom.set_xlabel("Time from beginning of observation (s)", fontsize=15)
        axbottom.set_ylabel("Signal-to-noise ratio", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, new_top)
        canvas.get_tk_widget().grid(row=0, column=0)







    def plot_next_burst(self, index):
        
        if index==len(self.bursts):
            self.value = 0
            self.burst = self.bursts[self.value]
            self.create_frames()
        else:
            self.value = index
            self.burst = self.bursts[self.value]

            self.create_frames()




 