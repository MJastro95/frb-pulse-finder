#!/usr/bin/env python3
# save pruned candidates to disk after completion of program
#import psutil
#import resource
import gc
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap 
from astropy.io import fits
from astropy.coordinates import Angle
from astropy import units as u
import scipy.signal as sig
from scipy.stats import median_absolute_deviation as mad
from scipy.stats import norm
from tqdm import tqdm
from scipy.optimize import curve_fit
# import tkinter as tk
#import pulse_find_gui as gui

#from multiprocessing import Pool
import filterbank as fb

from sim_burst import simulatedData

np.seterr(all='raise')



#
#
# FUNCTION DEFINITIONS TO BE USED IN MAIN
#
#
#

def print_string_sep(length, start):
    string_to_print = ''
    for num in np.arange(start - length):
        string_to_print = string_to_print + " "

    return string_to_print



def reshape(array, x, y, z):
    reshaped_array = np.zeros((x,y,z))
    for val in np.arange(x):
        reshaped_array[val, : , :] = array[:, val*z: (val + 1)*z]
    return reshaped_array

def gaussian_2d(input_tuple, mean_x, mean_y, sigma_x, sigma_y, rho, scale):
    x = input_tuple[0]
    y = input_tuple[1]
    
    x_term = ((x-mean_x)**2)/(sigma_x**2)
    y_term = ((y-mean_y)**2)/(sigma_y**2)
    
    cross_term = (2*rho*(x-mean_x)*(y-mean_y))/(sigma_x*sigma_y)
    value = (scale)*np.exp((-1/(2*(1-(rho**2))))*(x_term + y_term - cross_term))
    return value.ravel()


def auto_corr2d_fft(spec_2d, search_width, dtype):

    number_searches = np.shape(spec_2d)[1]/search_width 
    acfs = []
    for i in np.arange(number_searches):
        #start_now = datetime.now()
        spec = spec_2d[search_width*int(i):search_width*int(i+1)]


        median = np.median(spec)

        med_dev = mad(spec, axis=None)


        if med_dev ==0:
            #prevent RuntimeWarnings for bad blocks 
            #med_dev==0 only happens for blocks with all points equal to zero
            med_dev=1

        shape = np.shape(spec)


        zero_padded_spec = median*np.ones((int(3*shape[0]/2), int(3*shape[1]/2)), dtype=dtype)

        zero_padded_spec[:shape[0], :shape[1]] = spec

        shape_padded = np.shape(zero_padded_spec)


        burst_fft = np.fft.fft2((zero_padded_spec - median)/(shape[0]*shape[1]*(med_dev**2)))
        burst_fft_conj = np.conj(burst_fft)
        acf = np.real(np.fft.ifft2(burst_fft*burst_fft_conj))#/(shape[0]*shape[1])


        quadrant_1 = acf[:int(np.round(shape_padded[0]/2)), :int(np.round(shape_padded[1]/2))]
        quadrant_2 = acf[int(np.round(shape_padded[0]/2)):, :int(np.round(shape_padded[1]/2))]
        quadrant_3 = acf[int(np.round(shape_padded[0]/2)):, int(np.round(shape_padded[1]/2)):]
        quadrant_4 = acf[:int(np.round(shape_padded[0]/2)), int(np.round(shape_padded[1]/2)):]

        right_half = np.concatenate((quadrant_2, quadrant_1), axis=0)
        left_half = np.concatenate((quadrant_3, quadrant_4), axis=0)

        whole_acf = np.concatenate((left_half, right_half), axis=1)
        
        acfs.append(whole_acf[int(shape[0]/4):int(5*shape[0]/4), int(shape[1]/4): int(5*shape[1]/4)])



    return acfs


def process_acf(record, time_samp, chan_width, num_chans, index, acf_array, dtype):
    record = np.transpose(record)#.copy()


    if len(ignore) != 0:
        for value in ignore:
            value = value.split(":")

            if len(value)==1:
                record[int(value[0]),:] = 0

            else:

                begin = value[0]
                end = value[1]

                record[int(begin):int(end), :] = 0






    record_ravel = record.ravel()


    record_nonzero=np.where(record.ravel()!=0)

    try:

        median = np.median(record_ravel[record_nonzero])


        med_dev = mad(record_ravel[record_nonzero])

        record_zero = np.where(record_ravel==0)

        normal_draw = np.random.normal(loc=median, scale=med_dev, size=np.shape(record_zero)[1])



        record_ravel[record_zero] = normal_draw

        record = np.reshape(record_ravel, (np.shape(record)[0], np.shape(record)[1]))


    except FloatingPointError:
        median=0

        med_dev = 1

        record_zero = np.where(record_ravel==0)

        normal_draw = np.random.normal(loc=median, scale=med_dev, size=np.shape(record_zero)[1])



        record_ravel[record_zero] = normal_draw

        record = np.reshape(record_ravel, (np.shape(record)[0], np.shape(record)[1]))



    median = np.median(record, axis=1)
    med_dev = mad(record, axis=1)

    try:
        bandpass_corr_record = np.transpose((np.transpose(record) - median)/med_dev)
    except FloatingPointError:
        bandpass_corr_record = np.zeros(np.shape(record))

    acf_array[index, :, :] = np.ma.array(np.array(auto_corr2d_fft(bandpass_corr_record, np.shape(record)[1], dtype)[0]))



    return #acf_array #means)






def delta(f1, f2, DM):
    # f1<f2 
    #return value in ms
    return (4.148808e6*(f1**(-2) - f2**(-2))*DM)


def dedisperse(spectra, DM, ctr_freq, chan_width, time_samp, dtype):
    num_chans = np.shape(spectra)[0]
    num_time_bins = np.shape(spectra)[1]

    #freq = np.linspace(ctr_freq + (chan_width*32), ctr_freq + (chan_width*32) - num_chans*chan_width, num_chans)
    freq = np.flip(np.linspace(ctr_freq - (chan_width*num_chans/2) + chan_width/2, ctr_freq + (chan_width*num_chans/2) - chan_width/2, num_chans))


    time_lags = delta(freq, freq[0], DM)/(1000*time_samp)

    max_lag = int(np.round(time_lags[-1]))


    dedispersed_data = np.ones((num_chans, num_time_bins - max_lag), dtype=dtype)

    for index, time_lag in enumerate(time_lags):

        bin_shift = int(np.round(time_lag))

        dedispersed_data[index, :] = spectra[index,  bin_shift: bin_shift + np.shape(dedispersed_data)[1]]

    return dedispersed_data



def fits_parse(infile):
    
    with fits.open(infile) as hdu:

        index = hdu.index_of(('primary', 1))
        hdr0 = hdu[0].header
        hdr1 = hdu[1].header
        data = hdu[1].data


        tstart_days = hdr0['STT_IMJD']
        tstart_seconds = hdr0['STT_SMJD']

        tstart = tstart_days + (tstart_seconds/86400)

        sub_int = data['TSUBINT'][0]

        ctr_freq = hdr0['OBSFREQ']
        chan_width = hdr1['CHAN_BW']
        time_samp = hdr1['TBIN']

        ra_string = hdr0['RA']
        dec_string = hdr0['DEC']

        dtype = hdr0['BITPIX']


    return data['DATA'], sub_int, ctr_freq, chan_width, time_samp, ra_string, dec_string, tstart, dtype

def prune_candidates(candidates, acfs, search_length, sub_int):

    search_length = int(np.round(search_length/sub_int))
    pruned_candidates = []

    for index, candidate in enumerate(candidates):
        #print(candidate.location, candidate.sigma)
        
        if candidate.location - search_length < 0:
            acf_to_search = acfs[:candidate.location + search_length]
        elif candidate.location + search_length > np.shape(acfs):
            acf_to_search = acfs[candidate.location - search_length:] 
        else:
            acf_to_search = acfs[candidate.location- search_length: candidate.location + search_length]
            
        comp_result = abs(acf_to_search) >= thresh_sigma
        num_result = np.count_nonzero(comp_result)


        # summed = np.sum(candidate.image, axis=0)
        # # print(np.shape(summed))
        # median = np.median(summed)
        # med_dev = mad(summed)

        # norm = (summed - median)/med_dev



        if (num_result <= 5): #and (np.amax(norm)>=thresh_sigma):
            pruned_candidates.append(candidate)



    return pruned_candidates



def prune_candidates_modified(candidate, acfs, search_length, sub_int):
    search_length = int(np.round(search_length/sub_int))

    if candidate.location - search_length < 0:
        acf_to_search = acfs[:candidate.location + search_length]
    elif candidate.location + search_length > np.shape(acfs):
        acf_to_search = acfs[candidate.location - search_length:] 
    else:
        acf_to_search = acfs[candidate.location- search_length: candidate.location + search_length]


    comp_result = abs(acf_to_search) >= thresh_sigma
    num_result = np.count_nonzero(comp_result)

    if num_result <= 5:
        return True
    else:
        return False




def prune_candidates_windows(candidates, num_chans, num_samps, min_t, min_f):

    pruned_candidates = []
    for candidate in candidates:
        acf_window = candidate.acf_window

        t_widths = []
        f_widths = []
        for window in acf_window:
            t_widths.append(window[0])
            f_widths.append(window[1])

        min_t_widths = min(t_widths)
        min_f_widths = min(f_widths)

        if min_t_widths==min_t and min_f_widths==min_f:
            pruned_candidates.append(candidate)




    return pruned_candidates





def print_candidates(total_candidates_sorted):


    print("**************************************************************************")
    print("**************************************************************************")
    print("************************                           ***********************")
    print("************************ Detected burst properties ***********************")
    print("************************                           ***********************")
    print("**************************************************************************")
    print("**************************************************************************\n")
    print("**************************************************************************")
    print("Burst location (s)            ACF SNR            Burst fluence (Jyms)  ")
    print("**************************************************************************")

    for index, candidate in enumerate(total_candidates_sorted):

        print(str(candidate.location) + print_string_sep(len(str(candidate.location)), 30) + "{:0.2f}".format(max(candidate.sigma)) + print_string_sep(len(str(max(candidate.sigma))), 21) + str(candidate.fluence))

        with open(outfilename + "_detected_bursts.txt", "a") as f:
            if index==0:
                f.write("# Location (s), ACF SNR, Fluence\n")
                f.write(str(candidate.location) + "," + "{:0.2f}".format(max(candidate.sigma)) + "," + str(candidate.fluence) + "\n")
            else:
                f.write(str(candidate.location) + "," + "{:0.2f}".format(max(candidate.sigma)) + "," + str(candidate.fluence) + "\n")                

        np.save(str(outfilename) + "_" + str(np.around(candidate.location, decimals=2)) + "s_" + "burst", \
            (candidate.metadata, candidate.acf, candidate.location, candidate.image, candidate.sigma, \
                candidate.gauss_fit, candidate.selected_window, candidate.acf_window), allow_pickle=True)
        #np.save(str(outfilename) + "_image_" + str(np.around(candidate.location, decimals=2)) + "s_" + "burst", (candidate.metadata, candidate.location, candidate.image), allow_pickle=True)


    print("**************************************************************************")
    print("**************************************************************************\n")    


    if len(total_candidates_sorted)!= 0:
        np.save(outfilename + "_bursts", total_candidates_sorted, allow_pickle=True)

    return 


def pad_factor_data(data, num_chans):

    log_2 = int(np.ceil(np.log2(np.shape(data)[0])))
    num_of_zeros = 2**log_2 - np.shape(data)[0]

    zeros = np.zeros((num_of_zeros, num_chans), dtype=np.uint8)

    zero_padded_data = np.concatenate((data, zeros), axis=0)
    zero_padded_data = np.reshape(zero_padded_data, (int(np.shape(zero_padded_data)[0]/sub), sub, num_chans))

    return zero_padded_data




#
#
#CLASS DEFINITIONS
#
#

class Candidate:
    def __init__(self, location, sigma, image, acf, fluence, metadata, snr, gauss_fit, selected_window, true_burst, acf_window):
        self.location = location
        self.sigma = [sigma]
        self.acf_window = [(acf_window[0], acf_window[1])]
        self.image = image
        self.acf = acf
        self.fluence = fluence
        self.metadata = metadata
        self.snr = snr
        self.gauss_fit = gauss_fit
        self.selected_window = selected_window
        self.true_burst = true_burst

    def update_acf_windows(self, sigma, time, freq):
        self.sigma.append(sigma)
        self.acf_window.append((time, freq))





class interactivePlot:
    def __init__(self, index, candidate):
        self.index = index
        self.candidate = candidate
        self.cands_to_pop = []
        self.makePlot()
        self.true_burst = False


    def makePlot(self):

        self.data = self.candidate.image
        self.metadata = self.candidate.metadata
        self.acf_window = self.candidate.acf_window

        sub_int = self.metadata[0]
        time_samp = self.metadata[1]
        ctr_freq = self.metadata[2]
        chan_width = self.metadata[3]
        num_chan = self.metadata[4]
        dm = self.metadata[5]
        ra = self.metadata[6]
        dec = self.metadata[7]
        tstart = self.metadata[8]

        sigma = self.candidate.sigma
        bandwidth = num_chan*chan_width


        self.where_max = sigma.index(max(sigma))

        self.acf_window_max = self.acf_window[self.where_max]

        self.loc = self.candidate.location
        width = 10
        height = 5.63
        self.fig = plt.figure(figsize=(width, height), dpi=100)
        self.ax1 = self.fig.add_axes([0.1, 0.1, 0.8, 0.65])
        
        self.ax1.set_ylabel("Frequency (MHz)", fontsize=12)

        divider = make_axes_locatable(self.ax1)
        time_array = np.linspace(self.loc*sub_int, (self.loc+1)*sub_int, np.shape(self.data)[1])
        axbottom1 = divider.append_axes("bottom", size=1, pad=0.4, sharex=self.ax1)

        mean = np.mean(self.data, axis=0)

        median = np.median(mean)
        med_dev = mad(mean)

        normed_mean = (mean-median)/med_dev

        axbottom1.plot(time_array, normed_mean, marker="o", mfc="k", ms=1, mec="k")
        axbottom1.set_xlabel("Time (s)", fontsize=12)
        self.extent = [self.loc*sub_int, (self.loc+1)*sub_int, ctr_freq - bandwidth/2, ctr_freq + bandwidth/2]

        self.ax1.imshow(self.data, aspect='auto', extent=self.extent)

        self.ax1.text(0.5, height - 0.25, "Metadata for observation:", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(0.5, height - 0.50, "Right Ascension (hms): " + ra, fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(0.5, height - 0.75, "Declination (dms): " + dec, fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(0.5, height - 1, "Start Time (MJD): {:.2f}".format(tstart),fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(0.5, height - 1.25, "Sampling time: {:.2e}".format(time_samp) + " s", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(4.5, height - 0.25, "Channel Width: {:.2f}".format(chan_width) + " MHz", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(4.5, height - 0.50, "Center Frequency: {:.2f}".format(ctr_freq) + " MHz", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(4.5, height - 0.75, "Total Bandwidth: {:.0f}".format(bandwidth) + " MHz", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(4.5, height - 1, "Dispersion Measure: {:.2f}".format(dm) + " pc cm$^{-3}$", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(4.5, height - 1.25, "SNR of ACF: {:.2f}".format(max(sigma)), fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(8.5, height - 0.5, "Time window width: {:.2f}".format(2*self.acf_window_max[0]*time_samp*1000) + " ms", fontsize=12, transform=self.fig.dpi_scale_trans)
        self.ax1.text(8.5, height - 0.75, "Frequency window width: {:.2f}".format(2*self.acf_window_max[1]*chan_width) + " MHz", fontsize=12, transform=self.fig.dpi_scale_trans)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.crel = self.fig.canvas.mpl_connect('button_release_event', self.onrel)
        self.keyPress = self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)

        self.b = False
        self.e = False
        self.d = False
        plt.show()

    def onKeyPress(self, event):
        if event.key=='b':
            #begin selection
            self.b = True
            print("Please make selection by dragging mouse from start point to end point.\n")

        if event.key=='d':
            #delete candidate
            self.d = True
            plt.close()

        if event.key=='c' and self.b==True:
            #confirm selected region

            print("Selection confirmed! Fitting 2d gaussian and plotting...\n")
            plt.close()

            self.fit2dGauss()

        elif event.key=='c' and self.b==False:
            print("No region has been selected! Please press 'b' to begin selecting region and 'c' to confirm\n")

        if event.key=='r' and self.b==True:
            #clear and reselect region

            self.xdata1 = 0
            self.ydata1 = 0
            self.xdata2 = 0
            self.ydata2 = 0

            self.b = False

            print("Clearing selected region. Please press 'b' to begin selection, 'c' to confirm selection, and 'r' to reselect.\n")

        elif event.key=='r' and self.b==False:
            print("No region selected! Please press 'b' to begin selection, 'c' to confirm selection, and 'r' to reselect.\n")

        if event.key=='e':
            self.e = True 
            plt.close()

        return

    def onpress(self, event):

        if self.b == True:
            self.xdata1 = event.xdata
            self.ydata1 = event.ydata
        else:
            print("Please press 'b' to begin selecting region.\n")

        return

    def onrel(self, event):

        if self.b==True:
            self.xdata2 = event.xdata
            self.ydata2 = event.ydata

            self.printData()

        return

    def printData(self):
        print("The selected time range is (begin, end): ")
        print("{:.2f}".format(self.xdata1) + " s", "{:.2f}".format(self.xdata2) + " s")

        print("The selected frequency range is (begin, end): ")
        print("{:.2f}".format(self.ydata1) + " MHz", "{:.2f}".format(self.ydata2) + " MHz\n")

        print("Press 'c' to confirm this selection, or press 'r' to reselect.\n")


    def fit2dGauss(self):
        sub_int = self.metadata[0]
        time_samp = self.metadata[1]
        ctr_freq = self.metadata[2]
        chan_width = self.metadata[3]
        num_chan = self.metadata[4]
        dm = self.metadata[5]
        ra = self.metadata[6]
        dec = self.metadata[7]
        tstart = self.metadata[8]

        bandwidth = num_chan*chan_width
        top_freq = ctr_freq + (chan_width*num_chan/2) + (chan_width/2)


        start_time= int((self.xdata1 - self.loc*sub_int)//time_samp)
        end_time = int((self.xdata2 - self.loc*sub_int)//time_samp)

        start_freq = int((top_freq - self.ydata1)//chan_width)
        end_freq = int((top_freq - self.ydata2)//chan_width)

        time_array = np.linspace(self.xdata1, self.xdata2, end_time - start_time)
        freq_array = np.linspace(self.ydata1, self.ydata2, end_freq - start_freq)

        x,y = np.meshgrid(time_array, freq_array)

        data_to_fit = self.data[start_freq: end_freq, start_time:end_time]

        median = np.median(data_to_fit)


        middle_time = time_array[int(np.shape(time_array)[0]/2)]
        middle_freq = freq_array[int(np.shape(freq_array)[0]/2)]


        max_bandwidth = abs(self.ydata2 - self.ydata1)
        max_timewidth = self.xdata2 - self.xdata1

        max_value = np.amax(data_to_fit)

        print(max_bandwidth, max_timewidth, max_value)
        print(time_array[0], time_array[-1])
        print(freq_array[-1], freq_array[0])


        self.popt, self.pcov = curve_fit(gaussian_2d, (x,y), data_to_fit.ravel(), p0=[middle_time, middle_freq, max_timewidth/4, max_bandwidth/4, -0.5, max_value] , bounds=([time_array[0], freq_array[-1], 0, 0, -1, 0],[time_array[-1], freq_array[0], max_timewidth/2, max_bandwidth/2, 0, np.inf]))
        
        print(self.popt)

        loc_x = self.popt[0]
        loc_y = self.popt[1]
        sigma_x = self.popt[2]
        sigma_y = self.popt[3]
        rho = self.popt[4]



        width = 8
        height = 6
        self.fig2 = plt.figure(figsize=(width, height), dpi=100)
        self.ax2 = self.fig2.add_axes([0.1, 0.1, 0.8, 0.7])

        extent = [time_array[0], time_array[-1], freq_array[-1], freq_array[0]]
        self.ax2.imshow(data_to_fit, aspect='auto', extent=extent, cmap='Greys')
        self.ax2.contour(x, y, gaussian_2d((x, y), *self.popt).reshape(end_freq - start_freq, end_time - start_time))

        self.ax2.text(0.5, height - 0.25, "Fit parameters:", fontsize=12, transform=self.fig2.dpi_scale_trans)
        self.ax2.text(0.5, height - 0.5, "Time location: {:0.3f}".format(loc_x) + " s", fontsize=12, transform=self.fig2.dpi_scale_trans)
        self.ax2.text(0.5, height - 0.75, "Frequency location: {:0.3f}".format(loc_y) + " MHz", fontsize=12, transform=self.fig2.dpi_scale_trans)        
        self.ax2.text(4.5, height - 0.5, "Time sigma: {:0.3f}".format(sigma_x) + " s", fontsize=12, transform=self.fig2.dpi_scale_trans)
        self.ax2.text(4.5, height - 0.75, "Frequency sigma: {:0.3f}".format(sigma_y) + " MHz", fontsize=12, transform=self.fig2.dpi_scale_trans)

        self.ax2.set_xlabel("Time (s)", fontsize=12)
        self.ax2.set_ylabel("Frequency (MHz)", fontsize=12)
        self.btnPress = self.fig2.canvas.mpl_connect('key_press_event', self.gaussPress)

        plt.show()

    def gaussPress(self, event):
        if event.key=='c':
            print("Fit confirmed!")
            self.true_burst = True

            plt.close()
        elif event.key=='d':
            print("Deleting fit parameters")
            plt.close()




class rfifind(object):
    def __init__(self, filename):
        self.basename = filename[:filename.find("_rfifind.")+8]
        #self.idata = infodata.infodata(self.basename+".inf")
        self.read_stats()
        # self.read_mask()
        #self.get_bandpass()
        # if len(self.goodints):
        #     self.get_median_bandpass()
        #     self.determine_padvals()

    def read_stats(self):
        x = open(self.basename+".stats")
        self.nchan, self.nint, self.ptsperint, self.lobin, self.numbetween = \
                    np.fromfile(x, dtype=np.int32, count=5)
        count = self.nchan * self.nint
        self.pow_stats = np.fromfile(x, dtype=np.float32, count=count)
        self.avg_stats = np.fromfile(x, dtype=np.float32, count=count)
        self.std_stats = np.fromfile(x, dtype=np.float32, count=count)
        self.pow_stats.shape = (self.nint, self.nchan)
        self.avg_stats.shape = (self.nint, self.nchan)
        self.std_stats.shape = (self.nint, self.nchan)
        x.close()

    def read_mask(self):
        x = open(self.basename+".mask")
        self.time_sig, self.freq_sig, self.MJD, self.dtint, self.lofreq, self.df = \
                       np.fromfile(x, dtype=np.float64, count=6)
        self.freqs = np.arange(self.nchan)*self.df + self.lofreq
        self.times = np.arange(self.nint)*self.dtint
        self.MJDs = self.times/86400.0 + self.MJD
        nchan, nint, ptsperint = np.fromfile(x, dtype=np.int32, count=3)
        nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
        if nzap:
            self.mask_zap_chans = np.fromfile(x, dtype=np.int32, count=nzap)
        else:
            self.mask_zap_chans = np.asarray([])
        self.mask_zap_chans = set(self.mask_zap_chans)
        if len(self.mask_zap_chans)==self.nchan:
            print("WARNING!:  All channels recommended for masking!")
        nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
        if nzap:
            self.mask_zap_ints = np.fromfile(x, dtype=np.int32, count=nzap)
        else:
            self.mask_zap_ints = np.asarray([])

        self.mask_zap_ints = set(self.mask_zap_ints)
        if len(self.mask_zap_ints)==self.nint:
            print("WARNING!:  All intervals recommended for masking!")
        nzap_per_int = np.fromfile(x, dtype=np.int32, count=nint)
        self.mask_zap_chans_per_int = []
        for nzap in nzap_per_int:
            if nzap:
                tozap = np.fromfile(x, dtype=np.int32, count=nzap)
            else:
                tozap = np.asarray([])
            self.mask_zap_chans_per_int.append(tozap)
        x.close()

        return (self.dtint, self.mask_zap_chans, self.mask_zap_ints)




class Burst:
    def __init__(self, location, snr, drift_rate):
        self.location = location
        self.snr= snr
        self.drift_rate = drift_rate



types = {'8':np.uint8, '16':np.uint16,'32':np.float32, '64':np.float64}



#
#
# MAIN FUNCTION 
#
#

def main():

    #read in fits data and parse for data
    #data is assumed to have structure of (subchunk, time, ~, freq, ~)

    if flag==0:

        if filename[-4:]=='fits':
            data, sub_int_orig, ctr_freq, chan_width, time_samp, ra_string, dec_string, tstart, dtype = fits_parse(filename)

            dtype = str(dtype)



            data_shape = np.shape(data)

            num_chans = data_shape[3]
            orig_time_samples = data_shape[0]*data_shape[1]
        else:
            data, sub_int_orig, ctr_freq, chan_width, time_samp, ra_val, dec_val, tstart, dtype = fb.filterbank_parse(filename, 0, 1)

            dtype=str(dtype)

            ra_val = str(ra_val)
            dec_val = str(dec_val)

            ra_string = ra_val[:2] + ":" + ra_val[2:4] + ":" + "{:.2f}".format(float(ra_val[4:]))
            dec_string = dec_val[:2] + ":" + dec_val[2:4] + ":" + "{:.2f}".format(float(dec_val[4:]))

            num_chans = np.shape(data)[0]
            orig_time_samples = np.shape(data)[1]

    else:
        # create simulated data
        np.seterr(all='ignore')
        sim_data = simulatedData(64, sub*500, 12.5, 64e-6, 1700)
        sim_data.add_bursts(10)
        sim_data.add_rfi()
        np.seterr(all='raise')

        data = sim_data.data
        num_chans = sim_data.nchan 
        time_samp = sim_data.tsamp 
        chan_width = sim_data.chan_width
        ctr_freq = sim_data.hifreq - num_chans*chan_width
        ra_string = "sim data"
        dec_string= "sim data"
        tstart = 0
        total_time_samples = sim_data.nbins
        dtype = '32'
        sub_int_orig = sub*time_samp

    if chan_width < 0:
        chan_width = abs(chan_width)



    mask_chan = []
    if maskfile:
        # read in maskfile if it has been provided and set channels to ignore

        mask = rfifind(maskfile)
        dtint, mask_chan, mask_int = mask.read_mask()

        global ignore


        mask_chan = [str(chan) for chan in mask_chan]

        for val in ignore:
            value = val.split(":")
            if len(value)==1:
                mask_chan.append(str(value[0]))
            else:
                values = list(np.arange(int(value[0]), int(value[1]), int(value[2])))

                for num in values:
                    mask_chan.append(str(numf))


        ignore = ignore + mask_chan

        if len(mask_chan)/num_chans >= 0.5:
            print("More than half the bandwidth is masked! Aborting...")
            with open("aborted_runs.txt", "a") as f:
                f.write(filename + " aborted because {:0.0f}".format(100*len(mask_chan)/num_chans) + " percent of channels are masked\n")

            sys.exit()
    else:
        for val in ignore:
            value = val.split(":")
            if len(value)==1:
                mask_chan.append(str(value[0]))
            else:
                values = list(np.arange(int(value[0]), int(value[1]), int(value[2])))

                for num in values:
                    mask_chan.append(str(num))


    mask_chan = set(mask_chan)



    #we use a standardized subintegration size of sub time bins to make processing of differently structured 
    #data sets easier to accomplish. Thus set sub_int time to sub*time_samp.
    #We will later split up all data into sub integratons of sub time bins.

    
    sub_int = sub*time_samp
    burst_metadata = (sub_int, time_samp, ctr_freq, chan_width, num_chans, dm, ra_string, dec_string, tstart)


    if dm != 0: 
        print("\n Dedispersing data using DM = " + str(dm) + " ...")

        if filename[-4:]=="fits":

            all_data = np.ones((data_shape[3], data_shape[1]*data_shape[0]), dtype=types[dtype])

            for index, record in enumerate(data):
                all_data[:, index*data_shape[1]:(index+1)*data_shape[1]] = np.transpose(record[:,0,:,0])

            if interval is None:
                dedispersed_data = np.transpose(dedisperse(all_data, dm, ctr_freq, chan_width, time_samp, types[dtype]))
            else:
                dedispersed_data = np.transpose(dedisperse(all_data[:, int(np.shape(all_data)[1]*interval[0]): int(np.shape(all_data)[1]*interval[1])],\
                        dm, ctr_freq, chan_width, time_samp, types[dtype]))

            total_time_samples = np.shape(dedispersed_data)[0]
            del all_data
            del data
        else:
            if interval is None:
                dedispersed_data = np.transpose(dedisperse(data, dm, ctr_freq, chan_width, time_samp, types[dtype]))
            else:
                dedispersed_data = np.transpose(dedisperse(data[:, int(np.shape(data)[1]*interval[0]): int(np.shape(data)[1]*interval[1])],\
                        dm, ctr_freq, chan_width, time_samp, types[dtype]))

            total_time_samples = np.shape(dedispersed_data)[0]
            del data

        print("\n Dedispersion complete!")



        print("\n Processing ACFs of data chunk...")



        acf_array = np.ma.ones((int(total_time_samples/sub), num_chans, sub), dtype=np.float32)
        means = []

        for index in tqdm(np.arange(np.shape(dedispersed_data)[0]//sub)):
            record = dedispersed_data[index*sub:(index+1)*sub,:]
            process_acf(record, time_samp, chan_width, num_chans, index, acf_array, types[dtype])



        print("\n\n ...processing complete!\n")


    else:


        if filename[-4:]=="fits":
            all_data = np.ones((data_shape[3], data_shape[1]*data_shape[0]), dtype=types[dtype])

            for index, record in enumerate(data):
                all_data[:, index*data_shape[1]:(index+1)*data_shape[1]] = np.transpose(record[:,0,:,0])

            all_data = np.transpose(all_data)



        else:
            #filterbank format

            all_data = np.transpose(data)





        acf_array = np.ma.ones((int(total_time_samples/sub), num_chans, sub), dtype=np.float32)
        means = []            

        print("\n Processing ACFs of data chunk...")

        for index in tqdm(np.arange(np.shape(all_data)[0]//sub)):
            record = all_data[index*sub:(index+1)*sub,:]
            process_acf(record, time_samp, chan_width, num_chans, index, acf_array, types[dtype])

        print("\n\n ...processing complete!\n")

    center_freq_lag = int(num_chans/2)
    center_time_lag = 1024



    acf_array.mask = np.zeros((int(total_time_samples/sub), num_chans, sub), dtype=np.uint8)
    acf_array.mask[:, center_freq_lag, :] = np.ones((int(total_time_samples/sub),sub), dtype=np.uint8)
    acf_array.mask[:, center_freq_lag-1, :] = np.ones((int(total_time_samples/sub),sub), dtype=np.uint8)
    acf_array.mask[:, center_freq_lag+1, :] = np.ones((int(total_time_samples/sub),sub), dtype=np.uint8)

    min_t = 1
    min_f = 3

    t_wins = np.logspace(np.log2(min_t), np.log2(int(sub_int/2/time_samp)), 10, base=2)
    f_wins = np.linspace(min_f, int(num_chans/2), 10)




    locs = set({})
    cand_dict = {}
    acf_shape = np.shape(acf_array[0,:,:])



    np.seterr(all='ignore')

    print("Calculating ACF means...")
    for i, time in enumerate(tqdm(t_wins)):
        for j, freq in enumerate(f_wins):


            means = acf_array[:,int(acf_shape[0]/2 - freq): int(acf_shape[0]/2 + freq), int(acf_shape[1]/2 - time): int(acf_shape[1]/2 + time)].mean(axis=(1, 2))




            median = np.median(np.ma.getdata(means))
            med_dev = mad(means)

            acf_norm = (means - median)/med_dev

    

            threshold_locs = np.where(acf_norm >= thresh_sigma)[0]

            for loc in threshold_locs:
                if loc not in locs:
                    if dm!=0:
                        burst = np.ma.array(np.transpose(dedispersed_data[loc*sub:(loc+1)*sub, :]))
                        burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)
                        for chan in mask_chan:
                            burst[int(chan), :] = np.ones(np.shape(burst)[1], dtype=np.uint8)
                        candidate = Candidate(loc, np.round(abs(acf_norm[loc]), decimals=2), \
                                burst, acf_array[loc], 0, burst_metadata, 0, 0, 0, False, (time, freq))
                    else:
                        burst = np.ma.array(np.transpose(all_data[loc*sub:(loc+1)*sub, :]))
                        burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)
                        for chan in mask_chan:
                            burst[int(chan), :] = np.ones(np.shape(burst)[1], dtype=np.uint8)

                        candidate = Candidate(loc, np.round(abs(acf_norm[loc]), decimals=2), \
                                burst, acf_array[loc], 0, burst_metadata, 0, 0, 0, False, (time, freq))

                    cand_dict[loc] = candidate
                    locs.add(loc)

                else:

                    cand_dict[loc].update_acf_windows(acf_norm[loc], time, freq)





    cand_list = [cand_dict[key] for key in cand_dict]
    np.seterr(all='raise')   


    prune_cand_list = []
    for candidate in cand_list:

        sigma_max = candidate.sigma.index(max(candidate.sigma))
        acf_window_where = candidate.acf_window[sigma_max]

        max_t = acf_window_where[0]*time_samp

        t_windows = [window[0] for window in candidate.acf_window]
        f_windows = [window[1] for window in candidate.acf_window]

        min_f_window = min(f_windows)

        if (max_t <= prune_value/1000) and (len(candidate.sigma)!= 1) and min_f_window<=5:

            prune_cand_list.append(candidate)


    pruned_cand_sorted = sorted(prune_cand_list, reverse=True, key= lambda prune_candidate: max(prune_candidate.sigma))


    if plot==1:

        cands_to_pop = []

        for index, candidate in enumerate(pruned_cand_sorted):
            ip = interactivePlot(index, candidate)


            if ip.e==True:
                for i in np.arange(index, len(pruned_cand_sorted), 1):
                    cands_to_pop.append(i)

                break

            # if ip.true_burst==True:
            #     candidate.true_burst=True

            if ip.d == True:
                candidate.true_burst=False
            else:
                candidate.gauss_fit = (ip.popt, ip.pcov)
                candidate.selected_window = (ip.xdata1, ip.xdata2, ip.ydata1, ip.ydata2)




        for index, candidate in enumerate(pruned_cand_sorted):

            if candidate.gauss_fit != 0:

                candidate.true_burst=True

                print(candidate.true_burst)
                if dm!=0:
                    popt = candidate.gauss_fit[0]
                    center = int(popt[0]/time_samp)

                    candidate.image = np.transpose(dedispersed_data[center - int(0.02/time_samp): center + int(0.02/time_samp),:])

                else:
                    popt = candidate.gauss_fit[0]
                    center = int(popt[0]/time_samp)
                    candidate.image = np.transpose(all_data[center - int(0.02/time_samp): center + int(0.02/time_samp),:])                



    for cand in pruned_cand_sorted:
        if interval is None:
            offset = 0
        else:
            offset = orig_time_samples*interval[0]*time_samp
        if filename[-4:]=="fits":
            cand.location = np.round(cand.location*sub_int + offset , decimals=2)

        else:
            cand.location = np.round(cand.location*sub_int + offset, decimals=2)


    # gain = gain #K/Jy
    # sys_temp =30 #Kelvin
    sys_flux = sys_temp/gain



    # for index, candidate in enumerate(pruned_cand_sorted):

    #     median = np.median(np.mean(candidate.image[:num_chans,:], axis=0))
    #     med_dev = mad(np.mean(candidate.image[:num_chans, :], axis=0))

    #     jy_per_counts = sys_flux/median
    #     pulse_profile = np.mean(candidate.image[:num_chans, :], axis=0)

    #     normed_pulse_profile = pulse_profile - median

    #     sig_profile = np.where(normed_pulse_profile>=3*med_dev)

    #     length = np.shape(sig_profile)[1]

    #     try:
    #         total_snr = (np.sum(pulse_profile[sig_profile]) - length*median)/(med_dev*np.sqrt(length))
    #     except FloatingPointError:
    #         total_snr=0


    #     total_fluence = np.around(1000*total_snr*sys_temp*np.sqrt(length*time_samp/(chan_width*num_chans*1e6))/(gain), decimals=2)
    #     try:
    #         candidate.snr = int(np.round(total_snr, decimals=0))
    #         candidate.fluence = np.round(total_fluence, decimals=2)
    #     except ValueError:
    #         candidate.snr = 0
    #     except FloatingPointError:
    #         candidate.snr=0

    burst_metadata = (time_samp, chan_width, ctr_freq, num_chans)



    if flag==1:
        with open(outfilename + "_simulated_bursts.txt", "a") as f:
            for index, loc in enumerate(sim_data.timeloc):
                if index==0:
                    f.write("# Location simulated burst (s), Peak Scale\n")
                    f.write("{:0.2f}".format(loc) + "," + "{:0.2f}".format(sim_data.peak_snrs[index]) + "\n")
                else:
                    f.write("{:0.2f}".format(loc) + "," + "{:0.2f}".format(sim_data.peak_snrs[index]) + "\n")

        with open(outfilename + "_bursts.txt", "a") as f:
            for index, candidate in enumerate(pruned_cand_sorted):



                has_window = 0
                if (min_t, min_f) in candidate.acf_window:
                    has_window=1

                if index==0:
                    f.write("# Location detected burst (s), ACF SNR, has min window\n")
                    f.write("{:0.2f}".format(candidate.location) + "," + "{:0.2f}".format(max(candidate.sigma)) + "," + "{}".format(has_window) + "\n")
                else:
                    f.write("{:0.2f}".format(candidate.location) + "," + "{:0.2f}".format(max(candidate.sigma)) + "," + "{}".format(has_window) + "\n")                

    print_candidates(pruned_cand_sorted)



    return 



if __name__=='__main__':
    parser = ap.ArgumentParser()

    parser.add_argument("--infile", help="Fits filename containing data.")
    parser.add_argument("--sigma", help="Search for peaks in the acf above this threshold. Default = 10.", type=int, default=10)
    parser.add_argument("--d", help="Dispersion measure value to use for dedispersion. Default = 0. If zero, program assumes that the data is already dedispersed.", type=float, default=0)
    parser.add_argument("--plot", help="Use interactive plotting feature to view bursts. Default=0. Set to one for interactive plotting.", type=int, default=0)
    parser.add_argument("--plotfile", help="Pickled burst data filename to regenerate plots of bursts.")
    parser.add_argument("--gain", help="System gain of telescope. Used for calculating burst fluences.", type=float, default=1)
    parser.add_argument("--temp", help="System temperature of telescope. Used for calculating burst fluences.", type=float, default=0)
    parser.add_argument("--ignorechan", help="Frequency channels to ignore when processing.", type=str, nargs="+", default=None)
    parser.add_argument("--sim_data", help="Run ACF analysis on simulated data. Default=0: do not run simulated data.", type=int, default=0)
    parser.add_argument("--mask", help="PRESTO rfifind maskfile")
    parser.add_argument("--prune", help="All candidates with max SNR occuring at time window above this value will be pruned. Default=10 ms", type=float, default=10)
    parser.add_argument("--sub_int", help="Length of desired sub-integration in time bins. The data is split into chunks of this size in time before the ACF is calculated. Default=2048", type=int, default=2048)
    parser.add_argument("--interval", help="Start and end time of data to process as fraction of observation length. Default is to process entire observation.)", nargs=2, type=float, default=None)
    parser.add_argument("outfile", help="String to append to output files.", default='')



    args = parser.parse_args()


    filename = args.infile
    thresh_sigma = args.sigma
    dm = args.d
    outfilename = args.outfile
    plot = args.plot
    plotfile = args.plotfile
    gain = args.gain
    sys_temp = args.temp
    ignore = args.ignorechan
    flag = args.sim_data
    maskfile = args.mask
    prune_value = args.prune
    sub = args.sub_int
    interval = args.interval

    if flag==1:
        filename="dummy"


    if ignore is None:
        ignore = []






    main()






