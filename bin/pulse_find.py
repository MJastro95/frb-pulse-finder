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


def process_acf(record, width, freq_width, time_samp, chan_width, num_chans, index, acf_array, means, dtype):
    record = np.transpose(record)


    if len(ignore) != 0:
        for value in ignore:
            value = value.split(":")

            if len(value)==1:
                record[int(value[0]),:] = 0

            else:

                begin = value[0]
                end = value[1]

                record[int(begin):int(end), :] = 0






    record_ravel = record.copy().ravel()


    record_nonzero=np.where(record.ravel()!=0)

    try:

        median = np.median(record_ravel[record_nonzero])


        med_dev = mad(record_ravel[record_nonzero])

        record_zero = np.where(record_ravel==0)

        normal_draw = np.random.normal(loc=median, scale=med_dev, size=np.shape(record_zero)[1])



        record_ravel[record_zero] = normal_draw

        record = np.reshape(record_ravel, (np.shape(record)[0], np.shape(record)[1]))

        #zero_padded_data[index, :, : ] = np.transpose(record)
    except FloatingPointError:
        median=0

        med_dev = 1

        record_zero = np.where(record_ravel==0)

        normal_draw = np.random.normal(loc=median, scale=med_dev, size=np.shape(record_zero)[1])



        record_ravel[record_zero] = normal_draw

        record = np.reshape(record_ravel, (np.shape(record)[0], np.shape(record)[1]))

        #zero_padded_data[index, :, : ] = np.transpose(record)
    #masked_record = np.ma.masked_where(record==0, record)



    median = np.median(record, axis=1)
    med_dev = mad(record, axis=1)

    try:
        bandpass_corr_record = np.transpose((np.transpose(record) - median)/med_dev)
    except FloatingPointError:
        bandpass_corr_record = np.zeros(np.shape(record))



    # acf = np.array(auto_corr2d_fft(bandpass_corr_record, np.shape(record)[1], dtype)[0])
    # acf_shape = np.shape(acf)

    # #interpolate central lag to reduce correlation resulting solely from noise correlating with itself
    # acf[int(acf_shape[0]/2), int(acf_shape[1]/2)] = (acf[int(acf_shape[0]/2)+1, int(acf_shape[1]/2)] + acf[int(acf_shape[0]/2) - 1, int(acf_shape[1]/2)] \
    #                                                 + acf[int(acf_shape[0]/2), int(acf_shape[1]/2)+1] +  acf[int(acf_shape[0]/2), int(acf_shape[1]/2)- 1])/4



    # #interpolate central frequency lag to mitigate effects of narrow band rfi
    # acf[int(acf_shape[0]/2), :] = (acf[int(acf_shape[0]/2) + 1,:] + acf[int(acf_shape[0]/2)-1,:])/2



    # mask the central frequency lag to reduce effects of narrow-band RFI in the ACF

    # center_freq_lag = int(num_chans/2)
    # center_time_lag = 1024

    # mask =  np.zeros((num_chans, 2048))
    # mask[center_freq_lag, :] = np.ones(2048)
    # mask[center_freq_lag-1, :] = np.ones(2048)


    #acf = np.ma.array(acf, mask=mask)


    # mean = np.mean(acf[center_freq_lag - int(freq_width/chan_width):center_freq_lag + int(freq_width/chan_width),\
    #                 center_time_lag - int(width/time_samp):center_time_lag + int(width/time_samp)])

    #acf_array.append(acf)
    #means.append(mean)

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
    # for frequency in freq:

    #     time_lags.append(delta(frequency, freq[0], DM)/(1000*time_samp))

    # time_lags = np.array(time_lags)
    max_lag = int(np.round(time_lags[-1]))


    dedispersed_data = np.ones((num_chans, num_time_bins - max_lag), dtype=dtype)

    for index, time_lag in enumerate(time_lags):

        bin_shift = int(np.round(time_lag))

        dedispersed_data[index, :] = spectra[index,  bin_shift: bin_shift + np.shape(dedispersed_data)[1]]
        #printProgressBar(index, np.shape(time_lags)[0])
    #print(np.shape(dedispersed_data[:, max_lag: num_time_bins]))
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





def print_candidates(all_candidates):
    total_candidates = []
    for cand_list in all_candidates:
        for candidate in cand_list:
            total_candidates.append(candidate)

    total_candidates_sorted = sorted(total_candidates, reverse=True, key= lambda total_candidates: max(total_candidates.sigma))

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

    for candidate in total_candidates_sorted:

        print(str(candidate.location) + print_string_sep(len(str(candidate.location)), 30) + "{:0.2f}".format(max(candidate.sigma)) + print_string_sep(len(str(max(candidate.sigma))), 21) + str(candidate.fluence))

        with open(outfilename + "_detected_bursts.txt", "a") as f:
            f.write(str(candidate.location) + "," + str(max(candidate.sigma)) + "," + str(candidate.fluence) + "\n")

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
    zero_padded_data = np.reshape(zero_padded_data, (int(np.shape(zero_padded_data)[0]/2048), 2048, num_chans))

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



types = {'8':np.uint8, '16':np.uint16,'32':np.uint32, '64':np.uint64}



#
#
# MAIN FUNCTION 
#
#

def main(loop_index):

    #read in fits data and parse for data
    #data is assumed to have structure of (subchunk, time, ~, freq, ~)

    if flag==0:

        if filename[-4:]=='fits':
            data, sub_int_orig, ctr_freq, chan_width, time_samp, ra_string, dec_string, tstart, dtype = fits_parse(filename)

            dtype = str(dtype)

            data = data[6000:,:,:,:,:]

            chunk_size = int(np.shape(data)[0]/split)

        #split data into smaller chunks for ease of processing
            try:
                data = data[chunk_size*(loop_index):chunk_size*(loop_index+1), :, :, :, :]
            except IndexError:
                data = data[chunk_size*(loop_index):, :, :, :, :]

            data_shape = np.shape(data)
            num_chans = data_shape[3]
            total_time_samples = data_shape[0]*data_shape[1]
        else:
            data, sub_int_orig, ctr_freq, chan_width, time_samp, ra_val, dec_val, tstart, dtype = fb.filterbank_parse(filename, loop_index, split, types[dtype])

            dtype=str(dtype)

            ra_val = str(ra_val)
            dec_val = str(dec_val)

            ra_string = ra_val[:2] + ":" + ra_val[2:4] + ":" + "{:.2f}".format(float(ra_val[4:]))
            dec_string = dec_val[:2] + ":" + dec_val[2:4] + ":" + "{:.2f}".format(float(dec_val[4:]))

            num_chans = np.shape(data)[0]
            total_time_samples = np.shape(data)[1]


        if chan_width < 0:
            chan_width = abs(chan_width)




        if maskfile:
            # read in maskfile if it has been provided and set channels to ignore

            mask = rfifind(maskfile)
            dtint, mask_chan, mask_int = mask.read_mask()

            global ignore

            mask_chan = [str(chan) for chan in mask_chan]

            ignore = ignore + mask_chan

        #we use a standardized subintegration size of 2048 time bins to make processing of differently structured 
        #data sets easier to accomplish. Thus set sub_int time to 2048*time_samp.
        #We will later split up all data into sub integratons of 2048 time bins.

        
        sub_int = 2048*time_samp
        burst_metadata = (sub_int, time_samp, ctr_freq, chan_width, num_chans, dm, ra_string, dec_string, tstart)

        # #plotting routine for replotting pickled burst images. Only executes if plotfile is given as an argument
        # if plotfile is not None:
        #     burst_metadata = (time_samp, chan_width, ctr_freq, num_chans)

        #     bursts = np.load(plotfile, allow_pickle=True)

        #     root = tk.Tk()

        #     root.title("FRB pulse find")

        #     app = gui.Application(root,  bursts,  burst_metadata, outfilename, data)
        #     app.mainloop()

        #     return


        #zero_padded_data = []

        if dm != 0: 
            print("\n Dedispersing data using DM = " + str(dm) + " ...")

            if filename[-4:]=="fits":
                all_data = np.ones((data_shape[3], data_shape[1]*data_shape[0]), dtype=types[dtype])

                for index, record in enumerate(data):
                    all_data[:, index*data_shape[1]:(index+1)*data_shape[1]] = np.transpose(record[:,0,:,0])


                dedispersed_data = np.transpose(dedisperse(all_data, dm, ctr_freq, chan_width, time_samp, types[dtype]))

            else:
                dedispersed_data = np.transpose(dedisperse(data, dm, ctr_freq, chan_width, time_samp, types[dtype]))


            print("\n Dedispersion complete!")
            del all_data
            #zero_padded_data = pad_factor_data(dedispersed_data, num_chans)

            print("\n Processing ACFs of data chunk " + str(loop_index + 1) + "...")



            acf_array = np.ma.ones((int(total_time_samples/2048), num_chans, 2048), dtype=np.float32)
            means = []

            for index in tqdm(np.arange(np.shape(dedispersed_data)[0]//2048)):
                record = dedispersed_data[index*2048:(index+1)*2048,:]
                process_acf(record, time_width, freq_width, time_samp, chan_width, num_chans, index, acf_array, means, types[dtype])

            # for index, record in enumerate(tqdm(zero_padded_data[:int(total_time_samples/2048), :, :])):

            #     process_acf(record, time_width, freq_width, time_samp, chan_width, num_chans, zero_padded_data, index, acf_array, means, types[dtype])

            print("\n\n ...processing complete!\n")


        else:


            if filename[-4:]=="fits":
                all_data = np.ones((data_shape[3], data_shape[1]*data_shape[0]), dtype=types[dtype])

                for index, record in enumerate(data):
                    all_data[:, index*data_shape[1]:(index+1)*data_shape[1]] = np.transpose(record[:,0,:,0])

                all_data = np.transpose(all_data)

                #zero_padded_data = pad_factor_data(all_data, num_chans)

            else:
                #filterbank format

                all_data = np.transpose(data)

                #zero_padded_data = pad_factor_data(all_data, num_chans)


            acf_array = np.ma.ones((int(total_time_samples/2048), num_chans, 2048), dtype=np.float32)
            means = []            

            print("\n Processing ACFs of data chunk " + str(loop_index + 1) + "...")

            for index in tqdm(np.arange(np.shape(all_data)[0]//2048)):
                record = all_data[index*2048:(index+1)*2048,:]
                process_acf(record, time_width, freq_width, time_samp, chan_width, num_chans, index, acf_array, means, types[dtype])

            # for index, record in enumerate(tqdm(zero_padded_data[:int(total_time_samples/2048), :, :])):

            #     process_acf(record, time_width, freq_width, time_samp, chan_width, num_chans, zero_padded_data, index, acf_array, means, types[dtype])

            print("\n\n ...processing complete!\n")

    center_freq_lag = int(num_chans/2)
    center_time_lag = 1024

    # # mask =  np.zeros((int(total_time_samples/2048), num_chans, 2048), dtype=np.uint8)
    # mask[:, center_freq_lag, :] = np.ones((int(total_time_samples/2048),2048), dtype=np.uint8)
    # mask[:, center_freq_lag-1, :] = np.ones((int(total_time_samples/2048),2048), dtype=np.uint8)

    # acf_array.mask = np.zeros((int(total_time_samples/2048), num_chans, 2048), dtype=np.uint8)
    # acf_array.mask[:, center_freq_lag, :] = np.ones((int(total_time_samples/2048),2048), dtype=np.uint8)
    # acf_array.mask[:, center_freq_lag-1, :] = np.ones((int(total_time_samples/2048),2048), dtype=np.uint8)

    min_t = 3
    min_f = 3

    t_wins = np.linspace(min_t, int(sub_int/2/time_samp), 10)
    f_wins = np.linspace(min_f, int(num_chans/2), 10)



    locs = []
    prune_cand_list = []
    acf_shape = np.shape(acf_array[0,:,:])
    #print(acf_shape)
    #acf_array = np.array(acf_array)



    np.seterr(all='ignore')
    for i, time in enumerate(tqdm(t_wins)):
        for j, freq in enumerate(f_wins):
            # means = []

            # for k, acf in enumerate(acf_array):
                # acf_shape = np.shape(acf)

            # plt.imshow(acf_array[0,int(acf_shape[0]/2 - time): int(acf_shape[0]/2 + time), int(acf_shape[1]/2 - freq): int(acf_shape[1]/2 + freq)], aspect='auto')
            # plt.show()

            means = (acf_array[: , int(acf_shape[0]/2 - freq):center_freq_lag -1 , int(acf_shape[1]/2 - time): int(acf_shape[1]/2 + time)].mean(axis=(1,2)) + \
                    acf_array[:, center_freq_lag + 1: int(acf_shape[0]/2 + freq) , int(acf_shape[1]/2 - time): int(acf_shape[1]/2 + time)].mean(axis=(1,2)))/2
            #means = acf_array[:,int(acf_shape[0]/2 - time): int(acf_shape[0]/2 + time), int(acf_shape[1]/2 - freq): int(acf_shape[1]/2 + freq)].mean(axis=(1, 2))


            #means = np.array(means)

            median = np.median(means)
            med_dev = mad(means)

            acf_norm = (means - median)/med_dev

    

            threshold_locs = np.where(acf_norm >= thresh_sigma)[0]

            for loc in threshold_locs:
                if dm!=0:
                    candidate = Candidate(loc, np.round(abs(acf_norm[loc]), decimals=2), \
                            np.transpose(dedispersed_data[loc*2048:(loc+1)*2048, :]), acf_array[loc], 0, burst_metadata, 0, 0, 0, True, (time, freq))
                else:
                    candidate = Candidate(loc, np.round(abs(acf_norm[loc]), decimals=2), \
                            np.transpose(all_data[loc*2048:(loc+1)*2048, :]), acf_array[loc], 0, burst_metadata, 0, 0, 0, True, (time, freq))

                if prune_candidates_modified(candidate, acf_norm, 10, sub_int):

                    if loc not in locs:
                        prune_cand_list.append(candidate)
                        locs.append(loc)
                    else:
                        index = locs.index(loc)


                        prune_cand_list[index].update_acf_windows(acf_norm[loc], time, freq)


    np.seterr(all='raise')       
    prune_cand_list = prune_candidates_windows(prune_cand_list, num_chans, 2048, min_t, min_f)




    # means = np.array(means)



    # median = np.median(means)
    # med_dev = mad(means)


    # acf_norm = (means - median)/med_dev


    # plt.scatter(np.arange(len(acf_norm)), acf_norm)
    # plt.show()

    
    # cand_list = []

    # threshold_locs = np.where(abs(acf_norm) >= thresh_sigma)[0]


    # for loc in threshold_locs:

    #     cand_list.append(Candidate(loc, np.round(abs(acf_norm[loc]), decimals=2), np.transpose(zero_padded_data[loc, :, :]), acf_array[loc], 0, burst_metadata, 0, 0, 0, True))


    # prune_cand_list = prune_candidates(cand_list, acf_norm, 10, sub_int)
    pruned_cand_sorted = sorted(prune_cand_list, reverse=True, key= lambda prune_candidate: max(prune_candidate.sigma))


    if plot==1:

        cands_to_pop = []

        for index, candidate in enumerate(pruned_cand_sorted):
            ip = interactivePlot(index, candidate)


            if ip.e==True:
                for i in np.arange(index, len(pruned_cand_sorted), 1):
                    cands_to_pop.append(i)

                break

            if ip.d == True:
                candidate.true_burst=False
                #cands_to_pop.append(index)
            else:
                candidate.gauss_fit = (ip.popt, ip.pcov)
                candidate.selected_window = (ip.xdata1, ip.xdata2, ip.ydata1, ip.ydata2)


        #pruned_cand_sorted = [cand for index, cand in enumerate(pruned_cand_sorted) if index not in cands_to_pop]


        for index, candidate in enumerate(pruned_cand_sorted):

            if candidate.gauss_fit != 0:

                if dm!=0:
                    popt = candidate.gauss_fit[0]
                    center = int(popt[0]/time_samp)

                    candidate.image = np.transpose(dedispersed_data[center - int(0.02/time_samp): center + int(0.02/time_samp),:])
                    print(np.shape(candidate.image))
                else:
                    popt = candidate.gauss_fit[0]
                    center = int(popt[0]/time_samp)
                    candidate.image = np.transpose(all_data[center - int(0.02/time_samp): center + int(0.02/time_samp),:])                





    # gain = gain #K/Jy
    # sys_temp =30 #Kelvin
    sys_flux = sys_temp/gain



    for index, candidate in enumerate(pruned_cand_sorted):

        median = np.median(np.mean(candidate.image[:num_chans,:], axis=0))
        med_dev = mad(np.mean(candidate.image[:num_chans, :], axis=0))

        jy_per_counts = sys_flux/median
        pulse_profile = np.mean(candidate.image[:num_chans, :], axis=0)

        normed_pulse_profile = pulse_profile - median

        sig_profile = np.where(normed_pulse_profile>=3*med_dev)

        length = np.shape(sig_profile)[1]

        try:
            total_snr = (np.sum(pulse_profile[sig_profile]) - length*median)/(med_dev*np.sqrt(length))
        except FloatingPointError:
            total_snr=0


        total_fluence = np.around(1000*total_snr*sys_temp*np.sqrt(length*time_samp/(chan_width*num_chans*1e6))/(gain), decimals=2)
        try:
            candidate.snr = int(np.round(total_snr, decimals=0))
            candidate.fluence = np.round(total_fluence, decimals=2)
        except ValueError:
            candidate.snr = 0
        except FloatingPointError:
            candidate.snr=0

        if filename[-4:]=="fits":
            candidate.location = np.round(candidate.location*sub_int + chunk_size*(loop_index)*sub_int_orig, decimals=2)
        else:
            candidate.location = np.round(candidate.location*sub_int + loop_index*sub_int_orig, decimals=2)

    burst_metadata = (time_samp, chan_width, ctr_freq, num_chans)

    # if plot==1:
    #     root = tk.Tk()

    #     root.title("FRB pulse find")

    #     app = gui.Application(root,  pruned_cand_sorted,  burst_metadata, outfilename, zero_padded_data, types[dtype])
    #     app.mainloop()



    if len(pruned_cand_sorted) != 0:
        np.save(outfilename + "_bursts_" + str(np.round(loop_index*sub_int_orig, decimals=2)) + "-" + str(np.round((loop_index+1)*sub_int_orig, decimals=2)), pruned_cand_sorted, allow_pickle=True)

    return 



if __name__=='__main__':
    parser = ap.ArgumentParser()

    parser.add_argument("infile", help="Fits filename containing data.")
    parser.add_argument("--sigma", help="Search for peaks in the acf above this threshold. Default = 10.", type=int, default=10)
    parser.add_argument("--d", help="Dispersion measure value to use for dedispersion. Default = 0. If zero, program assumes that the data is already dedispersed.", type=float, default=0)
    parser.add_argument("--plot", help="Use interactive plotting feature to view bursts. Default=0. Set to one for interactive plotting.", type=int, default=0)
    parser.add_argument("--plotfile", help="Pickled burst data filename to regenerate plots of bursts.")
    parser.add_argument("--split", help="Split data into this amount of chunks for processing. Helps for processing large files. Default=1", type=int, default=1)
    parser.add_argument("--gain", help="System gain of telescope. Used for calculating burst fluences.", type=float, default=1)
    parser.add_argument("--temp", help="System temperature of telescope. Used for calculating burst fluences.", type=float, default=0)
    #parser.add_argument("--bits", help="Number of bits used to sample data. Default=32.", default='32')
    parser.add_argument("--ignorechan", help="Frequency channels to ignore when processing.", type=str, nargs="+", default=None)
    parser.add_argument("--t", help="Range of time lags to take the mean of the acf in. Units of seconds.", type=float, default=0)
    parser.add_argument("--f", help="Range of frequency lags to take the mean of the acf in. Units of megahertz.", type=float, default=0)
    parser.add_argument("--no_acf_calc", help="Do not calcualte acfs. Instead load in acfs from pickled file.", type=int, default=0)
    parser.add_argument("--mask", help="PRESTO rfifind maskfile")
    parser.add_argument("outfile", help="String to append to output files.", default='')



    args = parser.parse_args()


    filename = args.infile
    thresh_sigma = args.sigma
    dm = args.d
    outfilename = args.outfile
    plot = args.plot
    plotfile = args.plotfile
    split = args.split
    gain = args.gain
    sys_temp = args.temp
    #dtype = args.bits
    ignore = args.ignorechan
    time_width = args.t
    freq_width = args.f
    flag = args.no_acf_calc
    maskfile = args.mask



    if ignore is None:
        ignore = []




    # all_candidates = []
    for i in np.arange(split):
        main(i)


    #Because python is dumb with memory management, we have to save candidates from each loop iteration to disk to avoid
    #running out of RAM. To print them to screen ranked by SNR, we now have to load them back in, and send them to 
    #the print_candidates function

    directory = '.'


    all_candidates = []
    for filename in os.listdir(directory):
        if filename.startswith(outfilename + "_bursts_"):
            candidates = np.load(filename, allow_pickle=True)
            all_candidates.append(candidates)

    
    print_candidates(all_candidates)




