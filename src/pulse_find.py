#!/usr/bin/env python3

import gc
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use("TkAgg")
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
import filterbank as fb
from sim_burst import simulatedData
from low_pass import LowPassFilter

np.seterr(all='raise')



#
#
# FUNCTION DEFINITIONS TO BE USED IN MAIN
#
#
#

def print_string_sep(length, start):
    """
    Return string with start - length blank spaces

    Arguments:
        length -- length in characters between items on terminal window
        start -- start in characters 

    Returns:
        String with start - length blank spaces
    """

    string_to_print = ''
    for num in np.arange(start - length):
        string_to_print = string_to_print + " "

    return string_to_print


def gaussian_2d(input_tuple, mean_x, mean_y, 
                sigma_x, sigma_y, rho, scale):
    """
    Return a 2d dimensional gaussian function flattened along axis.

    Arguments:
        input_tuple -- tuple of the form (x,y) where x,y are 
                       np.meshgrid like arrays
        mean_x -- x-axis location parameter of gaussian
        mean_y -- y-axis location parameter of gaussian
        sigma_x -- x-axis shape parameter of gaussian
        sigma_y -- y-axis shape paramter of gaussian
        rho -- correlation coefficient between x an y (between -1 and 1)
        scale -- scaling parameter controlling max value of gaussian

    Returns:
        A flattened version of the two-dimensional gaussian with the given inputs

    Exceptions:
        ValueError -- if absolute value of rho exceeds one
    """

    if abs(rho) > 1:
        raise ValueError('abs(rho) should be less than one.')

    x = input_tuple[0]
    y = input_tuple[1]
    
    x_term = ((x-mean_x)**2)/(sigma_x**2)
    y_term = ((y-mean_y)**2)/(sigma_y**2)
    
    cross_term = (2*rho*(x-mean_x)*(y-mean_y))/(sigma_x*sigma_y)
    value = (scale)*np.exp((-1/(2*(1-(rho**2))))*(x_term + y_term - cross_term))
    return value.ravel()


def auto_corr2d_fft(spec_2d, search_width, dtype):
    """
    Return 2d autocorrelation function of input array using numpy fft2.

    Arguments:
        spec_2d -- two dimensional input array with axes structure (freq, time)
        search_width -- break spec_2d into chunks of this length in time. 
                    Should always be equal to length of time axis of spec_2d.
        dtype -- a valid numpy datatype to use for the return array.

    Returns:
        A list of acfs, with length equal to np.shape(spec_2d)[1]/search_width.
        If search_width is set to length of time axis of spec_2d, as recommended,
        then the return list is length 1.
    """

    number_searches = np.shape(spec_2d)[1]/search_width 
    acfs = []
    for i in np.arange(number_searches):

        spec = spec_2d[search_width*int(i):search_width*int(i+1)]

        # fig = plt.figure(figsize=(10,8), dpi=100)
        # ax1 = fig.add_axes([0.2, 0.05, 0.75, 0.9])
        # ax2 = fig.add_axes([0.05, 0.05, 0.1, 0.9])
        # ax1.imshow(spec, aspect='auto')
        # ax2.plot(np.flip(np.mean(spec, axis=1)), np.arange(64))
        # plt.show()

        median = np.median(spec)
        med_dev = mad(spec, axis=None)

        if med_dev ==0:
            #prevent RuntimeWarnings for bad blocks 
            #med_dev==0 only happens for blocks with all points equal to zero
            med_dev=1

        shape = np.shape(spec)
        zero_padded_spec = median*np.ones((int(3*shape[0]/2), int(3*shape[1]/2))
                                            , dtype=dtype)

        zero_padded_spec[:shape[0], :shape[1]] = spec
        shape_padded = np.shape(zero_padded_spec)

        burst_fft = np.fft.fft2(((zero_padded_spec) - median)/(med_dev))
        burst_fft_conj = np.conj(burst_fft)

        power_spec = burst_fft*burst_fft_conj



        # lpf = LowPassFilter(power_spec, 10, 16)
        # lpf.reshape_power_spec()

        # plt.imshow(np.real(lpf.reshaped_power_spec), aspect='auto')
        # plt.show()

        # filt_powspec = lpf.apply_filter(filt='gauss')

        # plt.imshow(np.real(lpf.filt_pow_spec), aspect='auto')
        # plt.show()


        acf = np.real(np.fft.ifft2(power_spec)/(shape[0]*shape[1]))

        # The 2d acf returned via a the fft2 method is the same shape as the
        # input array. However, it is not automatically structured correctly,
        # so we need to rearrange it. 


        # +++++++++++      
        # +  1 +  4 +    
        # +    +    +    
        # +++++++++++ 
        # +  2 +  3 +
        # +    +    +
        # +++++++++++

        # The 2d acf is layed out like the square above.

        # Quadrant 1 -- contains positive lags in time and frequency.
        # Quadrant 2 -- contains negative lags, ordered in reverse, 
        #               of frequency. Contains positive lags in time.
        # Quadrant 3 -- contains negative lags in time and frequency, ordered in
        #               reverse.
        # Quadrant 4 -- contains postive lags in frequency, negative lags
        #               time, ordered in reverse.

        # Therefore, we need to reorder the acf such that it looks like the
        # following square.

        # +++++++++++   
        # +  3 +  2 +    
        # +    +    +    
        # +++++++++++ 
        # +  4 +  1 +
        # +    +    +
        # +++++++++++

        # In addition, zero padding the ACF with N amount of zeros provides 
        # 2N + 1 "good" lags (N positive lags, N negative lags, zero lag).
        # Good here means that the calculation is not corrupted by the signal
        # "wrapping" around on itself (a result of the fast fourier transform
        # see e.g. Numerical Recipes by Press, Teukolsky, Vetterling 
        # and Flannery for a discussion).
        #   There are therefore K= M-(2N + 1) bad lags, where M is the shape
        # of the zero padded array. Moreover, the bad lags occur at positions
        # N+1 through N+K, as elements 0 is the zero lag, elements 1 through N
        # are the good positive lags, and elements -1 through -N are the good 
        # negative lags. So we need to be careful when indexing the output of
        # the fft method to only grab the good lags.

        # Since we zero pad with N=shape(data)/2 zeros, we grab the data like
        # below. 
        quadrant_1 = acf[:int(shape[0]/2)+1, 
                            :int(shape[1]/2) + 1]

        quadrant_2 = acf[-int(shape[0]/2):, 
                            :int(shape[1]/2) + 1]

        quadrant_3 = acf[-int(shape[0]/2):, 
                         -int(shape[1]/2):]

        quadrant_4 = acf[:int(shape[0]/2)+1, 
                            -int(shape[1]/2):]

        right_half = np.concatenate((quadrant_2, quadrant_1), axis=0)
        left_half = np.concatenate((quadrant_3, quadrant_4), axis=0)

        whole_acf = np.concatenate((left_half, right_half), axis=1)
        

        acfs.append(whole_acf)
    return acfs



def process_acf(record, time_samp, chan_width, 
                num_chans, index, acf_array, dtype):
    """
    Normalize dynamic spectrum and update acf_array with current acf.

    Arguments:
        record -- chunk of dynamic spectrum with axes as (time, freq)
        time_samp -- sampling time of observation in seconds
        chan_width -- channel bandwidth of observation in megahertz
        num_chans -- total number of frequency channels in observation
        index -- index of current loop
        acf_array -- array to hold autocorrelation functions
        dtype -- valid numpy data type

    Returns:
        None 
    """
    record_T = np.transpose(record)

    # Represent masked data with zeros.
    if len(ignore) != 0:
        for value in ignore:
            value = value.split(":")

            if len(value)==1:
                record_T[int(value[0]),:] = 0

            else:

                begin = value[0]
                end = value[1]

                record_T[int(begin):int(end), :] = 0

    record_ravel = record_T.ravel()
    record_nonzero=np.where(record_T.ravel()!=0)

    try:
        # Replace masked data with a sampling of noise consistent
        # with the rest of the observation. This is done to prevent
        # 'ringing' in the autocorrelation function from the masked
        # channels. One can not simply take the autocorrelation of 
        # masked data using the FFT method because the FFT algorithm
        # implicitly assumes uniformly sampled data.

        median = np.median(record_ravel[record_nonzero])
        med_dev = mad(record_ravel[record_nonzero])
        record_zero = np.where(record_ravel==0)
        normal_draw = np.random.normal(loc=median, scale=med_dev,
                                         size=np.shape(record_zero)[1])

        record_ravel[record_zero] = normal_draw
        record_T = np.reshape(record_ravel, (np.shape(record_T)[0], 
                                            np.shape(record_T)[1]))

    except FloatingPointError:
        median=0
        med_dev = 1
        record_zero = np.where(record_ravel==0)
        normal_draw = np.random.normal(loc=median, scale=med_dev, 
                                        size=np.shape(record_zero)[1])

        record_ravel[record_zero] = normal_draw
        record_T = np.reshape(record_ravel, (np.shape(record_T)[0], 
                                            np.shape(record_T)[1]))


    # Correct for the bandpass of the telescope by 
    # normalizing each frequency channel. If this step 
    # is not done, then there is again 'ringing' in the acf
    # due to the frequency dependant baseline changes.

    median = np.median(record_T, axis=1)
    med_dev = mad(record_T, axis=1)

    try:
        bandpass_corr_record = np.transpose((np.transpose(record_T) - median)/med_dev)
        freq_mean = np.mean(bandpass_corr_record, axis=1)
        where = np.where(freq_mean>= (5/np.sqrt(sub)))
        where_num = np.shape(where)[1]
        if where_num<=3:
            if where_num==1:
                size=sub
            else:
                size=(where_num, sub)
            bandpass_corr_record[where,:] = np.random.normal(loc=0, scale=1, size=size)
            record[:, where] = 0



    except FloatingPointError:
        bandpass_corr_record = np.zeros(np.shape(record_T))



    acf_array[index, :, :] = np.ma.array(np.array(auto_corr2d_fft(
                                        bandpass_corr_record, 
                                        np.shape(record_T)[1], dtype)[0]))


    return






def delta(f1, f2, DM):
    """
    Calculate the dispersive time delay in milliseconds.

    Arguments:
        f1 -- frequency of channel 1 in megahertz
        f2 -- frequency of channel 2 in megahertz. Should be greater than f1.
        DM -- dispersion measure in pc/cm^3
    
    Returns:
        Dispersive time delay in milliseconds between f1 and f2.
    
    Exceptions:
        ValueError -- if f1 > f2
    """

    if f1.any()>f2:
        raise ValueError('f1 must not be greater than f2')

    return (4.148808e6*(f1**(-2) - f2**(-2))*DM)


def dedisperse(spectra, DM, ctr_freq, 
                chan_width, time_samp, dtype):
    """
    Dedisperse a dynamic spectrum.

    Arguments:
        spectra -- dynamic spectrum
        DM -- dispersion measure (pc/cm^3)
        ctr_freq -- center frequency of band in megahertz
        chan_width -- channel width in megahertz
        time_samp -- sampling time of observation in seconds
        dtype -- valid numpy data type 

    Returns:
        Dedispersed dynamic spectrum
    """
    num_chans = np.shape(spectra)[0]
    num_time_bins = np.shape(spectra)[1]


    freq = np.flip(np.linspace(ctr_freq - (chan_width*num_chans/2) + chan_width/2, 
                    ctr_freq + (chan_width*num_chans/2) - chan_width/2, num_chans))


    time_lags = delta(freq, freq[0], DM)/(1000*time_samp)

    max_lag = int(np.round(time_lags[-1]))


    dedispersed_data = np.ones((num_chans, num_time_bins - max_lag), dtype=dtype)

    for index, time_lag in enumerate(time_lags):

        bin_shift = int(np.round(time_lag))

        dedispersed_data[index, :] = spectra[index,  bin_shift: 
                                                bin_shift + 
                                                np.shape(dedispersed_data)[1]]

    return dedispersed_data



def fits_parse(infile):
    """
    Parse PSRFITS file using astropy

    Arguments:
        infile -- filename of fits file

    Returns:
        data and metadata of observation
    """
    
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


def print_candidates(total_candidates_sorted, burst_metadata):
    """
    Print candidates to terminal window and save to disk.

    Arguments:
        total_candidates_sorted -- list of candidates sorted by SNR
        burst_metadata -- metadata of observation

    Returns:
        None
    """

    sub_int, time_samp, ctr_freq, chan_width, num_chans, dm, ra_string, dec_string, tstart = burst_metadata
    print("********************************************************************************************************************************************")
    print("********************************************************************************************************************************************")
    print("*******************************************************                             ********************************************************")
    print("*******************************************************  Detected burst properties  ********************************************************")
    print("*******************************************************                             ********************************************************")
    print("********************************************************************************************************************************************")
    print("********************************************************************************************************************************************\n")
    print("********************************************************************************************************************************************")
    print("Burst location (s)             MJD                  Max ACF SNR              Time Window Max SNR (ms)         Frequency Window Max SNR (MHz)")
    print("********************************************************************************************************************************************")

    for index, candidate in enumerate(total_candidates_sorted):

        sigma_max = candidate.sigma.index(max(candidate.sigma))
        acf_window_where = candidate.acf_window[sigma_max]
        t_window = np.round(acf_window_where[0]*time_samp*1000, decimals=2)
        f_window = np.round(acf_window_where[1]*chan_width, decimals=2)

        burst_mjd = tstart + (candidate.location/86400)

        print(str(candidate.location) 
            + print_string_sep(len(str(candidate.location)), 31)
            + "{:0.5f}".format(burst_mjd)
            + print_string_sep(len("{:0.5f}".format(burst_mjd)), 21) 
            +"{:0.2f}".format(max(candidate.sigma)) 
            + print_string_sep(len("{:0.2f}".format(max(candidate.sigma))), 25) 
            + str(t_window) + print_string_sep(len(str(t_window)), 33) 
            + str(f_window))

        with open(outfilename + "_detected_bursts.txt", "a") as f:
            if index==0:
                f.write("# Location (s), MJD (topo), Max ACF SNR, DM, Time Window (ms), Frequency Window (MHz)\n")

                f.write(str(candidate.location) 
                    + "," + "{:0.5f}".format(burst_mjd)
                    + "," + "{:0.2f}".format(max(candidate.sigma))
                    + "," + str(dm) + ","
                    + str(t_window) + "," + str(f_window) + "\n")
            else:

                f.write(str(candidate.location) 
                    + "," + "{:0.5f}".format(burst_mjd)
                    + "," + "{:0.2f}".format(max(candidate.sigma)) 
                    + "," + str(dm) + "," 
                    + str(t_window) + "," + str(f_window) + "\n")                

        np.save(str(outfilename) + "_" + 
                str(np.around(candidate.location, decimals=2)) 
                + "s_" + "burst", 
                [candidate], 
                allow_pickle=True)
        


    print("********************************************************************************************************************************************")
    print("********************************************************************************************************************************************\n")    


    if len(total_candidates_sorted)!= 0:
        np.save(outfilename + "_bursts", total_candidates_sorted, 
                allow_pickle=True)
    else:
        with open(outfilename + "_detected_bursts.txt", 'w') as f:
            f.write("No bursts found :(")
    return 



def preprocess(data, metadata, bandpass_avg, bandpass_std):
    """
    Find short duration RFI and mask.

    Arguments:
        data -- dynamic spectrum of data
        metadata -- metadata of observation
        bandpass_avg -- bandpass average from PRESTO rfifind .stats file
        bandpass_std -- bandpass standard deviation from PRESTO .stats file

    Returns:
        None
    """
    sub_int, time_samp, ctr_freq, chan_width, num_chans, dm, ra_string, \
                                                dec_string, tstart = metadata

    N = np.shape(data)[0]
    data_mean = np.mean(data, axis=0)

    data_mean[np.where(data_mean==0)] = 1

    data_var = np.zeros(num_chans)

    # if data standard deviation is taken directly 
    # on entire dataset using np.std a huge memory leak occurs. 
    # Therefore use this hack by finding std of each chunk
    # to build total std. If you try to take the standard deviation
    # using the below method on the entire dataset without splitting it up
    # a memory leak again occurs. I have no idea why.


    for index in np.arange(10):

        chunk = data[index*np.shape(data)[0]//10:(index+1)*np.shape(data)[0]//10,:]

        data_var = data_var \
                    + np.sum((chunk - data_mean)**2, axis=0)



    data_std = np.sqrt(data_var/N)




    data_std[np.where(data_std==0)] = 1
    bandpass_avg[np.where(bandpass_avg==0)] = 1


    for index in np.arange(np.shape(data)[0]//(sub)):

        chunk = data[index*sub:(index+1)*sub, :]


        bandpass_mean = chunk.mean(axis=0, dtype=np.float32)

        bandpass_mean = bandpass_mean/bandpass_avg

        bandpass_normed = (bandpass_mean - data_mean/bandpass_avg)\
                            /(data_std/(bandpass_avg*np.sqrt(sub)))


        where_normed = np.where(abs(bandpass_normed) >= 10)


        chunk[:, where_normed] = 0

        data[index*sub:(index+1)*sub, :] = chunk

    return


#
#
#CLASS DEFINITIONS
#
#

class Candidate:

    """Class for candidate bursts."""

    def __init__(self, location, sigma, image, 
                    acf, metadata, gauss_fit, 
                    selected_window, true_burst, 
                    acf_window, freq_center):
        """
        __init__ method for Candidate.

        Arguments:
            location -- burst location either in seconds or subintegrations
            sigma -- signal to noise ratio of acf
            image -- dynamic spectrum of data at location
            acf -- autocorrelation function of data at location
            metadata -- metadata of observation
            gauss_fit -- tuple of best fit parameters and covariance matrix
                         of 2d gaussian fit to burst dynamic spectrum. This 
                         should be set to zero if the interactive plotting 
                         is not run. Otherwise, the interative plotting routine
                         will set this automatically.
            selected_window -- tuple of the form (x_left, x_right, 
                               y_top, y_bottom), which represents the user
                               selected window when running the interactive
                               plotting routine. Again this should be set to 
                               zero and the interactive plotting routine will
                               set this automatically based on the user's
                               choice.
            true_burst -- boolean flag on whether a candidate is a true or false
                          false positive. If the user confirms a burst using the
                          interactive plotter, this will be set to true. Useful
                          for statistics on the true positive rate of the 
                          algorithm.
            acf_window -- tuple which represents the window used to take the mean
                          of the autocorrelation function. Should be of the form
                          (time_width, freq_width).
            freq_center -- frequency center of the burst as determined by the
                           cross correlation analysis.

        Returns:
            None
        """
        self.location = location
        self.sigma = [sigma]
        self.acf_window = [(acf_window[0], acf_window[1])]
        self.image = image 
        self.acf = acf
        self.metadata = metadata
        self.gauss_fit = gauss_fit
        self.selected_window = selected_window
        self.true_burst = true_burst
        self.freq_center = freq_center

        self.cross_corr = False
        self.cc_snr = 0

    def update_acf_windows(self, sigma, time, freq):
        """
        Append a new acf_window and sigma to the respective existing lists.

        Arguments:
            sigma -- signal to noise ratio to append to sigma list
            time -- time width of window used to take the mean of the acf.
            freq -- freq width of window used to take the mean of the acf.

        Returns:
            None
        """

        self.sigma.append(sigma)
        self.acf_window.append((time, freq))

    def cross_corr_2d(self, boxcar, spec):


        zero_padded_1 = np.zeros((3*np.shape(boxcar)[0]//2, 
                                3*np.shape(boxcar)[1]//2))

        zero_padded_1[:np.shape(boxcar)[0], :np.shape(boxcar)[1]] = boxcar
        zero_padded_2 = np.zeros((3*np.shape(spec)[0]//2, 
                                3*np.shape(spec)[1]//2))

        zero_padded_2[:np.shape(spec)[0], :np.shape(spec)[1]] = spec
        shape_padded = np.shape(zero_padded_1)
        shape = np.shape(boxcar)

        fft_1 = np.fft.fft2(zero_padded_1)
        fft_2 = np.fft.fft2(zero_padded_2)

        conj_2 = np.conj(fft_2)
        corr = np.real(np.fft.ifft2(conj_2*fft_1))

        quadrant_1 = corr[:(shape[0]//2)+1, 
                            :(shape[1]//2)+1]

        quadrant_2 = corr[-shape[0]//2:, 
                            :(shape[1]//2) + 1]

        quadrant_3 = corr[-shape[0]//2:, 
                            -shape[1]//2:]

        quadrant_4 = corr[:(shape[0]//2) + 1, 
                            -shape[1]//2:]

        right_half = np.concatenate((quadrant_2, quadrant_1), axis=0)
        left_half = np.concatenate((quadrant_3, quadrant_4), axis=0)

        whole_corr = np.concatenate((left_half, right_half), axis=1)


        return whole_corr

    def take_cross_corr(self):
        self.cross_corr = True

        sigma_max = self.sigma.index(max(self.sigma))
        acf_window_where = self.acf_window[sigma_max]

        time_width = acf_window_where[0]
        freq_width = acf_window_where[1]

        #             burst_metadata = (sub_int, time_samp, ctr_freq, chan_width, 
        #                       num_chans, dm, ra_string, dec_string, tstart)

        num_chans = self.metadata[4]

        boxcar = np.zeros((num_chans, sub))
        fcenter = num_chans//2
        tcenter = sub//2
        boxcar[int(fcenter-freq_width//2): int(fcenter + (freq_width//2) + 1), 
                int(tcenter - time_width//2):int(tcenter + (time_width//2) + 1)]\
                                 = np.ones((int(2*(freq_width//2) + 1), int(2*(time_width//2) + 1)))

        spec = self.image

        median = np.ma.median(spec, axis=1)
        med_dev = mad(spec, axis=1)

        try:
            bandpass_corr_spec = np.transpose((np.transpose(spec) - median)/med_dev)
        except FloatingPointError:
            bandpass_corr_spec = np.zeros(np.shape(spec))

        bandpass_corr_spec = np.ma.getdata(bandpass_corr_spec)


        N = 4*((freq_width//2) + 1)*((time_width//2) + 1)

        cc = self.cross_corr_2d(bandpass_corr_spec, boxcar)/np.sqrt(N)

        center = cc.argmax()
        cc_max = np.amax(cc)

        self.cc_snr = cc_max


        # cross_corr_mean = np.mean(cc)

        # N = (2*freq_width//2)*(2*time_width//2)
        # M = num_chans*sub 

        # stdev = np.sqrt(N/M)


        return center




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
        width = 12
        height = 6.756
        self.fig = plt.figure(figsize=(width, height), dpi=100)
        self.ax1 = self.fig.add_axes([0.1, 0.1, 0.8, 0.65])
        
        self.ax1.set_ylabel("Frequency (MHz)", fontsize=12)

        divider = make_axes_locatable(self.ax1)
        time_array = np.linspace(self.loc*sub_int, (self.loc+1)*sub_int, 
                                                    np.shape(self.data)[1])
        axbottom1 = divider.append_axes("bottom", size=1, pad=0.4,
                                                     sharex=self.ax1)

        mean = np.mean(self.data, axis=0)

        median = np.median(mean)
        med_dev = mad(mean)

        normed_mean = (mean-median)/med_dev

        axbottom1.plot(time_array, normed_mean, 
                        marker="o", mfc="k", ms=1, mec="k")

        axbottom1.set_xlabel("Time (s)", fontsize=12)
        self.extent = [self.loc*sub_int, (self.loc+1)*sub_int, 
                        ctr_freq - bandwidth/2, ctr_freq + bandwidth/2]

        self.ax1.imshow(self.data, aspect='auto', extent=self.extent)

        self.ax1.text(0.5, height - 0.25, "Metadata for observation:", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(0.5, height - 0.50, "Right Ascension (hms): " + ra, 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(0.5, height - 0.75, "Declination (dms): " + dec, 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(0.5, height - 1, "Start Time (MJD): {:.2f}".format(tstart),
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(0.5, height - 1.25, 
                        "Sampling time: {:.2e}".format(time_samp) + " s", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(4.5, height - 0.25, 
                        "Channel Width: {:.2f}".format(chan_width) + " MHz", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(4.5, height - 0.50, 
                        "Center Frequency: {:.2f}".format(ctr_freq) + " MHz", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(4.5, height - 0.75, 
                        "Total Bandwidth: {:.0f}".format(bandwidth) + " MHz", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(4.5, height - 1, 
                        "Dispersion Measure: {:.2f}".format(dm) + " pc cm$^{-3}$", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(4.5, height - 1.25, 
                        "SNR of ACF: {:.2f}".format(max(sigma)), 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(8.5, height - 0.5, 
                        "Time window width: {:.2f}".format(self.acf_window_max[0]\
                                                        *time_samp*1000) + " ms", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

        self.ax1.text(8.5, height - 0.75, 
                        "Frequency window width: {:.2f}".format(self.acf_window_max[1]\
                                                                *chan_width) + " MHz", 
                        fontsize=12, transform=self.fig.dpi_scale_trans)

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


        self.popt, self.pcov = curve_fit(gaussian_2d, (x,y), data_to_fit.ravel(), 
                                        p0=[middle_time, middle_freq, 
                                        max_timewidth/4, max_bandwidth/4, 
                                        -0.5, max_value] , 
                                        bounds=([time_array[0], freq_array[-1], 
                                        0, 0, -1, 0],[time_array[-1], 
                                        freq_array[0], max_timewidth/2, 
                                        max_bandwidth/2, 0, np.inf]))
        
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
        self.ax2.contour(x, y, gaussian_2d((x, y), 
                                *self.popt).reshape(end_freq - start_freq, 
                                                    end_time - start_time))

        self.ax2.text(0.5, height - 0.25, 
                        "Fit parameters:", fontsize=12, 
                        transform=self.fig2.dpi_scale_trans)

        self.ax2.text(0.5, height - 0.5, 
                        "Time location: {:0.3f}".format(loc_x) + " s", 
                        fontsize=12, transform=self.fig2.dpi_scale_trans)

        self.ax2.text(0.5, height - 0.75, 
                        "Frequency location: {:0.3f}".format(loc_y) + " MHz", 
                        fontsize=12, transform=self.fig2.dpi_scale_trans)  

        self.ax2.text(4.5, height - 0.5, 
                        "Time sigma: {:0.3f}".format(sigma_x) + " s", 
                        fontsize=12, transform=self.fig2.dpi_scale_trans)

        self.ax2.text(4.5, height - 0.75, 
                        "Frequency sigma: {:0.3f}".format(sigma_y) + " MHz", 
                        fontsize=12, transform=self.fig2.dpi_scale_trans)

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

    """ 
    This class definition comes from the PRESTO rfifind.py 
    file by Scott Ransom, licensed under the GNU General Public 
    License. This modified version is included here to read 
    in rfifind mask files, and is compliant with GPL.
    """

    def __init__(self, filename):
        self.basename = filename[:filename.find("_rfifind.")+8]
        self.read_stats()

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

    def get_bandpass(self, plot=False):
        """
        get_bandpass():
            This routine returns a 'good' bandpass based on an average
                of the average bandpasses, with the exception of the
                intervals that were recommended for zapping in the mask.
        """
        ints = np.arange(self.nint)
        badints = self.mask_zap_ints
        goodints = set(ints) - set(badints)
        goodints = np.asarray(list(goodints))
        self.goodints = goodints
        if not len(goodints):
            print("WARNING!:  Cannot get bandpass because all intervals zapped.")
            return 0.0
        self.bandpass_avg = self.avg_stats[goodints,:].mean(0)
        self.bandpass_std = self.std_stats[goodints,:].mean(0)
        self.bandpass_pow = self.pow_stats[goodints,:].mean(0)
        if plot:
            plotxy(self.bandpass_avg, self.freqs, labx="Frequency (MHz)")
            plotxy(self.bandpass_avg+self.bandpass_std, self.freqs, color="red")
            plotxy(self.bandpass_avg-self.bandpass_std, self.freqs, color="red")
            closeplot()
        return self.bandpass_avg


types = {'8':np.uint8, '16':np.uint16,'32':np.float32, '64':np.float64}



#
#
# MAIN FUNCTION 
#
#

def main():



    if flag==0: # read in real data

        print("Reading data from {}".format(filename))

        if filename[-4:]=='fits':
            #read in fits data and parse for data
            #data is assumed to have structure of (subchunk, time, ~, freq, ~)

            data, sub_int_orig, ctr_freq, chan_width, \
            time_samp, ra_string, dec_string, tstart, dtype = fits_parse(filename)

            dtype = str(dtype)

            data_shape = np.shape(data)

            num_chans = data_shape[3]
            orig_time_samples = data_shape[0]*data_shape[1]

        else: # read in sigproc filterbank data
            data, sub_int_orig, ctr_freq, chan_width, \
            time_samp, ra_val, dec_val, tstart, dtype = fb.filterbank_parse(filename)#, 0, 1)

            dtype=str(dtype)

            ra_val = str(ra_val)
            dec_val = str(dec_val)

            ra_string = ra_val[:2] + ":" + ra_val[2:4] + \
                        ":" + "{:.2f}".format(float(ra_val[4:]))

            dec_string = dec_val[:2] + ":" + dec_val[2:4] + \
                        ":" + "{:.2f}".format(float(dec_val[4:]))

            num_chans = np.shape(data)[0]
            orig_time_samples = np.shape(data)[1]

    else: # create simulated data

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



    # Use a standardized subintegration size of sub time bins to make 
    # processing of differently structured data sets easier to accomplish. 
    # Thus set sub_int time to sub*time_samp.
    # Later split up all data into sub integratons of sub time bins.

    
    sub_int = sub*time_samp # time per subintegration



    mask_chan = []
    acfs_to_mask = []

    if maskfile:
        # read in maskfile if it has been provided and set channels to ignore

        mask = rfifind(maskfile)
        dtint, mask_chan, mask_int = mask.read_mask()
        mask.get_bandpass()


        bandpass_avg = np.flip(mask.bandpass_avg)
        bandpass_std = np.flip(mask.bandpass_std)


        ptsperint = mask.ptsperint
        subperint = int(np.ceil(ptsperint/sub))

        if interval is None:
            offset = 0
        else:
            # define offset time from beginning of observation 
            # if beginning of specified interval is not beginning
            # of observation
            offset = orig_time_samples*interval[0]*time_samp

        for i in mask_int:
            pts = i*ptsperint
            start = int(np.floor(pts/sub)) - int(np.floor((offset/time_samp)/sub))

            if start>=0:
                rng = list(range(start, start+subperint))
                acfs_to_mask = acfs_to_mask + rng


        acfs_to_mask = set(acfs_to_mask)


        global ignore

        # presto indexes assuming first frequency channel is lowest 
        # this program assumed that first frequency channel is highest, 
        # so need to adjust the indices of the masked channels to reflect that
        mask_chan = [str(num_chans - chan - 1) for chan in mask_chan]

        for val in ignore:
            value = val.split(":")
            if len(value)==1:
                mask_chan.append(str(value[0]))
            else:
                values = list(np.arange(int(value[0]), int(value[1]), int(value[2])))

                for num in values:
                    mask_chan.append(str(num))


        ignore = ignore + mask_chan

        if len(set(mask_chan))/num_chans >= 0.5:
            print("More than half the bandwidth is masked! Aborting...")
            with open("aborted_runs.txt", "a") as f:
                f.write(filename + \
                        " aborted because {:0.0f}".format(100*len(mask_chan)/num_chans) \
                        + " percent of channels are masked\n")



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







    if filename[-4:]=="fits":


        all_data = np.ones((data_shape[3], data_shape[1]*data_shape[0]), 
                            dtype=types[dtype])

        for index, record in enumerate(data):
            all_data[:, index*data_shape[1]:(index+1)*data_shape[1]] \
                                    = np.transpose(record[:,0,:,0])


        if chan_width>0:
            # if data is ordered with ascending frequency
            # flip so that descending order is used.
            all_data = np.flip(all_data, axis=0)

        chan_width = abs(chan_width)

        burst_metadata = (sub_int, time_samp, ctr_freq, chan_width, 
                            num_chans, dm, ra_string, dec_string, tstart)



        if zero_dm_filt:
            data_mean = np.mean(data, axis=0)
            data = data - data_mean                


        print("\nPreprocessing data...")
        if maskfile:
            preprocess(np.transpose(all_data), burst_metadata, 
                                    bandpass_avg, bandpass_std)


        print("\nPreprocessing complete!")
        print("\n Dedispersing data using DM = " + str(dm) + " ...")

        if interval is None:
            dedispersed_data = np.transpose(dedisperse(all_data, dm, 
                                            ctr_freq, chan_width, time_samp,
                                            types[dtype]))
        else:
            dedispersed_data = np.transpose(dedisperse(all_data[:, 
                                int(np.shape(all_data)[1]*interval[0]): 
                                int(np.shape(all_data)[1]*interval[1])],
                                dm, ctr_freq, chan_width, time_samp, 
                                types[dtype]))

        total_time_samples = np.shape(dedispersed_data)[0]
        del all_data
        del data
    else:
        
        if chan_width>0:
            # if data is ordered with ascending frequency
            # flip so that descending order is used.
            all_data = np.flip(data, axis=0)

        chan_width = abs(chan_width)
        
        burst_metadata = (sub_int, time_samp, ctr_freq, chan_width, 
                            num_chans, dm, ra_string, dec_string, tstart)   

        if zero_dm_filt:

            data_mean = np.mean(data, axis=0)
            data = data - data_mean


        print("\nPreprocessing data...")
        if maskfile:
            preprocess(np.transpose(data), burst_metadata, 
                        bandpass_avg, bandpass_std)
        print("\nPreprocessing complete!")


        print("\n Dedispersing data using DM = " + str(dm) + " ...")

        if interval is None:
            dedispersed_data = np.transpose(dedisperse(data, dm, ctr_freq, 
                                chan_width, time_samp, types[dtype]))
        else:
            dedispersed_data = np.transpose(dedisperse(data[:, 
                                int(np.shape(data)[1]*interval[0]): 
                                int(np.shape(data)[1]*interval[1])],
                                dm, ctr_freq, chan_width, time_samp, 
                                types[dtype]))

        total_time_samples = np.shape(dedispersed_data)[0]
        del data

    print("\n Dedispersion complete!")



    print("\n Processing ACFs of each data chunk...")


    if time_to_plot==0:
        # calculate acfs of entire dataset
        acf_array = np.ma.ones((int(total_time_samples/sub), 
                        num_chans + 1, sub + 1), dtype=np.float32)
        for index in tqdm(np.arange(np.shape(dedispersed_data)[0]//sub)):
            record = dedispersed_data[index*sub:(index+1)*sub,:]
            process_acf(record, time_samp, 
                        chan_width, num_chans, 
                        index, acf_array, types[dtype])
            dedispersed_data[index*sub:(index+1)*sub,:]=record
    else:
        # calculate acf of single provided time
        acf_array = np.ma.ones((1, num_chans+1, sub+1), dtype=np.float32)

        index = int(time_to_plot//sub_int)
        record = dedispersed_data[index*sub:(index+1)*sub,:]
        process_acf(record, time_samp, 
                    chan_width, num_chans, 
                    0, acf_array, types[dtype])



    print("\n\n ...processing complete!\n")





    center_freq_lag = int(num_chans/2)
    center_time_lag = int(sub/2)



    acf_array.mask = np.zeros((int(total_time_samples/sub), 
                                num_chans + 1, sub + 1), dtype=np.uint8)
    #acf_array.mask[:, :, center_time_lag] = 1
    #acf_array.mask[:, center_freq_lag, :] = 1
    acf_array.mask[:, center_freq_lag, center_time_lag] = 1
    min_t = 1
    min_f = 1 #3

    t_wins = np.logspace(np.log2(min_t), np.log2(int(sub_int/2/time_samp)), 
                            10, base=2)
    f_wins = np.linspace(min_f, int(num_chans/2), 10)



    locs = set({})
    cand_dict = {}
    acf_shape = np.shape(acf_array[0,:,:])



    np.seterr(all='ignore')

    print("Calculating ACF means...")
    for i, time in enumerate(tqdm(t_wins)):
        for j, freq in enumerate(f_wins):


            means = acf_array[:,int(acf_shape[0]/2 - freq): int(acf_shape[0]/2 + freq + 1), \
            int(acf_shape[1]/2 - time): int(acf_shape[1]/2 + time + 1)].mean(axis=(1, 2))


            #means.mask = np.zeros(np.shape(means))


            N = ((2*time) + 1)*((2*freq) + 1) - 1#((2*time) + 1)#1 #3*((2*time) + 1)
            stdev = 1/np.sqrt(N*num_chans*sub)



            acf_norm = means/stdev

    
            if time_to_plot==0:
                threshold_locs = np.where(acf_norm >= thresh_sigma)[0]
            else:
                threshold_locs = [int(time_to_plot//sub_int)]

            for loc in threshold_locs:
                if (loc not in acfs_to_mask) or (time_to_plot!=0): 
                    if loc not in locs:
                        burst = np.ma.array(np.transpose(dedispersed_data[loc*sub
                                                            :(loc+1)*sub, :]))

                        burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)

                        burst = np.ma.masked_where(np.ma.getdata(burst)==0, burst) # mask zeros

                        for chan in mask_chan: # mask channels provided by mask and ignorechan
                            burst[int(chan), :].mask = np.ones(np.shape(burst)[1], dtype=np.uint8)

                        if time_to_plot==0:
                            candidate = Candidate(loc, 
                                        np.round(abs(acf_norm[loc]), decimals=2),
                                        burst, acf_array[loc], burst_metadata, 
                                        0, 0, False, (time, freq), 0)
                        else:
                            candidate = Candidate(loc, 
                                        np.round(abs(acf_norm[0]), decimals=2),
                                        burst, acf_array[0], burst_metadata, 
                                        0, 0, False, (time, freq), 0)       


                        cand_dict[loc] = candidate
                        locs.add(loc)

                    else:

                        if time_to_plot==0:
                            cand_dict[loc].update_acf_windows(acf_norm[loc], time, freq)
                        else:
                            cand_dict[loc].update_acf_windows(acf_norm[0], time, freq)






    cand_list = [cand_dict[key] for key in cand_dict]

    if time_to_plot!=0:
        # save single acf tp disk and exit
        candidate = cand_list[0]
        candidate.location *= sub_int
        np.save(str(outfilename) + "_" + 
        str(np.around(time_to_plot, decimals=2)) 
        + "s_" + "burst", 
        [candidate], 
        allow_pickle=True)

        sys.exit("Plot saved as {}".format(str(outfilename) + "_" + 
        str(np.around(time_to_plot, decimals=2)) 
        + "s_" + "burst.npy"))




    np.seterr(all='raise')   


    prune_cand_list = []
    for candidate in cand_list:

        # prune candidates
        sigma_max = candidate.sigma.index(max(candidate.sigma))
        acf_window_where = candidate.acf_window[sigma_max]

        max_t = acf_window_where[0]*time_samp
        max_f = acf_window_where[1]*chan_width

        t_windows = [window[0] for window in candidate.acf_window]
        f_windows = [window[1] for window in candidate.acf_window]

        min_f_window = min(f_windows)

        acf = np.ma.getdata(candidate.acf)
        acf[np.shape(acf)[0]//2, np.shape(acf)[1]//2] = 0

        # scale by acf size to avoid underflow issues
        power_acf = (num_chans*sub*acf)**2

        power_tot = np.sum(power_acf)
        power_freq0 = np.sum(power_acf[np.shape(power_acf)[0]//2, :])

        frac = power_freq0/power_tot

        # if frac <= 0.2:
        if (max_t <= prune_value/1000):
            if max_f >= 0.05*chan_width*num_chans:
                prune_cand_list.append(candidate)


    pruned_cand_sorted = sorted(prune_cand_list, 
                                reverse=True, 
                                key= lambda prune_candidate: max(prune_candidate.sigma))



    if cross_corr:

        for cand in tqdm(pruned_cand_sorted):

            center = cand.take_cross_corr()


            # use sub+1 to account for the fact that
            # cross correlated data has a zero lag
            # giving total length in time of sub + 1
            tcenter = (center%(sub+1)) + (sub*cand.location)
            fcenter = (center//(sub+1))

            burst = np.ma.array(np.transpose(dedispersed_data[tcenter - 
                        sub//2: tcenter + sub//2 + 1]))

            burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)

            # mask any zeros in dynamic spectrum
            burst = np.ma.masked_where(np.ma.getdata(burst)==0, burst)

            # explicitly mask channels provided by rfifind mask and
            # ignorechan argument
            for chan in mask_chan:
                burst[int(chan), :].mask = np.ones(np.shape(burst)[1], 
                                                        dtype=np.uint8)

            cand.image = burst
            cand.location = tcenter/sub
            cand.freq_center = (ctr_freq + num_chans*chan_width/2) \
                                - fcenter*chan_width   


        # Remove duplicates of the same burst.
        # Duplicates can occur if burst is split
        # across different chunks of data.
        cands_to_remove = []
        for i in range(len(pruned_cand_sorted)-1):
            loc = pruned_cand_sorted[i].location 

            for j in range(i+1, len(pruned_cand_sorted)):
                if abs(pruned_cand_sorted[j].location - loc) <= 1:
                    # if candidates are within one sub integration of each
                    # other, remove the one with lower snr
                    cands_to_remove.append(j)

        pruned_cand_sorted = [cand for cand in pruned_cand_sorted if 
                                pruned_cand_sorted.index(cand) 
                                not in cands_to_remove]

        #pruned_cand_sorted = [cand for cand in pruned_cand_sorted if cand.cc_snr>=thresh_sigma]



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
            else:
                candidate.gauss_fit = (ip.popt, ip.pcov)
                candidate.selected_window = (ip.xdata1, ip.xdata2, 
                                            ip.ydata1, ip.ydata2)




        for index, candidate in enumerate(pruned_cand_sorted):

            if candidate.gauss_fit != 0:

                candidate.true_burst=True

                if interval is None:
                    offset = 0
                else:
                    offset = orig_time_samples*interval[0]*time_samp
           
                if dm!=0:
                    popt = candidate.gauss_fit[0]
                    popt[0] = popt[0] + offset
                    center = int(popt[0]/time_samp)

                    burst = np.ma.array(np.transpose(dedispersed_data[center - 
                            int(0.02/time_samp): center + int(0.02/time_samp),:]))

                    burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)

                    burst = np.ma.masked_where(np.ma.getdata(burst)==0, burst)

                    for chan in mask_chan:
                        burst[int(chan), :].mask = np.ones(np.shape(burst)[1], 
                                                                dtype=np.uint8)

                    candidate.image = burst
                    candidate.freq_center = popt[1]

                else:
                    popt = candidate.gauss_fit[0]

                    popt[0] = popt[0] + offset
                    center = int(popt[0]/time_samp)

                    burst = np.ma.array(np.transpose(all_data[center - 
                            int(0.02/time_samp): center + int(0.02/time_samp),:]))
                    burst.mask = np.zeros(np.shape(burst), dtype=np.uint8)

                    burst = np.ma.masked_where(np.ma.getdata(burst)==0, burst)
                    for chan in mask_chan:
                        burst[int(chan), :].mask = np.ones(np.shape(burst)[1], 
                                                                dtype=np.uint8)

                    candidate.image = burst              



    for cand in pruned_cand_sorted:
        if interval is None:
            offset = 0
        else:
            offset = orig_time_samples*interval[0]*time_samp


        cand.location = np.round(cand.location*sub_int + offset , decimals=2)

    if flag==1:
        with open(outfilename + "_simulated_bursts.txt", "a") as f:
            for index, loc in enumerate(sim_data.timeloc):
                if index==0:
                    f.write("# Location simulated burst (s), Peak Scale\n")
                    f.write("{:0.2f}".format(loc) 
                            + "," 
                            + "{:0.2f}".format(sim_data.peak_snrs[index]) + "\n")

                else:
                    f.write("{:0.2f}".format(loc) 
                        + "," 
                        + "{:0.2f}".format(sim_data.peak_snrs[index]) + "\n")

        with open(outfilename + "_bursts.txt", "a") as f:
            for index, candidate in enumerate(pruned_cand_sorted):



                has_window = 0
                if (min_t, min_f) in candidate.acf_window:
                    has_window=1

                if index==0:
                    f.write("# Location detected burst (s), ACF SNR, has min window\n")
                    f.write("{:0.2f}".format(candidate.location) 
                            + "," 
                            + "{:0.2f}".format(max(candidate.sigma)) 
                            + "," 
                            + "{}".format(has_window) + "\n")
                else:
                    f.write("{:0.2f}".format(candidate.location) 
                        + "," 
                        + "{:0.2f}".format(max(candidate.sigma)) 
                        + "," 
                        + "{}".format(has_window) + "\n")                

    print_candidates(pruned_cand_sorted, burst_metadata)



    return 



if __name__=='__main__':
    parser = ap.ArgumentParser()

    parser.add_argument("--infile", help=("PSRFITS or SIGPROC filterbank filename."
                                          " Filename must end in .fits or .fil."  
                                          " No filename should be passed if sim_data"
                                          " is set to 1."))

    parser.add_argument("--sigma", help=("Search for peaks in the acf" 
                                        " above this threshold. Default = 10."), 
                                        type=int, default=10)

    parser.add_argument("--d", help=("Dispersion measure value to use"
                                     " for dedispersion. Default = 0."
                                     " If zero, program assumes that the data"
                                     " is already dedispersed."), 
                                    type=float, default=0)

    parser.add_argument("--plot", help=("1: Use interactive plotting feature"
                                        " to view bursts. Default=0."
                                        " Set to one for interactive plotting."), 
                                        type=int, default=0)


    parser.add_argument("--ignorechan", help=("Frequency channels to ignore"
                                            " when processing."), 
                                            type=str, nargs="+", default=None)

    parser.add_argument("--sim_data", help=("1: Run ACF analysis on simulated data."
                                        " Default=0: do not run simulated data."), 
                                        type=int, default=0)

    parser.add_argument("--mask", help="PRESTO rfifind .mask file")

    parser.add_argument("--prune", help=("All candidates with max SNR occuring" 
                                        " at time window above this value will" 
                                        " be pruned. Default=10 ms"), 
                                        type=float, default=10)

    parser.add_argument("--sub_int", help=("Length of desired sub-integration in"
                                        " time bins. The data is split into" 
                                        " chunks of this size in time before the"
                                        " ACF is calculated. Default=2048"), 
                                        type=int, default=2048)


    parser.add_argument("--interval", help=("Start and end time of data to" 
                                            " process as fraction of observation"
                                            " length. Default is to process"
                                            " entire observation.)"), 
                                            nargs=2, type=float, default=None)

    parser.add_argument("--zero_dm_filt", help=("1: Use a zero DM filter to remove" 
                                            " broadband RFI in the dispersed data." 
                                            " Default=0: do not use zero DM filter."), 
                                            type=int, default=0)

    parser.add_argument("--cross_corr", help=("1: Find the pulse arrival time and"
                                              " frequency center by cross correlating"
                                              " boxcar, with time and frequency" 
                                                " widths equal to the best fit"
                                                " ACF window width, with the dynamic"
                                                " spectrum."), type=int, default=0)

    parser.add_argument("--time", help=("Single time to calculate the ACF of."
                                        " The ACF is taken of the chunk of data"
                                        " which the time corresponds to."
                                        " Default is zero, which runs the normal"
                                        " program, which takes ACFs of entire"
                                        " dataset."),
                                        type=float, default=0)

    parser.add_argument("outfile", help="String to append to output files.", 
                                                                    default='')


    args = parser.parse_args()


    filename = args.infile
    thresh_sigma = args.sigma
    dm = args.d
    outfilename = args.outfile
    plot = args.plot
    ignore = args.ignorechan
    flag = args.sim_data
    maskfile = args.mask
    prune_value = args.prune
    sub = args.sub_int
    interval = args.interval
    zero_dm_filt = args.zero_dm_filt
    cross_corr = args.cross_corr
    time_to_plot = args.time




    if flag==1:
        filename="dummy"


    if ignore is None:
        ignore = []






    main()






