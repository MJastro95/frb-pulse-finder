#!/usr/bin/env python3


import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap 
import sys
import matplotlib as mpl

from scipy.stats import cauchy
from scipy.optimize import curve_fit
from scipy.stats import norm

from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import minimize

from scipy.stats import median_absolute_deviation as mad

from pulse_find import rfifind

np.seterr(under="warn")

class Candidate:
    def __init__(self, location, sigma, image, acf, fluence, metadata, snr):
        self.location = location
        self.sigma = sigma
        self.image = image
        self.acf = acf
        self.fluence = fluence
        self.metadata = metadata
        self.snr = snr


def gaussian_2d(input_tuple, mean_x, mean_y, sigma_x, sigma_y, rho, scale):
    x = input_tuple[0]
    y = input_tuple[1]
    
    x_term = ((x-mean_x)**2)/(sigma_x**2)
    y_term = ((y-mean_y)**2)/(sigma_y**2)
    
    cross_term = (2*rho*(x-mean_x)*(y-mean_y))/(sigma_x*sigma_y)
    value = (scale)*np.exp((-1/(2*(1-(rho**2))))*(x_term + y_term - cross_term))
    return value.ravel()

def lorentzian_2d(input_tuple, x0, y0, a, b, c, d, scale):
    x = input_tuple[0] - x0
    y = input_tuple[1] - y0
    det = a*d - b*c
    
    thing = (1 + ((d*(x**2) - (b+c)*(x*y) + a*(y**2))/det))**(-1.5)
    
    return (scale*thing).ravel()

def squared_error(input_tuple, data, mesh):

    x,y = mesh

    x0, y0, a, b, c, d, scale = input_tuple

    model_pred = lorentzian_2d((x, y), x0, y0, a, b, c, d, scale)
    return np.sum((data - model_pred)**2)

def cauchy_fit(x, a, loc, scale):
    return a*cauchy.pdf(x, loc=loc, scale=scale)

def normal_fit(x, a, mean, sigma):
    return a*norm.pdf(x, loc=mean, scale=sigma)


def equations_to_solve(input_tuple, sigmax, sigmay, rho):
    alpha, a, b = input_tuple

    equation_1 = (np.cos(alpha)**2)/(a**2) + (np.sin(alpha)**2)/(b**2) - (1/((1 - (rho**2))*(sigmax**2)))
    equation_2 = np.cos(alpha)*np.sin(alpha)*((1/(a**2)) - (1/(b**2))) + (rho/(sigmax*sigmay))
    equation_3 = (np.sin(alpha)/a)**2 + (np.cos(alpha)/b)**2 - (1/((sigmay**2)*(1 - (rho**2))))
    return (equation_1, equation_2, equation_3)

def equations_to_solve_lorentz(input_tuple, a, b, c, d):
    alpha, a_prime, b_prime = input_tuple

    equation_1 = (np.cos(alpha)**2)/(a_prime**2) + (np.sin(alpha)**2)/(b_prime**2) - d
    equation_2 = np.cos(alpha)*np.sin(alpha)*((1/(a_prime**2)) - (1/(b_prime**2))) + (b+c)
    equation_3 = (np.sin(alpha)/a_prime)**2 + (np.cos(alpha)/b_prime)**2 - a

    return (equation_1, equation_2, equation_3)



def linear(x, slope):
    y_array = []

    for num in x:
        y = int(np.around(slope*num)) + 256
        y_array.append(y)

    return np.array(y_array)




def main():

    parser = ap.ArgumentParser()


    parser.add_argument("infile", help="File containing the acf to plot")
    # parser.add_argument("--time", help="Sampling time of telescope in seconds", type=float)
    # parser.add_argument("--bw", help="Bandwidth of observation in MHz", type=float)
    # parser.add_argument("--chan", help="Number of frequency channels", type=int)
    parser.add_argument("--bounds", help="Length of time axis to plot in time bins", type=int, default=2048)
    parser.add_argument("--fit", help="Type of model to fit to autocorrelation function. Either gaussian or lorentzian. Default value is empty string, which means don't fit.", default='')
    parser.add_argument("--p0", help="Initial guess for parameters to fit autocorrelation function.", type=float, nargs="+", default=None)
    parser.add_argument("--mask", help="PRESTO rfifind mask")

    args = parser.parse_args()

    infile = args.infile
    fit = args.fit
    bounds = args.bounds
    p0 = args.p0
    maskfile = args.mask

    metadata, acf, loc, burst, snr, gauss_fit, selected_window, acf_window = np.load(infile, allow_pickle=True)

    sub_int = metadata[0]
    time_samp = metadata[1]
    ctr_freq = metadata[2]
    chan_width = metadata[3]
    num_chan = metadata[4]
    dm = metadata[5]
    ra = metadata[6]
    dec = metadata[7]
    tstart = metadata[8]



    if maskfile:
        # read in maskfile if it has been provided and set channels to ignore

        mask = rfifind(maskfile)
        dtint, mask_chan, mask_int = mask.read_mask()

        burst = np.ma.array(burst)

        np_mask = np.zeros(np.shape(burst))

        for chan in mask_chan:
            np_mask[chan,:] = np.ones(np.shape(burst)[1])

        burst.mask = np_mask



    where_max = snr.index(max(snr))
    acf_window_max = acf_window[where_max]


    bandwidth = chan_width*num_chan

    acf_shape = np.shape(acf)

    axis1_center = int(acf_shape[0]/2)
    axis2_center = int(acf_shape[1]/2)

    acf[axis1_center, axis2_center] = (acf[axis1_center, axis2_center + 1] + acf[axis1_center, axis2_center-1] + acf[axis1_center-1, axis2_center] + acf[axis1_center+1, axis2_center])/4


    acf.mask = np.zeros(acf_shape, dtype=np.uint8)
    acf.mask[int(acf_shape[0]/2), :] = np.ones(acf_shape[1], dtype=np.uint8)
    acf.mask[int(acf_shape[0]/2)-1, :] = np.ones(acf_shape[1], dtype=np.uint8)

    acf_to_plot = acf[:, axis2_center - int(bounds/2): axis2_center + int(bounds/2)]

    extent = [-bounds*time_samp*1000/2, bounds*time_samp*1000/2, -bandwidth/2, bandwidth/2]


    width = 15
    height = 8.4375

    fig = plt.figure(figsize=(width, height), dpi=100)

    ax1 = fig.add_axes([0.55, 0.075, 0.4, 0.5])#fig.add_subplot(111)
    ax2 = fig.add_axes([0.05, 0.075, 0.4, 0.5])
    ax3 = fig.add_axes([0.55, 0.7, 0.4, 0.25])


    t_widths = []
    f_widths = []

    for window in acf_window:
        t_widths.append(1000*time_samp*window[0])
        f_widths.append(chan_width*window[1])

    ax3.scatter(t_widths, f_widths, c=snr)
    ax3.set_xlabel("Time window width (ms)", fontsize=12)
    ax3.set_ylabel("Frequency window width (MHz)", fontsize=10)
    ax3.set_xlim((time_samp*1000, np.shape(acf)[1]*time_samp*1000/2 + time_samp*1000))
    ax3.set_ylim((chan_width, num_chan*chan_width/2 + chan_width))
    ax3.set_xscale('log')


    norm = mpl.colors.Normalize(vmin=min(snr), vmax=max(snr))

    cbar = fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm), ax=ax3)
    cbar.set_label("Signal-to-noise Ratio", fontsize=12)


    ax1.imshow(acf_to_plot, extent=extent, aspect="auto", cmap="Greys", interpolation='none')
    ax1.set_ylabel("Frequency lag (MHz)", fontsize=12)
    ax1.set_title("Autocorrelation function", fontsize=12)

    divider = make_axes_locatable(ax1)

    axbottom1 = divider.append_axes("bottom", size=1.2, pad=0.4, sharex=ax1)
    axright1 = divider.append_axes("right", size=1.2, pad=0.6, sharey=ax1)

    time_array = np.linspace(-bounds*time_samp*1000/2, bounds*time_samp*1000/2, bounds)
    freq_array = np.linspace(bandwidth/2, -bandwidth/2, num_chan)
    mean = np.mean(acf_to_plot, axis=0)

    axbottom1.plot(time_array, mean, marker="o", mfc="k", ms=1, mec="k")

    mean = np.mean(acf_to_plot, axis=1)
    axright1.plot(mean, freq_array, marker="o", mfc='k', ms=1, mec='k')

    axbottom1.set_xlabel("Time lag (ms)", fontsize=12)


    if selected_window==0:
        extent= [loc, loc + 2048*time_samp, ctr_freq - bandwidth/2, ctr_freq + bandwidth/2]
        time_array = np.linspace(loc, loc + 2048*time_samp, 2048)
        ax2.text(4, height - 0.75, "Burst location: {:.2f}".format(loc) + " s", fontsize=12, transform=fig.dpi_scale_trans)

    else:
        popt = gauss_fit[0]
        time_center = int(popt[0]/time_samp)
        data_shape = np.shape(burst)

        extent = [-time_samp*data_shape[1]*1000/2, time_samp*data_shape[1]*1000/2, ctr_freq - bandwidth/2, ctr_freq + bandwidth/2]
        time_array = np.linspace(-time_samp*data_shape[1]*1000/2, time_samp*data_shape[1]*1000/2, data_shape[1])

        ax2.text(4, height - 0.75, "Burst location (Gaussian fit): {:.2f}".format(popt[0]) + " s", fontsize=12, transform=fig.dpi_scale_trans)

    ax2.imshow(burst, aspect='auto', extent=extent)
    divider = make_axes_locatable(ax2)

    axbottom2 = divider.append_axes("bottom", size=1.2, pad=0.3, sharex=ax2)

    median = np.median(np.mean(burst, axis=0))
    med_dev = mad(np.mean(burst, axis=0))

    axbottom2.plot(time_array,(np.mean(burst,axis=0)-median)/med_dev, mfc="k", ms=1, mec="k", color='grey')


    if selected_window!=0:
        xdata1, xdata2, ydata1, ydata2 = selected_window 

        top_freq = ctr_freq + (bandwidth + chan_width)/2

        start_freq = int((top_freq - ydata1)//chan_width)
        end_freq = int((top_freq - ydata2)//chan_width)


        narrow_band_burst = burst[start_freq:end_freq, :]
        narrow_band_mean = np.mean(narrow_band_burst, axis=0)
        narrow_band_median = np.median(narrow_band_mean)
        narrow_band_med_dev = mad(narrow_band_mean)

        axbottom2.plot(time_array, (narrow_band_mean - narrow_band_median)/narrow_band_med_dev, mfc='k', ms=1, mec='k', color='k')

        axbottom2.set_xlabel("Time (ms)", fontsize=12)
    else:

        axbottom2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Frequency (MHz)", fontsize=12)

    #scientific_notation = "{:e}".format(12300000)
    #def printC(answer):
    #print("\nYour Celsius value is {:0.2f}ºC.\n".format(answer))
    ax2.text(0.5, height - 0.5, "Metadata for observation:", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 0.75, "Right Ascension (hms): " + ra, fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 1, "Declination (dms): " + dec, fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 1.25, "Start Time (MJD): {:.2f}".format(tstart),fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 1.5, "Sampling time: {:.2e}".format(time_samp) + " s", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 1.75, "Channel Width: {:.2f}".format(chan_width) + " MHz", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 2, "Center Frequency: {:.2f}".format(ctr_freq) + " MHz", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 2.25, "Total Bandwidth: {:.2f}".format(bandwidth) + " MHz", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(0.5, height - 2.5, "Dispersion Measure: {:.2f}".format(dm) + " pc cm$^{-3}$", fontsize=12, transform=fig.dpi_scale_trans)

    ax2.text(4, height - 0.5, "Fit parameters:", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(4, height - 1, "SNR of ACF: {:.2f}".format(max(snr)), fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(4, height - 1.25, "Time window width: {:0.2f}".format(1000*time_samp*acf_window_max[0]) + " ms", fontsize=12, transform=fig.dpi_scale_trans)
    ax2.text(4, height - 1.5, "Frequency window width: {:0.2f}".format(chan_width*acf_window_max[1]) + " MHz", fontsize=12, transform=fig.dpi_scale_trans)
    fig.canvas.draw()

    if fit=='':
        ax2.text(4, height - 1.75, "Fit type: " + "N/A", fontsize=12, transform=fig.dpi_scale_trans)            
        ax2.text(4, height - 2, "Time width: N/A" , fontsize=12, transform=fig.dpi_scale_trans)
        ax2.text(4, height - 2.25, "Frequency Width: N/A", fontsize=12, transform=fig.dpi_scale_trans)            
        ax2.text(4, height - 2.5, "Drift Rate: N/A", fontsize=12, transform=fig.dpi_scale_trans)
        plt.show()


    if fit != '':

        if fit=='gaussian' or fit=='Gaussian':

            time_array = np.linspace(-bounds*time_samp*1000/2, bounds*time_samp*1000/2, bounds)
            freq_array = np.linspace(-bandwidth/2, bandwidth/2, num_chan)

            x,y = np.meshgrid(time_array, freq_array)

            weights = np.ones(np.shape(acf_to_plot))
            weights[int(np.shape(weights)[0]/2),:] = np.inf*np.ones(np.shape(weights)[1])
            weights[int(np.shape(weights)[0]/2) - 1, :] = np.inf*np.ones(np.shape(weights)[1])


            maximum = np.amax(np.mean(np.ma.getdata(acf_to_plot), axis=0))

            if p0 is None:
                popt, pcov = curve_fit(gaussian_2d, (x,y), np.flip(acf_to_plot, axis=0).ravel(), p0=[0, 0, 2, 100, -0.5, maximum], bounds=([-np.inf, -np.inf, 0, 0, -1, 0],[np.inf, np.inf, np.inf, np.inf, 1, np.inf]))
            else:
                popt, pcov = curve_fit(gaussian_2d, (x,y), np.flip(acf_to_plot, axis=0).ravel(), p0= p0 + [maximum], bounds=([-np.inf, -np.inf, 0, 0, -1, 0],[np.inf, np.inf, np.inf, np.inf, 1, np.inf]))
            sigma_x = popt[2]
            sigma_y = popt[3]
            rho = popt[4]

            print(popt)

            result= least_squares(equations_to_solve, (np.pi/2, 1, 1), bounds=((0, -np.inf, -np.inf), (2*np.pi, np.inf, np.inf)), args=(sigma_x, sigma_y, rho))

            alpha = result.x[0] - np.pi/2
            print(alpha)

            slope = -np.round(1/(np.tan(alpha)))
            print("The slope is: " + str(slope) + " MHz/ms")



            ax1.contour(x, y, gaussian_2d((x, y), *popt).reshape(int(num_chan), int(bounds)))

            axbottom1.plot(time_array, np.mean(gaussian_2d((x, y), *popt).reshape(int(num_chan), int(bounds)), axis=0), color='r')
            axbottom1.margins(x=0)


        
            mean = np.mean(acf_to_plot, axis=1)


            freq_array = np.linspace(-bandwidth/2, bandwidth/2, num_chan)


            popt_freq, pcov_Freq = curve_fit(normal_fit, freq_array, mean, p0=(1, 1, 100))


            axright1.plot(np.mean(gaussian_2d((x, y), *popt).reshape(int(num_chan), int(bounds)), axis=1), freq_array, color='r')
            axright1.margins(y=0)


            ax2.text(4, height - 1.75, "Fit type: " + fit, fontsize=12, transform=fig.dpi_scale_trans)            
            ax2.text(4, height - 2, "Time width: {:0.2f}".format(2*np.sqrt(2*np.log(2))*sigma_x) + " ms" , fontsize=12, transform=fig.dpi_scale_trans)
            ax2.text(4, height - 2.25, "Frequency Width: {:0.2f}".format(2*np.sqrt(2*np.log(2))*sigma_y) + " MHz", fontsize=12, transform=fig.dpi_scale_trans)            
            ax2.text(4, height - 2.5, "Drift Rate: {:0.2f}".format(slope) + " MHz/ms", fontsize=12, transform=fig.dpi_scale_trans)

        if fit=="lorentz" or fit=="Lorentz":
            x,y = np.meshgrid(time_array, freq_array)

            res = minimize(squared_error, (0, 0, 0.01, 0, 0, 0.04, 100), args=(np.flip(acf_to_plot, axis=0).ravel(), np.meshgrid(time_array, freq_array)), \
             method='slsqp', constraints={'type':'ineq', 'fun': lambda input_tuple: input_tuple[2]*input_tuple[5] - input_tuple[3]*input_tuple[4]}, \
             bounds=((-0.01, 0.01), (-0.01, 0.01), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)), options={'disp':True})


            # popt, pcov = curve_fit(lorentzian_2d, (x, y), np.flip(acf_to_plot, axis=0).ravel(), p0=[0, 0, 1, 0, 0, 1, 1])
            ax1.contour(x, y, lorentzian_2d((x, y), *res.x).reshape(int(num_chan), int(bounds)))

            print(res.x)

            a = res.x[2]
            b = res.x[3]
            c = res.x[4]
            d = res.x[5]


            result = least_squares(equations_to_solve_lorentz, (np.pi/4, 1, 1), bounds= ((0, -np.inf, -np.inf),(2*np.pi, np.inf, np.inf)), args=(a, b, c, d))

            alpha = result.x[0]# - (np.pi/2)
            print(alpha)
            print("The slope is: " + str(-np.round(1/(np.tan(alpha)), decimals=2)) + " MHz/ms")


            # popt_time, pcov_time = curve_fit(cauchy_fit, time_array, mean, p0=(1, 0, 1))

            #axbottom.plot(time_array, mean, marker="o", mfc="k", ms=1, mec="k")
            axbottom.plot(time_array, np.mean(lorentzian_2d((x, y), *res.x).reshape(int(num_chan), int(bounds)), axis=0), color='r')
            axbottom.margins(x=0)

            axbottom.set_xlabel("Time (ms)", fontsize=12)

            #axright = divider.append_axes("right", size=1.2, pad=0.6, sharey=ax1)

        
            mean = np.mean(acf_to_plot, axis=1)


            array = np.linspace(-bandwidth/2, bandwidth/2, num_chan)


            popt_freq, pcov_freq = curve_fit(cauchy_fit, array, mean)

            #axright.plot(mean, array, marker="o", mfc='k', ms=1, mec='k')
            axright.plot(np.mean(lorentzian_2d((x, y), *res.x).reshape(int(num_chan), int(bounds)), axis=1), array, color='r')
            axright.margins(y=0)

    plt.show()


    return

if __name__=='__main__':

    main()








