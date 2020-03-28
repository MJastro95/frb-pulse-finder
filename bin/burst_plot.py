#!/usr/bin/env python3


import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap 

from scipy.stats import median_absolute_deviation as mad


def main():
    parser = ap.ArgumentParser()

    parser.add_argument("--infile", help="File containing the burst to plot")
    # parser.add_argument("--time", help="Sampling time of telescope in seconds", type=float)
    # parser.add_argument("--bw", help="Bandwidth of observation in MHz", type=float)
    # parser.add_argument("--chan", help="Number of frequency channels", type=float)
    # parser.add_argument("--cfreq", help="Center frequency of observation in MHz", type=float)

    args = parser.parse_args()


    infile = args.infile


    metadata, loc, burst = np.load(infile, allow_pickle=True)

    time_samp = metadata[1]
    chan_width = metadata[3]
    num_chan = metadata[4]
    ctr_freq = metadata[2]

    bandwidth=chan_width*num_chan

    extent= [loc, loc + 2048*time_samp, ctr_freq - bandwidth/2 + (bandwidth/num_chan)/2, ctr_freq + bandwidth/2 - (bandwidth/num_chan)/2]
    time_array = np.linspace(loc, loc + 2048*time_samp, 2048)



    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(burst, aspect='auto', extent=extent)
    divider = make_axes_locatable(ax)

    axbottom = divider.append_axes("bottom", size=1.2, pad=0.3, sharex=ax)

    median = np.median(np.mean(burst, axis=0))
    med_dev = mad(np.mean(burst, axis=0))

    axbottom.plot(time_array,(np.mean(burst,axis=0)-median)/med_dev, marker="o", mfc="k", ms=1, mec="k")
    axbottom.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (MHz)", fontsize=12)
    plt.show()



if __name__=='__main__':
    main()
