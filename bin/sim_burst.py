import numpy as np 
import matplotlib.pyplot as plt 
import datetime
from matplotlib import interactive
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse as ap 
from astropy.io import fits
import scipy.signal as sig
from scipy.stats import median_absolute_deviation as mad
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.gridspec as gridspec
# from pulse_find import auto_corr2d_fft
# from pulse_find import gaussian_2d
# from pulse_find import prune_candidates

def gaussian_2d(input_tuple, mean_x, mean_y, sigma_x, sigma_y, rho, scale):
    x = input_tuple[0]
    y = input_tuple[1]
    
    x_term = ((x-mean_x)**2)/(sigma_x**2)
    y_term = ((y-mean_y)**2)/(sigma_y**2)
    
    cross_term = (2*rho*(x-mean_x)*(y-mean_y))/(sigma_x*sigma_y)
    value = (scale)*np.exp((-1/(2*(1-(rho**2))))*(x_term + y_term - cross_term))
    return value.ravel()


class simulatedData:
    def __init__(self, nchan, nbins, chan_width, tsamp, hifreq):
        self.nchan = nchan
        self.nbins = nbins
        self.chan_width = chan_width
        self.tsamp = tsamp
        self.hifreq = hifreq
        self.create_data()



    def create_data(self):
        self.data = np.ones((self.nchan, self.nbins), dtype=np.float32)
        self.add_noise()

    def add_noise(self):
        self.data  = np.random.normal(10, 1, self.nchan*self.nbins).reshape(self.nchan, self.nbins)



    def add_rfi(self):


        self.chans = np.random.randint(0, self.nchan, 2)

        for index, chan in enumerate(self.chans):
            
            self.timestart = np.random.randint(0, self.nbins, self.nbins//500)

            self.lengths = np.random.randint(5, 30, self.nbins//500)

            self.brightness = np.random.uniform(5, 15, self.nbins//500)

            for index, time in enumerate(self.timestart):
                self.data[chan, time: time + self.lengths[index]] = self.data[chan, time: time + self.lengths[index]] + self.brightness[index]

        self.masked_chan = np.random.randint(0, self.nchan, 5)
        self.data[self.masked_chan] = 0



    def add_bursts(self, num):

        self.timeloc = np.random.randint(512, self.nbins, num)
        self.peak_snrs = []

        self.freqloc = np.random.randint(5, self.nchan-5, num)

        self.slopes = np.random.randint(100, 300, num) # MHz/ms

        t = np.linspace(0, self.nbins, self.nbins)
        f = np.linspace(0, self.nchan, self.nchan)

        t, f = np.meshgrid(t,f)

        print("\nSimulating bursts...")
        for index, loc in enumerate(tqdm(self.timeloc)):
            freq_width = np.random.uniform(self.nchan//10, self.nchan//3)
            time_width = np.random.uniform(0.000125/self.tsamp, 0.00075/self.tsamp)
            scale = np.random.uniform(0.5, 10)
            self.peak_snrs.append(scale)


            self.data = self.data + gaussian_2d((t,f), loc, self.freqloc[index], time_width, freq_width, 0, scale).reshape(self.nchan,self.nbins)

            slope = self.slopes[index]*self.tsamp*1000/self.chan_width

            delta_t = np.random.uniform(0.5*time_width, time_width)
            delta_f = int(slope*delta_t)
            sub_burst_loc = (delta_t + loc, self.freqloc[index] + delta_f)

            sb_twidth = np.random.uniform(0.000125/self.tsamp, 0.00075/self.tsamp)
            sb_fwidth = np.random.uniform(self.nchan//10, self.nchan//3)


            self.data = self.data + gaussian_2d((t,f), sub_burst_loc[0], sub_burst_loc[1], sb_twidth, sb_fwidth, 0, scale).reshape(self.nchan,self.nbins)



    def plot_data(self):
        plt.imshow(self.data, aspect='auto')
        plt.show()







# def reshape(array, x, y, z):
#     reshaped_array = np.zeros((x,y,z))
#     for val in np.arange(x):
#         reshaped_array[val, : , :] = array[:, val*z: (val + 1)*z]
#     return reshaped_array

# class Candidate:
#     def __init__(self, location, sigma, true_dect, image):
#         self.location = location
#         self.sigma = sigma
#         self.true_dect = true_dect
#         self.image = image

# class Burst:
#     def __init__(self, location, snr, drift_rate, detection):
#         self.location = location
#         self.snr = snr
#         self.drift_rate = drift_rate
#         self.detection = detection


# def burst_gen_gauss(freq_width, time_width, drift_rate, sub_burst_width, peak_snr, noise, rfi=False, pos=(12, 430), size=(64, 3712000), multi=False, num_burst=100):
#     total_signal = noise #np.random.normal(loc=10, scale=3, size=size)
    
#     chan_start = pos[0]
#     time_start = pos[1]
    
#     rand_width = sub_burst_width
#     rand_time_width = time_width
    
#     x = np.linspace(0, size[1] - 1, size[1])
#     y = np.linspace(0, size[0] - 1 , size[0])


#     x,y = np.meshgrid(x, y)
    
#     #total_signal = total_signal #+ 0.2*(np.sin(y/30) + np.cos(y/20) + np.sin(y/10) + 1)
    
#     mean = np.mean(total_signal)
#     std = np.std(total_signal)
    
#     scale = peak_snr*std
    
#     sigmas_x = []
#     sigmas_y = []
#     starts_x = []
#     starts_y = []
    
#     if multi==False:
#         for i in np.arange(10):
#             slope = int(np.round(i*drift_rate*rand_time_width))
#             burst = gaussian_2d((x, y), time_start + (i*rand_time_width), chan_start + slope, rand_time_width, rand_width, 0, scale)
            
#             sigmas_x.append(rand_time_width)
#             sigmas_y.append(rand_width)
#             starts_x.append(time_start + (i*rand_time_width))
#             starts_y.append(chan_start + slope)
            
            
#             rand_width = int(np.round(np.random.uniform(0.5*sub_burst_width, sub_burst_width)))
#             rand_time_width = int(np.round(np.random.uniform(0.5*time_width, time_width)))

#             total_signal = total_signal + burst.reshape(size)
        
#         #start_x, end_x, start_y, end_y
#         sigmas = (2*sigmas_x[0], 2*sigmas_x[-1], 2*sigmas_y[0], 2*sigmas_y[-1])
        
#         bounds = (starts_x[0] - sigmas[0], starts_x[-1] + sigmas[1], starts_y[0] - sigmas[2], starts_y[-1] + sigmas[3])
        
#         signal = np.mean(total_signal[bounds[2]:bounds[3], bounds[0]:bounds[1]])
        
#         area = (bounds[3] - bounds[2])*(bounds[1] - bounds[0])
        
#         snr = abs(signal - mean)/(std/np.sqrt(area))
        
        
#     else:
#         bursts = []
#         for i in np.arange(num_burst):
#             burst = np.zeros(size).ravel()

#             chan_start = pos[0]

#             time_start = pos[1] + 10000*(i+1)
#             freq_start = int(np.random.uniform(0, size[0]))
#             drift_rate = np.random.uniform(1,5)
#             scale = np.random.uniform(1, 15)
    
#             # start = datetime.datetime.now()
#             for j in np.arange(5):
                
#                 #slope = int(np.round(j*drift_rate*rand_time_width))
                
#                 burst = burst + gaussian_2d((x, y), time_start, chan_start, 0.5*rand_time_width, rand_width, 0, scale)
#                 # plt.imshow(burst.reshape(size)[:,10000:11000])
#                 # plt.gca().set_aspect(10)
#                 # plt.show()
#                 rand_width = int(np.round(np.random.uniform(0.5*sub_burst_width, sub_burst_width)))
#                 rand_time_width = int(np.round(np.random.uniform(time_width, 2*time_width)))

#                 time_start = time_start + rand_time_width
#                 chan_start = chan_start + drift_rate*rand_time_width

#             # end = datetime.datetime.now()
#             # print(end-start)
#             contour = np.where(burst.reshape(size) >= np.max(burst)/2)
#             # print(contour)
#             total_signal = total_signal + burst.reshape(size)

#             # plt.imshow(total_signal[:,10000:11000])
#             # plt.gca().set_aspect(10)
#             # plt.show()

#             # contour = np.where(total_signal.ravel()[contour[0]:contour[-1]] >= mean + 2*std)

#             signal = np.mean(total_signal[contour[0][0]:contour[0][-1], contour[1][0]:contour[1][-1]])

#             area = (contour[0][-1] - contour[0][0])*(contour[1][-1] - contour[1][0])
#             # print(area)
#             # print(signal, mean, std/np.sqrt(area))

#             # print(abs(signal - mean)/(std/np.sqrt(area)))    
#             bursts.append(Burst(np.floor(time_start/500), abs(signal - mean)/(std/np.sqrt(area)), drift_rate, 0))



#             #print(i)
        
#     if rfi==True:
#         chan = np.random.randint(low=0, high=63, size=10)
#         time = np.random.randint(low=0, high=100000, size=10)
#         length = np.random.randint(low=1, high=10, size=10)

#         time_2 = np.random.randint(low=0, high=100000, size=10)
#         length_2 = np.random.randint(low=20, high=100, size=10)
#         chan_2 = np.random.randint(low=0, high=63, size=10)

#         brightness = np.random.uniform(low=25, high=50, size=10)
#         brightness_2 = np.random.uniform(low=25, high=50, size=10)
#         for i in np.arange(10):
#             total_signal[chan[i]: chan[i] + length[i], time[i]] = brightness[i] + total_signal[chan[i]: chan[i] + length[i], time[i]]
#             total_signal[chan_2[i], time_2[i] + length_2[i]] = brightness_2[i] + total_signal[chan_2[i], time_2[i] + length_2[i]]
#         # zap = np.random.randint(low=0, high=63, size=5)
#         # total_signal[zap,:] = 0
#         #total_signal[:20,:] = 0
#         total_signal[50:55,:] = 0
#         bright = np.random.randint(low=0, high=63, size=5)
#         total_signal[bright, :] = 10
        
#     return (total_signal, bursts)


# def main():
#     parser = ap.ArgumentParser()

#     parser.add_argument("--num", help="Number of simulated bursts", type=int)
   
#     args = parser.parse_args()

#     num = args.num


#     bursts_2 = []
#     candidates_2 = []


#     noise = np.random.normal(loc=10, scale=3, size=(64, 200000))



#     for val in tqdm(np.arange(int(num/5))):


#         signal, bursts = burst_gen_gauss(20, 5, 10, 5, 10, noise, size=(64, 200000), multi=True, rfi=True, num_burst=5)



#         # plt.figure()
#         # plt.imshow(signal[:, 10000:11000])
#         # plt.gca().set_aspect(10)
#         # plt.show()
#         # plt.figure()
#         # plt.imshow(signal[:, 20000:21000])
#         # plt.gca().set_aspect(10)
#         # plt.show()
#         # plt.figure()
#         # plt.imshow(signal[:, 30000:31000])
#         # plt.gca().set_aspect(10)
#         # plt.show()
#         # plt.figure()
#         # plt.imshow(signal[:, 40000:41000])
#         # plt.gca().set_aspect(10)
#         # plt.show()
#         # plt.figure()
#         # plt.imshow(signal[:, 50000:51000])
#         # plt.gca().set_aspect(10)
#         # plt.show()


#         signal = reshape(signal, 400, 64, 500)


#         means = []
#         for index, record in enumerate(signal):
#             # if index==104:
#             #     plt.figure()
#             #     plt.imshow(record)
#             #     plt.show()
#             record_ravel = record.ravel()

#             median = np.median(record)
#             med_dev = mad(record)

#             record_zero = np.where(record_ravel== 0)

#             normal_draw = np.random.normal(loc=median, scale=med_dev, size=np.shape(record_zero)[0])

#             record_ravel[record_zero] = normal_draw

#             record = np.reshape(record_ravel, (np.shape(record)[0], np.shape(record)[1]))

#             acf = auto_corr2d_fft(record, np.shape(record)[1])[0]

#             # if index==104:
#             #     plt.figure()
#             #     plt.imshow(acf)
#             #     plt.show()
#             mean = np.mean(acf[:, 230:270])



#             means.append(mean)

#         means = np.array(means)

#         median = np.median(means)
#         med_dev = mad(means)

#         acf_norm = (means - median)/med_dev #(means - total_mean)/total_std

#         # plt.figure()
#         # plt.plot(acf_norm)
#         # plt.show()


#         cand_list = []

#         threshold_locs = np.where(abs(acf_norm) >= 10)[0]
#         # print(threshold_locs)

#         for loc in threshold_locs:

#             cand_list.append(Candidate(loc, abs(acf_norm[loc]), 1, signal[loc]))


#         prune_cand_list = prune_candidates(cand_list, acf_norm, 10000, 1, 1)


#         for index, burst in enumerate(bursts):

#             boolean = any(int(candidate.location) == int(burst.location) for candidate in prune_cand_list)

#             if boolean:
#                 bursts[index] = Burst(burst.location, burst.snr, burst.drift_rate, 1)

#         for index, candidate in enumerate(prune_cand_list):
#             #print(candidate.location)
#             boolean = any(int(burst.location) == int(candidate.location) for burst in bursts)

#             if boolean==False:
#                 prune_cand_list[index] = Candidate(candidate.location, candidate.sigma, 0, candidate.image)


#         # bursts_2.append(bursts)
#         # candidates_2.append(prune_cand_list)

#         with open("sim_burst_stats_newprune.txt", "a") as f:
#             #true detection rates-> given theres a burst, what's the probability its detected

#             for burst in bursts:
#                 f.write(str(burst.location)+ "," + str(burst.snr) + "," + str(burst.drift_rate) + "," + str(burst.detection) + "\n")


#         with open("detection_rates_newprune.txt", "a") as g:
#             #false detection rates -> given there's a detection, what's the probability theres actually a burst

#             for candidate in prune_cand_list:
#                 g.write(str(candidate.location) + "," + str(candidate.sigma) + "," + str(candidate.true_dect) + "\n")

#     return



# if __name__=="__main__":
#     main()

#     # x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])

#     # print(x)
#     # print(reshape(x, 2, 4, 2))











