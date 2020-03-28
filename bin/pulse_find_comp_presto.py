import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

class Candidate:
    def __init__(self, location, sigma, image, acf, fluence, metadata, snr, gauss_fit, selected_window):
        self.location = location
        self.sigma = sigma
        self.image = image
        self.acf = acf
        self.fluence = fluence
        self.metadata = metadata
        self.snr = snr
        self.gauss_fit = gauss_fit
        self.selected_window = selected_window




# load in pickled candidates

cands = np.load("./121102_test_bursts.npy", allow_pickle=True)


# load in candidates from singlepulse text file

singlepulse_cands =  np.loadtxt("./121102.singlepulse")

print(singlepulse_cands)