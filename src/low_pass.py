import numpy as np
import matplotlib.pyplot as plt

class LowPassFilter:
    def __init__(self, power_spec, t_cut, f_cut):

        self.power_spec = power_spec
        self.t_cut = t_cut
        self.f_cut = f_cut

    def reshape_power_spec(self):
        shape = np.shape(self.power_spec)

        quadrant_1 = self.power_spec[:int(shape[0]/2)+1, 
                        :int(shape[1]/2) + 1]

        quadrant_2 = self.power_spec[-int(shape[0]/2):, 
                            :int(shape[1]/2) + 1]

        quadrant_3 = self.power_spec[-int(shape[0]/2):, 
                         -int(shape[1]/2):]

        quadrant_4 = self.power_spec[:int(shape[0]/2)+1, 
                            -int(shape[1]/2):]

        right_half = np.concatenate((quadrant_2, quadrant_1), axis=0)
        left_half = np.concatenate((quadrant_3, quadrant_4), axis=0)

        self.reshaped_power_spec = np.concatenate((left_half, 
                                                    right_half), axis=1)

    def gaussian_filt(self):

        shape = np.shape(self.reshaped_power_spec)

        x = np.arange(shape[1])#np.arange(2*self.t_cut)
        y = np.arange(shape[0])


        x,y = np.meshgrid(x, y)

        return np.exp(-(y - (shape[0]/2))**2/(self.f_cut**2))\
                #*np.exp(-(x - (shape[1]/2))**2/(self.t_cut**2))

    def boxcar_filt(self):

        shape = np.shape(self.reshaped_power_spec)
        filt = np.ones(shape)

        # filt[:(shape[0]//2) - self.f_cut, 
        #     (shape[1]//2) - self.t_cut:(shape[1]//2) + self.t_cut] = 0
        # filt[(shape[0]//2) + self.f_cut:, 
        #     (shape[1]//2) - self.t_cut:(shape[1]//2) + self.t_cut] = 0

        filt[:, (shape[1]//2) - self.t_cut: (shape[1]//2) + self.t_cut] = 0 

        return filt

        

    def apply_filter(self, filt='gauss'):

        if filt=='gauss':
            filt = self.gaussian_filt()
        elif filt=='box':
            filt = self.boxcar_filt()

        shape = np.shape(self.reshaped_power_spec)
        # new_filt = np.ones(shape)

        # new_filt[:, (shape[1]//2) - self.t_cut: (shape[1]//2) + self.t_cut] = filt

        self.filt_pow_spec = filt*self.reshaped_power_spec

        quadrant_1 = self.filt_pow_spec[-(int(shape[0]/2)+1):, 
                                    -(int(shape[1]/2) + 1):]
        quadrant_2 = self.filt_pow_spec[:int(shape[0]/2), 
                                    -(int(shape[1]/2) + 1):]

        quadrant_3 = self.filt_pow_spec[:int(shape[0]/2), 
                                    :int(shape[1]/2)]

        quadrant_4 = self.filt_pow_spec[-(int(shape[0]/2) + 1):, :int(shape[1]/2)]

        left_half = np.concatenate((quadrant_1, quadrant_2), axis=0)
        right_half = np.concatenate((quadrant_4, quadrant_3), axis=0)

        filt_pow_spec_reshaped = np.concatenate((left_half, right_half), axis=1)


        return filt_pow_spec_reshaped


# power_spec = np.ones((64,2048))

# lpg = LowPassFilter(power_spec, 10, 10)
# lpg.reshape_power_spec()

# filt = lpg.boxcar_filt()
# plt.imshow(filt, aspect='auto')
# plt.show()

