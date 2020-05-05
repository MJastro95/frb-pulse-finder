import numpy as np 
import struct
import matplotlib.pyplot as plt
import os

types = {'8':np.uint8, '16':np.uint16,'32':np.float32, '64':np.float64}

def filterbank_parse(infile):#, loop_index, split):
    with open(infile, 'rb') as file:
    	fileContent = file.read(2048)

    file_size = os.path.getsize(infile)/1e6


    nifs_loc = fileContent.find(b'nifs')
    nchans_loc = fileContent.find(b'nchans')
    nbits_loc = fileContent.find(b'nbits')
    header_end_loc = fileContent.find(b'HEADER_END')
    #nsamp_loc = fileContent.find(b'nsamples')
    tsamp_loc = fileContent.find(b'tsamp')
    fch1_loc = fileContent.find(b'fch1')
    foff_loc = fileContent.find(b'foff')
    ra_loc = fileContent.find(b'src_raj')
    dec_loc = fileContent.find(b'src_dec')
    tstart_loc = fileContent.find(b'tstart')


    nifs_val = struct.unpack('i', fileContent[nifs_loc + 4:nifs_loc + 8])[0]
    nchans_val = struct.unpack('i', fileContent[nchans_loc + 6: nchans_loc + 10])[0]
    nbits_val = struct.unpack('i', fileContent[nbits_loc + 5: nbits_loc + 9])[0]
    #nsamp_val = struct.unpack('i', fileContent[nsamp_loc + 8: nsamp_loc + 12])[0]
    tsamp_val = struct.unpack('d', fileContent[tsamp_loc+5: tsamp_loc+13])[0]
    fch1_val = struct.unpack('d', fileContent[fch1_loc+4:fch1_loc+12])[0]
    foff_val = struct.unpack('d', fileContent[foff_loc+4:foff_loc+12])[0]
    ra_val = struct.unpack('d', fileContent[ra_loc+7:ra_loc+15])[0]
    dec_val = struct.unpack('d', fileContent[dec_loc+7:dec_loc+15])[0]
    tstart_val = struct.unpack('d', fileContent[tstart_loc+6:tstart_loc+14])[0]

    ctr_freq = (fch1_val - foff_val/2) + (foff_val*nchans_val/2)

    header_length= len(fileContent[:header_end_loc+10])

    data_size_per_spectra = nchans_val*nbits_val/8e6 #in megabytes

    #size_per_chunk = file_size#/split #in megabytes

    num_spectra = int((file_size - (header_length/1e6))/data_size_per_spectra) #number of spectra (frequency slices at given time) per chunk



    #total_block_size = int(num_spectra*nchans_val*nbits_val/8)


    with open(infile, "rb") as file:
    	file.seek(header_length)#file.seek(total_block_size*loop_index + data_length)
    	data = file.read()#data = file.read(total_block_size)


    proc_data = np.fromstring(data, dtype=types[str(nbits_val)])

    proc_data = np.reshape(proc_data, (num_spectra, nchans_val))
    proc_data = np.transpose(proc_data)

    sub_int = num_spectra*tsamp_val


    return proc_data, sub_int, ctr_freq, foff_val, tsamp_val, ra_val, dec_val, tstart_val, nbits_val

