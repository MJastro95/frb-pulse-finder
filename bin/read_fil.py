#!/usr/bin/env python3
import numpy as np 
import struct 
import os
import argparse as ap

header_dict = {
	b'HEADER_START':None,
	b'telescope_id':'int',
	b'machine_id':'int',
	b'data_type':'int',
	# b'rawdatafile':'char[]',
	# b'source_name':'char[]',
	b'barycentric':'int',
	b'pulsarcentric':'int',
	b'az_start':'double',
	b'za_start':'double',
	b'src_raj':'double',
	b'src_dej':'double',
	b'tstart':'double',
	b'tsamp':'double',
	b'nbits':'int',
	b'nsamples':'int',
	b'fch1':'double',
	b'foff':'double',
	b'FREQUENCY_START':None,
	b'fchannel':'double',
	b'FREQUENCY_END':None,
	b'nchans':'int',
	b'nifs':'int',
	b'refdm':'double',
	b'period':'double',
	b'HEADER_END':None
}



parser = ap.ArgumentParser()

parser.add_argument("--infile", help="File to read header data from. Must be filterbank format.")	

args = parser.parse_args()

infile = args.infile

with open(infile, 'rb') as file:
	fileContent = file.read(512)

# print(fileContent)

locations = []
tuple1 = []
tuple2 = []
not_found = []
for key in header_dict:

	loc = fileContent.find(key)
	if key==b'src_raj' or key==b'src_dej' or key==b'az_start' or key==b'za_start':
		loc = fileContent.find(key, loc + len(key))
		val = struct.unpack('d', fileContent[loc + len(key):loc + len(key) + 8])[0]
		tup2 = (key, loc, val)
		tuple2.append(tup2)
	elif loc!=-1:
		data_type = header_dict[key]
		if data_type==None:
			tup1 = (key, loc)
			tuple1.append(tup1)
		elif header_dict[key]=='int':
			val = struct.unpack('i', fileContent[loc + len(key):loc + len(key) + 4])[0]
			tup2 = (key, loc, val)
			tuple2.append(tup2)
		else:
			val = struct.unpack('d', fileContent[loc + len(key):loc + len(key) + 8])[0]
			#print(key, len(key))
			#print(fileContent[loc:loc + len(key) + 8])

			tup2 = (key, loc, val)
			tuple2.append(tup2)


	else:
		not_found.append(key)



print("\nHeader Data for " + infile)
for data in tuple2:

	key, loc, val = data

	key_as_string = key.decode('utf-8')

	print(key_as_string + " = " + str(val))


print("\nI didn't find these header keys:")

for key in not_found:
	print(key.decode('utf-8'))



