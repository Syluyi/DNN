#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:19:53 2017

@author: Danny
"""
# possible improvements: setting parameters from command line,
# remove the need for some hard coding. More options to stop/skip some
# preprocessing steps. Only fBANKS and MFCCs enabled now.
# N.B. system paths are hard coded
from glob import glob
from process_data import proc_data

base_path = "/ssdata/ubuntu/data/"
dest_path ="/bigdata/ubuntu/data/"
# regex for parsing of the label files
pattern = '"N[0-9]+_SEG"'
# regex for checking the file extensions
f_ex = 'fn[0-9]+.' 

# preprocessing locations, CGN based
preproc_path = base_path + "/preprocessing/"

# set data path for audio and label files
# filter files with glob library pattern
glob_file_pattern = "fn*"
dest_file_name = "fb_fnALL_spectro"
# glob_file_pattern = "fn00???[2,8]"
# dest_file_name = "fb_fn00xxx2o8"
# glob_file_pattern = "fn00??2?"
# dest_file_name = "fb_fn00xx2x"
# glob_file_pattern = "fn001001"
# dest_file_name = "fb_fn001001_spectro"

glob_wav_path = preproc_path + "/audio/" + glob_file_pattern + ".wav"
glob_annot_path = preproc_path + "/annot/"  + glob_file_pattern + ".awd.gz"
conv_table = preproc_path + "feature_conv_table.txt"
print("CGN audio path: " + glob_wav_path)
print("CGN label path: " + glob_annot_path)
print("CGN label print file: " + conv_table)
print("Audio files: \n" + format(glob(glob_wav_path),""))
print("Label files: \n" + format(glob(glob_annot_path),""))
print("Total files: " + format(len(glob(glob_wav_path)),""))
print("Feature conversion table: "+ format(glob(conv_table),""))

# indicate whether you use the CGN awd transcripts (1) or transcripts created
# with kaldi (0)
CGN=1

# some parameters for mfcc creation
params=[]
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired mel filter banks
nfilters = 24
# windowsize and shift in time
t_window=.025
t_shift=.01
# option to get raw filterbank energies (true) or mfccs (false)
filt = True
# option to include delta and double delta features (true)
use_deltas=False

# set the complete destination filename to save the features and labels
dest_name = dest_path + "/fb_prepared/" + dest_file_name 
if (use_deltas):
    dest_name = dest_name + "_delta75"
else:
    dest_name = dest_name + "_base25"
dest_name = dest_name + ".h5"
print("Destination_file: " + dest_name)

# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(filt)
params.append(dest_name)
params.append(use_deltas)

proc_data(pattern,f_ex,params,glob_annot_path,glob_wav_path,conv_table,CGN)
