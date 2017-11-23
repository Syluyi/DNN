#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:05:37 2017

@author: danny
"""
import numpy as np
import math
import pickle
import os
import errno
# contains function to handle the h5 data file for the DNNs. Split_dataset
# creates indices for train test and val set, for the minibatch iterator to load data: use 
# if the data is to big for working memory. Load_dataset actually loads all data in working memory
# and splits into train test and val set. use if data fits working memory



def Split_dataset(f_nodes,l_nodes,splice_size, 
    index_file='fnodesIndexes.py',
    werkindex_file='werkIndexes.py'    
    ):
    # prepare dataset. This is meant for datasets to big for working memory, data 
    # is split into train test and validation based on indexes. The indexes are 
    # used to retrieve data in minibatches. load_dataset is faster but only useable if the dataset 
    # fits in working memory
    index=[]
    offset=0
    try:        
        f=open(index_file,'rb')
        print('using existing indexfile: ' + index_file)
        x=True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    index.append(y)
            except:
                x=False
        f.close()
    except:
    # index triple, first number is the index of each frame. Because different wav files are stored 
    # in different leaf nodes of the h5 file, we also keep track of the node number and the index of 
    # the frame internal to the node.
        f=open(index_file, 'wb')
        print('creating indexfile: ' + index_file)
        for x in range (0,len(f_nodes)):
            print('creating index for audio file: ', x )
            temp=[]
            for y in range (splice_size,len(f_nodes[x])-splice_size):
            # 999 was used to indicate out of vocab values. These are removed
            # from the training data, however they are still valid for splicing
            # with a valid training frame
                if l_nodes[x][y][1]!=b'999':
                    temp.append((y+offset,x,y))
            for i in temp:
                index.append(i)
            offset=offset+len(f_nodes[x])
            pickle.dump(temp,f)
        f.close()

    # create and save shuffled index if not existing in werkindex_file
    werkindex = []       
    try:        
        f=open(werkindex_file,'rb')
        print('using existing werkindex file: ' + werkindex_file)
        x=True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    werkindex.append(y)
            except:
                x=False
        f.close()                
    except:
        # create folders if needed    
        if not os.path.exists(os.path.dirname(werkindex_file)):         
          try:
            os.makedirs(os.path.dirname(werkindex_file))             
          except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
              raise
                                
        f=open(werkindex_file, 'wb')
        print('creating shuffled werkindexfile: ' + werkindex_file)
        np.random.shuffle(index)
        pickle.dump(index,f)
        werkindex = index                
        f.close()              
    return split_index(werkindex)

def restore_index(index_file='werkIndexes.py'):
    werkindex = []
    f=open(index_file,'rb')
    print('opening existing werkindex file: ' + index_file)
    temp = pickle.load(f)
    try:
        for y in temp:
            werkindex.append(y)
    except (EOFError):
        print(EOFError)
        print(len(werkindex))        
    f.close()
    return split_index(werkindex)

def split_index(werkindex, train_part = 0.8, valid_part = 0.2):
    # split an index array in train, validation and test part
    # training defaults to 80%
    # validation defaults to 20%, located in train area (train_part > valid_part)
    # test defaults to every thing not used for training

    data_size=len(werkindex)
    train_size = int(math.floor(data_size*train_part))
    val_size= int(math.floor(data_size*valid_part))
    Train_index = werkindex[0:train_size]
    Val_index = werkindex[train_size-val_size:train_size]
    Test_index = werkindex[train_size:]
    return (Train_index, Val_index, Test_index)
