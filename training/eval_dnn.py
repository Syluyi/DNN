# -*- coding: utf-8 -*-
"""
Created on Sat Jul  22 08:56:04 2017

@author: Nikki
"""
import os
import tensorflow as tf
import numpy as np
import tables as tb
from dnn_data import Split_dataset
learn = tf.contrib.learn
from time import gmtime, strftime, sleep

tf.logging.set_verbosity(tf.logging.ERROR)
DATA = "/ssdata/ubuntu/data/processed/filterbanks_complete.h5"
print("opening filterbanks file: " + DATA)
file = tb.open_file(DATA, "r+")

l_nodes = file.root.labels._f_list_nodes()
f_nodes = file.root.features._f_list_nodes()

splice_size=5
batch_size = 512
max_batch = 120

def iterate_minibatches(index,batchsize,splice_size, shuffle=True):  
    batchsize = min(len(index), batchsize)
    batchseq = 0
    if shuffle:      
        print("Shuffling index...")
        np.random.shuffle(index)
    for start_idx in range(0, len(index) - batchsize + 1, batchsize):
        batchseq = batchseq + 1
        if batchseq >= max_batch:
          print("Max mini batches reached: " + format(max_batch,""))
          break                 
        if shuffle:
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ " excerpt shuffling...") 
            excerpt = index[start_idx:start_idx + batchsize]
        else:
            print("excerpt without shuffling...")
            excerpt = [x for x in range (start_idx, start_idx + batchsize)]
        inputs=[]
        targets=[]
        for ex in excerpt:
            # retrieve the frame indicated by index and splice it with surrounding frames
            inputs.append([f_nodes[ex[1]][ex[2]+x] for x in range (-splice_size,splice_size+1)])
            targets.append(l_nodes[ex[1]][ex[2]][0])
            
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ " reshaping and converting inputs and targets")
        shape= np.shape(inputs)
        inputs = np.float32(np.reshape(inputs,(shape[0], shape[1] * shape[2])))
        targets = np.int64(targets)
        yield inputs, targets

def main():  
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ' creating train, val and test sets, if needed create indexfile')
    [Train_index, Val_index,Test_index]=Split_dataset(f_nodes,l_nodes,splice_size)  

    test_input = np.float32(np.reshape(f_nodes[0][0:11],(1,825)))
    print("test_input", test_input.shape, test_input)
    feature_columns = learn.infer_real_valued_columns_from_input(test_input)
    classifier = learn.DNNClassifier( 
            feature_columns=feature_columns
            , hidden_units=[1024, 1024, 1024], n_classes=38
            , optimizer = tf.train.GradientDescentOptimizer(1E-4)
            , model_dir="/ssdata/ubuntu/data/model/fb_complete"
            , config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))
    
    for batch in iterate_minibatches(Val_index, batch_size, splice_size, shuffle=True):
        inputs, targets = batch
        accuracy_score = classifier.evaluate(inputs, targets)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score)) 
        
    file.close()
    

if __name__ == '__main__':
    main()