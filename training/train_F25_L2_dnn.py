# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:56:04 2017

@author: Nikki
"""
import tensorflow as tf
import numpy as np
import tables as tb
from dnn_data import Split_dataset
learn = tf.contrib.learn
from time import gmtime, strftime
tf.logging.set_verbosity(tf.logging.ERROR)

BASE_PATH = "/ssdata/ubuntu/data/"
FB_PATH = BASE_PATH + "fb_prepared/"
FB_FILE = FB_PATH + "fb_fnALL_base25.h5"
MODEL_PATH = BASE_PATH + "model/model_F25_L2_fnALL/"
IDX_FILE = FB_PATH + "fb_fnALL.idx"
WRK_IDX_NAME = "werk.idx" 
print("Using filterbanks file: " + FB_FILE)

# Iterating batches
batch_size = 16384

# DNN Classifier settings
DC_STEPS = 20
DC_SPLICE_SIZE = 5
DC_MINI_BATCH_SIZE = 128

file = tb.open_file(FB_FILE, "r+")

f_nodes = file.root.features._f_list_nodes()
l_nodes = file.root.labels._f_list_nodes()

context = 2 * DC_SPLICE_SIZE + 1
feature_size = len(f_nodes[0][0]) * context

def iterate_minibatches(index,batchsize,splice_size, shuffle=True):  
    batchsize = min(len(index), batchsize)
    if shuffle:      
        print("Shuffling index...")
        np.random.shuffle(index)
    for start_idx in range(0, len(index) - batchsize + 1, batchsize):       
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
            targets.append(l_nodes[ex[1]][ex[2]][1])
            
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ " reshaping and converting inputs and targets")
        shape= np.shape(inputs)
        inputs = np.float32(np.reshape(inputs,(shape[0], shape[1] * shape[2])))
        targets = np.int32(targets)
        yield inputs, targets

def train(run_nr):
    run_iteration = "run_"+ format(run_nr+1,"") + "/"
    print("run iteration: " + run_iteration)      
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ' creating train, val and test sets, if needed create indexfile')
    werkfile = MODEL_PATH + run_iteration + WRK_IDX_NAME
    [Train_index, Val_index,Test_index] = Split_dataset(f_nodes, 
        l_nodes,DC_SPLICE_SIZE, IDX_FILE, werkfile)  

    test_input = np.float32(np.reshape(f_nodes[0][0:context],(1,feature_size)))    
    feature_columns = learn.infer_real_valued_columns_from_input(test_input)
    classifier = learn.DNNClassifier( 
            feature_columns=feature_columns
            , hidden_units=[1024, 1024, 1024], n_classes=2
            , optimizer = tf.train.AdamOptimizer(1E-4)
#            , optimizer = tf.train.GradientDescentOptimizer(2E-3)
            , model_dir= MODEL_PATH + run_iteration)
            #, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))
    classifier = learn.SKCompat(classifier)
   
    test_inputs = []
    test_targets = []
    for batch in iterate_minibatches(Test_index, batch_size, DC_SPLICE_SIZE, shuffle=True):
        test_inputs, test_targets = batch
        break
    
    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
            }   
    
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_inputs,
        test_targets,
        every_n_steps=10,
        metrics=validation_metrics 
        )
    
    batchnr = 1      
    for batch in iterate_minibatches(Train_index,batch_size,DC_SPLICE_SIZE ,shuffle=True):
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+ " training minibatch: " + format(batchnr, ""))
        batchnr = batchnr + 1
        inputs, targets = batch                        
        classifier.fit(inputs, targets, steps = DC_STEPS, 
            batch_size=DC_MINI_BATCH_SIZE, monitors=[validation_monitor])
    
    for batch in iterate_minibatches(Val_index, batch_size, DC_SPLICE_SIZE, shuffle=True):
        inputs, targets = batch
        # accuracy_score = classifier.evaluate(inputs, targets)["accuracy"]
        accuracy_score = classifier.score(inputs, targets)["accuracy"]
        print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))
    
    for batch in iterate_minibatches(Test_index, batch_size, DC_SPLICE_SIZE, shuffle=True):
        inputs, targets = batch
        # accuracy_score = classifier.evaluate(inputs, targets)["accuracy"]
        accuracy_score = classifier.score(inputs, targets)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

def main ():
  train(4)
  file.close()
    
if __name__ == '__main__':
    main()