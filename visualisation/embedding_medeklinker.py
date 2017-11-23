# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tables as tb
import sprite
from dnn_data import restore_index
import tensorflow.contrib.learn as learn

USE_SPECTRO = False

VERSION = "fb_fnALL_spectro_base25"

WORK_DIR = "/ubuntu/data/"
MODEL_DIR = WORK_DIR + "/model/model_F25_L2_fnALL/run_2/"

MODEL_NAME = "model.ckpt"
MODEL_VERSION = "15760"
MODEL_FILE = MODEL_DIR + MODEL_NAME + "-" + MODEL_VERSION

EMBEDDING_DIR_NAME = "/embedding_medeklinker/"
EMBEDDING_DIR_SPECTRO_NAME = "/embedding_medeklinker_spectro/"

if USE_SPECTRO:
    EMBEDDING_DIR = MODEL_DIR + EMBEDDING_DIR_SPECTRO_NAME
else:
    EMBEDDING_DIR = MODEL_DIR + EMBEDDING_DIR_NAME
    
EMBEDDING_FILE = EMBEDDING_DIR + MODEL_NAME
EMBEDDING_LABEL_FILE = EMBEDDING_DIR + 'meta.tsv'
SPRITE_FILE = EMBEDDING_DIR + 'sprite.png'

INDEX_FILE = WORK_DIR + "/fb_prepared/fb_fnALL.idx"
WORKINDEX_FILE = MODEL_DIR + "werk.idx"  

DATA_DIR = WORK_DIR + "/fb_prepared/"
DATA_FILE = DATA_DIR + VERSION + ".h5"
print("opening file")
data_file = tb.open_file(DATA_FILE, "r+")

print("reading l_nodes")
l_nodes = data_file.root.labels._f_list_nodes()
print("reading f_nodes")
f_nodes = data_file.root.features._f_list_nodes()

print("reading s_nodes")
s_nodes = data_file.root.spectra._f_list_nodes()

FEATURE_FILE = WORK_DIR + "preprocessing/" + "feature_conv_table.txt"

splice_size=5
context= 2 * splice_size + 1
batch_size = 1024
feature_size = len(f_nodes[0][0]) * context
print("feature size = ", feature_size)

KLINKER= b'0'
MEDEKLINKER = b'1'

def create_metafile(test_labels, embedding_file):
    print("creating metafile")
    with open(embedding_file ,'w') as f:
        f.write("index\tphone\tvowel\tmanner\tplace\tvoice\tfrback\theight\tround\tduration_diphthongue\tphonetext\tmatch\n")
        for index,labels in enumerate(test_labels):
            label_string = format(index) + "\t"  
            for label_index, label in enumerate(labels):
               label_string += label.decode('ascii')
               if label_index < len(labels) - 1:
                   label_string += "\t" 
               else:
                   label_string += "\n" 
            f.write(label_string)
        f.close()

def get_filtered_batch(werk_index, batch_size=10, splice_size=5, white_listed=None):
    count = 0
    inputs = []
    targets = []
    labels = []
    spectra = []
    for ex in werk_index:                       
        # filter based on label content
        if is_white_listed(l_nodes[ex[1]][ex[2]], white_listed):
            inputs.append([f_nodes[ex[1]][ex[2]+x] for x in range (-splice_size,splice_size+1)])
            targets.append(l_nodes[ex[1]][ex[2]][1])
            labels.append(l_nodes[ex[1]][ex[2]])
            spectra.append([s_nodes[ex[1]][ex[2]+x] for x in range (-splice_size,splice_size+1)])
            count += 1
            if count == batch_size: 
                print("breaking news, batch full: ", len(labels))
                break

    shape= np.shape(inputs)
    inputs = np.float32(np.reshape(inputs,(shape[0], shape[1] * shape[2])))
    targets = np.int64(targets)
    spectra = np.float32(spectra)
    return inputs, targets, labels, spectra

def is_white_listed(label_list, white_list):
    for idx, label in enumerate(label_list):
        # as the whitelist does not need to contain all label columns, check it
        if idx >= len(white_list): break    
        if len(white_list[idx]) > 0 and not label in white_list[idx]:
            return False
    return True
        
'''add the prediction of a set of features to the corresponding test labels'''
def add_prediction(test_features, test_labels):
    print("adding prediction to labels")
    test_predictions = predict_batch(test_features)
    for idx, prediction in enumerate(test_predictions):
        '''vowels are in column 1 of test targets, verify if this matches 
           the prediciton'''
        isMatch = (prediction == int(test_labels[idx][1].decode('ascii')))
        test_labels[idx] = np.append(test_labels[idx], isMatch)                        
    return test_labels

'''return the prediciton of an set of features'''
def predict_batch(features):
    print("predicting dataset")    
    test_input = np.float32(np.reshape(f_nodes[0][0:context],(1,feature_size)))    
    feature_columns = learn.infer_real_valued_columns_from_input(test_input)
    classifier = learn.DNNClassifier( 
            feature_columns=feature_columns
            , hidden_units=[1024, 1024, 1024], n_classes=2
            , optimizer = tf.train.AdamOptimizer(1E-4)
            , model_dir= MODEL_DIR)    
    return classifier.predict(x=features, as_iterable = True)
    
def restore(checkpoint_file=MODEL_FILE):
    print("restoring session")    
    with tf.Session() as session:
        restore_saver = tf.train.import_meta_graph(MODEL_FILE + ".meta")
        restore_saver.restore(session, tf.train.latest_checkpoint(MODEL_DIR))
        graph = tf.get_default_graph()

        labels = graph.get_tensor_by_name("output:0")
        inputs = graph.get_tensor_by_name("input:0")

        in_embedding_var = tf.Variable(tf.zeros([batch_size, feature_size]), name="in_embedding")
        in_embedding_op = in_embedding_var.assign(inputs)
        
        l1 = graph.get_tensor_by_name("dnn/hiddenlayer_0/hiddenlayer_0/Relu:0")
        l1_embedding_var = tf.Variable(tf.zeros([batch_size,1024]), name="l1_embedding")
        l1_embedding_op = l1_embedding_var.assign(l1)
        
        l2 = graph.get_tensor_by_name("dnn/hiddenlayer_1/hiddenlayer_1/Relu:0")
        l2_embedding_var = tf.Variable(tf.zeros([batch_size,1024]), name="l2_embedding")
        l2_embedding_op = l2_embedding_var.assign(l2)
        
        l3 = graph.get_tensor_by_name("dnn/hiddenlayer_2/hiddenlayer_2/Relu:0")
        l3_embedding_var = tf.Variable(tf.zeros([batch_size,1024]), name="l3_embedding")
        l3_embedding_op = l3_embedding_var.assign(l3)
        
        session.run(tf.variables_initializer([in_embedding_var, l1_embedding_var, l2_embedding_var, l3_embedding_var]))        
        
        [Train_index, Val_index,Test_index] = restore_index(WORKINDEX_FILE)        
        print('building filtered batch set')
        # whitelist follows layout of target
        # white_list = [[],[KLINKER],[],[],[],[],[],[],[],[b'a', b'A:', b'A+', b'A', b'A~']]
        white_list = [[],[MEDEKLINKER]]
        batch = get_filtered_batch(Test_index, batch_size, white_listed = white_list)
        test_features, test_targets, test_labels, test_spectra  = batch                
        test_labels = add_prediction(test_features, test_labels)
        
        print('running tensorflow with emmbeddings')
        session.run([in_embedding_op, l1_embedding_op, l2_embedding_op, l3_embedding_op], {inputs: test_features, labels: test_targets})
        saver = tf.train.Saver()
        saver.save(session, EMBEDDING_FILE, 1)
        graph_new = tf.get_default_graph()
        
        plot_width, plot_height = sprite.create_sprite(test_spectra, SPRITE_FILE)
        create_metafile(test_labels, EMBEDDING_LABEL_FILE)
        
        print("creating embedding projector")
        # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()
       
        embedding_in = config.embeddings.add()
        embedding_in.tensor_name = in_embedding_var.name
        embedding_in.metadata_path = EMBEDDING_LABEL_FILE
    
        embedding_l1 = config.embeddings.add()
        embedding_l1.tensor_name = l1_embedding_var.name
        embedding_l1.metadata_path = EMBEDDING_LABEL_FILE
        
        embedding_l2 = config.embeddings.add()
        embedding_l2.tensor_name = l2_embedding_var.name
        embedding_l2.metadata_path = EMBEDDING_LABEL_FILE
        
        embedding_l3 = config.embeddings.add()
        embedding_l3.tensor_name = l3_embedding_var.name
        embedding_l3.metadata_path = EMBEDDING_LABEL_FILE

        if USE_SPECTRO:
            plot_width, plot_height = sprite.create_sprite(test_spectra, SPRITE_FILE)
        
            embedding_in.sprite.image_path = SPRITE_FILE
            embedding_in.sprite.single_image_dim.extend([plot_width, plot_height])

            embedding_l1.sprite.image_path = SPRITE_FILE
            embedding_l1.sprite.single_image_dim.extend([plot_width, plot_height])
            
            embedding_l2.sprite.image_path = SPRITE_FILE
            embedding_l2.sprite.single_image_dim.extend([plot_width, plot_height])
            
            embedding_l3.sprite.image_path = SPRITE_FILE
            embedding_l3.sprite.single_image_dim.extend([plot_width, plot_height])
    
    
        writer = tf.summary.FileWriter(EMBEDDING_DIR)
        writer.add_graph(graph_new)
        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(writer, config)
              
def reset():
    print("reset session")
    tf.reset_default_graph()


def main():
    reset()
    restore()
    data_file.close()
    

if __name__ == '__main__':
    main()