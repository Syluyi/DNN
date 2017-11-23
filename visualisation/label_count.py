# -*- coding: utf-8 -*-
import numpy as np
import tables as tb
import operator

VERSION = "fb_fnALL_spectro_base25"
WORK_DIR = "/ubuntu/data/"
DATA_DIR = WORK_DIR + "/fb_prepared/"
INPUT_FILE = DATA_DIR + VERSION + ".h5"
print("opening h5 prepared audio file")

h5_file = tb.open_file(INPUT_FILE, "r+")

print("reading l_nodes from audio file")
l_nodes = h5_file.root.labels._f_list_nodes()

def main():
    print("label count main program")    
    '''
    file-lbl-node -> a node with an audio file name and all labelnodes for it
    lbl-node -> a node with 10 audio characteristics of 10 ms of the audio file
    l-nodes -> collection of filelbl-nodes
    Loop all l_nodes to handle each filelbl-node
       loop all the lbl_nodes to collect the statistics for a file
           count occurences of all the types of interest                      
    '''
    my_stats = np.array([{},{},{},{},{},{},{},{},{},{}])
    print(my_stats)
    for idx_file_node, file_lbl_node in enumerate(l_nodes):
        print(idx_file_node, file_lbl_node )
        for idx_lbl_node, lbl_node in enumerate(file_lbl_node):
            for idx_lbl, lbl in enumerate(lbl_node):
                if idx_lbl != 9:
                    dict_index = int(lbl_node[idx_lbl])
                else:
                    dict_index = lbl_node[idx_lbl].decode('ascii')
                #.decode('ascii')
                try:
                    my_stats[idx_lbl][dict_index] += 1
                except (KeyError):
                    my_stats[idx_lbl][dict_index] = 1                
    
    # sort and print result
    for idx_dict_stats, dict_stats in enumerate(my_stats):
        sorted_stats = sorted(dict_stats.items(), key=operator.itemgetter(0))
        print("Index label: " + format(idx_dict_stats,""))
        for idx_stats, stats in sorted_stats:
            print("\t key and #: ", "{:>6}".format(idx_stats), str("{:,}".format(stats)).rjust(12))
        
    
if __name__ == '__main__':
    main()