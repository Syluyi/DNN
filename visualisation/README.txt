Code for visualizing the output of every layer in a DNN that is created using tensorflow.

- embedding_all: main program to create embeddings with all input
- embedding_klinker: main program to create embeddings with only klinkers as input
- embedding_medeklinker: main program to create embeddings with only medeklinkers as input
- rewrite_labels: Rewrite the metadata file to include category information of the original input.

After running either of the main programs, you have can visualize the embeddings by starting tensorboard in the command line with:

tensorboard --logdir="PATH/TO/EMBEDDING_DIRECTORY/"

Now you can go to http://localhost:6006/#embeddings in your browser to visualise the embeddings.