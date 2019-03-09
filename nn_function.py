#Functions for CNN
import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

#Weight and bias
def init_weight(shape):
    '''
    Initialize weights for CNN and DNN layers (Return a TF variable)
    '''
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    '''
    Initialize bias for CNN and DNN layers (Returns a TF variable)
    '''
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

#Convolution
def conv1d(x,W):
    '''
    Convolute a 1D array
    x --> Input tensor [batch,H,W,Channels/Color layers] (4 dimensions)
    W --> Kernel with 4 dimensions [filter H, filter W, Channel IN, Channel OUT]
    
    '''
    return tf.nn.conv1d(x,W,strides = [1,1,1,1],padding = 'SAME')

def conv_layer(x,shape,name=None):
    '''
    Returns convolution layer
    '''
    W = init_weight(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv1d(x,W)+b,name=name)

#RNN functions
def create_LSTM_cell(hidden_states):
    '''
        creates a basic LSTM cell
        Args:
        hidden states (int) : Number of hidden units in the cell
        Return:
        Basic LSTM cell
    '''
    #Creating the LSTM cell
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_states, activation = tf.nn.elu)
    #Returning the cell
    return cell

def create_Stacked_cell(cells):
    '''
        creates a basic Multi/Stacked cell structure
        Args:
        cells(list) : List for cells that are to be stacked
        Return:
        MultiRNNcell
    '''
    #Creating the MultiRNN cell
    cell = tf.contrib.rnn.MultiRNNCell(cells = cells)
    #Wrapping the cell    
    return cell

def dropout_wrapper(keep_prob, cell):
    '''
        Wraps the rnn cell's neurons in a dropout layer
        Args:
        keep_proba(tf.placeholder) : Probability of keeping a node's output
        cells(list) : List for cells that are to be stacked
        Return:
        Wrapped RNNcell
    '''
    wrapped = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return wrapped


def create_RNN(cell,inp):
    '''
    Creates a neural network using the provided cell
    Args:
    cell (tf.contrib.rnn) : The cell with which the rnn is to be created
    inp (tf.placeholder) : The input set given to the RNN
    Returns:
    output: Output generator of the neural network
    state: Final state of the neural network
    '''
    return tf.nn.dynamic_rnn(cell,inputs = inp, dtype=tf.float32)

#DNN functions

def dnn_layer(input_layer,size,name=None):
    '''
    Returns DNN layer
    '''
    input_size = int(input_layer.get_shape()[1])
    W = init_weight([input_size,size])
    b = init_bias([size])
    return tf.add(tf.matmul(input_layer,W),b,name=name)
