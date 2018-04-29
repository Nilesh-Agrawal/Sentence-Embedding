""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import random
import load_input_and_output as ld
import numpy as np
from time import gmtime, strftime
import codecs
import pickle
import sys
import time
# ====================
#  TOY DATA GENERATOR
# ====================
glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.300d.txt"

snli_zip_file = "snli_1.0.zip"
snli_dev_file = "snli_1.0_dev.txt"
snli_full_dataset_file = "snli_1.0_train.txt"

max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 128, 300, 64


glove_wordmap = []
glove_wordmap.append([0]*300)
glove_word2idx = {}
glove_idx2word = {}
with codecs.open(glove_vectors_file, "r", encoding='utf-8') as glove:
    count = 0
    for line in glove:
        if count%10000== 0:
            print(count)
        try:
            name, vector = tuple(line.split(" ", 1))
        except ValueError:
            continue
        count += 1
        if(count >= 50000):
            break
        glove_wordmap.append(list(np.fromstring(vector, sep=" ")))
        glove_word2idx[name] = count
        glove_idx2word[count] = name

with open("SIF_Pickles/s1.train_pc.pickle", "rb") as fp:
    pc1 = pickle.load(fp)
pc1 = pc1.reshape((-1))

with open("SIF_Pickles/s2.train_pc.pickle", "rb") as fp:
    pc2 = pickle.load(fp)
pc2 = pc2.reshape((-1))

with open("SIF_Pickles/word2weight.pickle", "rb") as fp:
    word_weights = pickle.load(fp)

def sentence2sequence(sentence):
    """
     
    - Turns an input sentence into an (n,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
    
      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_word2idx:
                rows.append(glove_word2idx[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows

def score_setup(row):
    convert_dict = {
      'entailment': 0,
      'neutral': 1,
      'contradiction': 2
    }
    score = np.zeros((3,))
    for x in range(1,6):
        tag = row["label"+str(x)]
        if tag in convert_dict: score[convert_dict[tag]] += 1
    return score / (1.0*np.sum(score))

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    matrix = matrix.reshape((-1))
    if len(matrix) >= shape:
        matrix = matrix[:shape]
    else:
        matrix =  np.pad(matrix, (0, shape-len(matrix)), 'constant')
    if(len(matrix) < 30):
        print(len(matrix))
    return matrix

def split_data_into_scores(mode):
    import csv
    with open("snli_1.0_"+mode+".txt","r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        seq_lengths_1 = []
        seq_lengths_2 = []
        for row in train:
            try:
                hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence1"].lower())))
                evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence2"].lower())))
            except ValueError:
                continue
            labels.append(row["gold_label"])
            scores.append(score_setup(row))
            seq_lengths_1.append(len(row['sentence1']))
            seq_lengths_2.append(len(row['sentence2']))
        
        padded_hyp = [fit_to_size(x, max_hypothesis_length)
                          for x in hyp_sentences]

        hyp_sentences = np.stack(padded_hyp)
        evi_sentences = np.stack([fit_to_size(x, max_evidence_length)
                          for x in evi_sentences])
                                     
        return hyp_sentences, evi_sentences, labels, np.array(scores), seq_lengths_1, seq_lengths_2


class Dataset(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self,mode):
        self.x1, self.x2, self.correct_values, self.correct_scores, self.seqlen1, self.seqlen2 = split_data_into_scores(mode)
        n_samples = len(self.correct_values)
        self.maxlen = 30
        self.batch_id = 0

    def get_word_avg(self, X):
        X_avg = []
        for x in X:
            lst = [0.0]*vector_size
            for i in range(len(x)):
                if x[i] in glove_idx2word and glove_idx2word[x[i]] in word_weights:
                    lst = np.add(lst,[float(num)*word_weights[glove_idx2word[x[i]]] for num in glove_wordmap[x[i]]])
            X_avg.append(lst-pc1)
        return X_avg

    def next(self, batch_size=None):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if(batch_size is None):
        	batch_size = len(self.x1)
        if self.batch_id == len(self.x1):
            self.batch_id = 0
        batch_x1 = (self.x1[self.batch_id:min(self.batch_id + batch_size, len(self.x1))])
        
        batch_x2 = (self.x2[self.batch_id:min(self.batch_id + batch_size, len(self.x1))])
        
        batch_x1_sif = (self.get_word_avg(self.x1[self.batch_id:min(self.batch_id + batch_size, len(self.x1))]))
        
        batch_x2_sif = (self.get_word_avg(self.x2[self.batch_id:min(self.batch_id + batch_size, len(self.x2))]))

        batch_correct_values = (self.correct_values[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_correct_scores = (self.correct_scores[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_seqlen1 = (self.seqlen1[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_seqlen2 = (self.seqlen2[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        self.batch_id = min(self.batch_id + batch_size, len(self.x1))
        return batch_x1, batch_x2, batch_x1_sif, batch_x2_sif, batch_correct_values, batch_correct_scores, batch_seqlen1, batch_seqlen2, self.maxlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 100000
batch_size = 128
display_step = 50
lstm_steps = 30

# Network Parameters
n_hidden = 300 # hidden layer num of features
n_classes = 3 # linear sequence or not


train_data = Dataset(mode='train')
valid_data = Dataset(mode='dev')


# tf Graph input
x1 = tf.placeholder(tf.int32, [None, lstm_steps],name='x1')
x2 = tf.placeholder(tf.int32, [None, lstm_steps],name='x2')
x1_sif = tf.placeholder(tf.float32, [None, n_hidden],name='x1_sif')
x2_sif = tf.placeholder(tf.float32, [None, n_hidden],name='x2_sif')
correct_scores = tf.placeholder("float",[None, n_classes],name='scores')

# A placeholder for indicating each osequence length
seqlen1 = tf.placeholder(tf.int32, [None], name='seqlen1')
seqlen2 = tf.placeholder(tf.int32, [None], name='seqlen2')
maxlen = tf.placeholder(tf.int32,name='maxlen')

multiple = tf.constant([1,30],name='multiple')
word_idx_map = tf.Variable(glove_wordmap,name="glove_map")
# Define weights
weights =  tf.Variable(tf.random_normal([n_hidden*2, n_classes]),name="W")

biases =  tf.Variable(tf.random_normal([n_classes]),name='b')

weights_concat =  tf.Variable(tf.random_normal([n_hidden*2, n_hidden]),name="W_c")

biases_concat =  tf.Variable(tf.random_normal([n_hidden]),name='b_c')

def dynamicRNN(x_idx, seqlen, maxlen, weights, biases, word_idx_map, reuse=False):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.nn.embedding_lookup(word_idx_map, x_idx)
    x = tf.unstack(x, lstm_steps, 1)

    
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse=reuse)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    
    outputs = tf.transpose(outputs, [1, 0, 2])
    rnn_outputs = outputs
    outputs = tf.stack(outputs)

    batch_size = tf.shape(outputs)[0]
    outputs = tf.reduce_mean(outputs,1)
    # index = tf.range(0, batch_size) * maxlen + (seqlen - 1)
    # outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    
    return outputs


outputs1 = dynamicRNN(x1, seqlen1, maxlen, weights, biases, word_idx_map)
outputs2 = dynamicRNN(x2, seqlen2, maxlen, weights, biases, word_idx_map, True)

sentemb1 = tf.nn.relu(tf.add(tf.matmul(tf.concat([outputs1,x1_sif],1), weights_concat), biases_concat), name='sentemb1')
sentemb2 = tf.nn.relu(tf.add(tf.matmul(tf.concat([outputs2,x2_sif],1), weights_concat), biases_concat ), name='sentemb2')
preds_score = tf.add(tf.matmul(tf.concat([sentemb1,sentemb2],1), weights), biases, name="pred_score")
    
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds_score, labels=correct_scores),name='cost')
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(preds_score,1), tf.argmax(correct_scores,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64),name="accuracy")

# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.global_variables())

# Start training
with tf.Session() as sess:

    # Run the initializer

    sess.run(init)
    prev_loss = 1000.0
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
    
    for step in range(1, training_steps + 1):
        batch_x1, batch_x2, batch_x1_sif, batch_x2_sif, batch_values, batch_scores, batch_seqlen_1, batch_seqlen_2, maxlength = train_data.next(batch_size)
        batch_x1 = np.asarray(batch_x1)
        batch_x2 = np.asarray(batch_x2)
        batch_x1_sif = np.asarray(batch_x1_sif)
        batch_x2_sif = np.asarray(batch_x2_sif)
        batch_values = np.asarray(batch_values)
        batch_scores = np.asarray(batch_scores)
        batch_seqlen_1 = np.asarray(batch_seqlen_1)
        batch_seqlen_2 = np.asarray(batch_seqlen_2)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x1: batch_x1,
                                       x2: batch_x2, 
                                       x1_sif: batch_x1_sif,
                                       x2_sif: batch_x2_sif,
        							   correct_scores: batch_scores,
                                       seqlen1: batch_seqlen_1,
                                       seqlen2: batch_seqlen_2,
                                       maxlen: maxlength})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            batch_x1, batch_x2, batch_x1_sif, batch_x2_sif, batch_values, batch_scores, batch_seqlen_1, batch_seqlen_2, maxlength = valid_data.next()
            batch_x1 = np.asarray(batch_x1)
            batch_x2 = np.asarray(batch_x2)
            batch_x1_sif = np.asarray(batch_x1_sif)
            batch_x2_sif = np.asarray(batch_x2_sif)
            batch_values = np.asarray(batch_values)
            batch_scores = np.asarray(batch_scores)
            batch_seqlen_1 = np.asarray(batch_seqlen_1)
            batch_seqlen_2 = np.asarray(batch_seqlen_2)

            acc, loss, pred_scores = sess.run([accuracy, cost, preds_score], feed_dict={x1: batch_x1,
                                                              x2: batch_x2, 
                                                              x1_sif: batch_x1_sif,
                                                              x2_sif: batch_x2_sif,
            												  correct_scores: batch_scores,
                                                              seqlen1: batch_seqlen_1,
                                                              seqlen2: batch_seqlen_2,
                                                			  maxlen: maxlength})
            print("Time: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ", Step " + str(step*batch_size) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Validation Accuracy= " + \
                  "{:.5f}".format(acc))
            if prev_loss > loss:
                save_path = saver.save(sess, "./tmp_sif_reg_avg/model",global_step=1000)
                print("Model saved in path: %s" % save_path)
                prev_loss = loss

    
    
    
    print("Optimization Finished!")