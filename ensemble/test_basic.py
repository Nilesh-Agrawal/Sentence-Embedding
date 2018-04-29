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
import codecs
import pickle
import argparse

glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.300d.txt"

snli_zip_file = "snli_1.0.zip"
snli_dev_file = "snli_1.0_dev.txt"
snli_full_dataset_file = "snli_1.0_train.txt"

max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 128, 300, 300


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

    def next(self, batch_size=None):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if(batch_size is None):
            batch_size = len(self.x1)
        if self.batch_id == len(self.x1):
            self.batch_id = 0
        batch_x1 = (self.x1[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_x2 = (self.x2[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_correct_values = (self.correct_values[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_correct_scores = (self.correct_scores[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_seqlen1 = (self.seqlen1[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        batch_seqlen2 = (self.seqlen2[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x1))])
        self.batch_id = min(self.batch_id + batch_size, len(self.x1))
        return batch_x1, batch_x2, batch_correct_values, batch_correct_scores, batch_seqlen1, batch_seqlen2, self.maxlen

# Parameters
learning_rate = 0.01
batch_size = 128
display_step = 50
lstm_steps = 30

# Network Parameters
n_hidden = 300 # hidden layer num of features
n_classes = 3 # linear sequence or not


test_data = Dataset(mode='test')
# test_data = Dataset(mode='test')

checkpoint_file = tf.train.latest_checkpoint("./tmp_lstm_avg/")
print("Checkpoint file: ", checkpoint_file)

def extract_embeddings(fp):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            x1 = graph.get_operation_by_name("x1").outputs[0]
            x2 = graph.get_operation_by_name("x2").outputs[0]
            correct_scores = graph.get_operation_by_name("scores").outputs[0]

            seqlen1 = graph.get_operation_by_name("seqlen1").outputs[0]
            seqlen2 = graph.get_operation_by_name("seqlen2").outputs[0]

            
            maxlen = graph.get_operation_by_name("maxlen").outputs[0]
            multiple = graph.get_operation_by_name("multiple").outputs[0]

            accuracy = graph.get_operation_by_name("accuracy").outputs[0]
            cost = graph.get_operation_by_name("cost").outputs[0]
            

            # Run the initializer
            batch_x1, batch_x2, batch_values, batch_scores, batch_seqlen_1, batch_seqlen_2, maxlength = test_data.next()
            batch_x1 = np.asarray(batch_x1)
            batch_x2 = np.asarray(batch_x2)
            batch_values = np.asarray(batch_values)
            batch_scores = np.asarray(batch_scores)
            batch_seqlen_1 = np.asarray(batch_seqlen_1)
            batch_seqlen_2 = np.asarray(batch_seqlen_2)
            # Run optimization op (backprop)
            acc, loss = sess.run([accuracy, cost], feed_dict={x1: batch_x1,
                                           x2: batch_x2, 
                                           correct_scores: batch_scores,
                                           seqlen1: batch_seqlen_1,
                                           seqlen2: batch_seqlen_2,
                                           maxlen: maxlength})
            
            print("Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences_file")
    args = parser.parse_args()
    return args

def main():
    args  = parse_args()
    sentence_file = args.sentences_file
    fp = open(sentence_file,"r")
    embeddings = extract_embeddings(fp)

if __name__ == '__main__':
    main()