from __future__ import print_function

import tensorflow as tf
import random
import load_input_and_output as ld
import numpy as np
import codecs
import pickle
import argparse

max_length = 30
batch_size, vector_size, hidden_size = 128, 300, 300


def build_glove():

    glove_zip_file = "glove.6B.zip"
    glove_vectors_file = "glove.6B.300d.txt"

    snli_zip_file = "snli_1.0.zip"
    snli_dev_file = "snli_1.0_dev.txt"
    snli_full_dataset_file = "snli_1.0_train.txt"

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

    return glove_wordmap, glove_idx2word, glove_word2idx, pc1, pc2, word_weights

def sentence2sequence(sentence, glove_word2idx):
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

def split_data_into_scores(sentences_list, glove_word2idx):
    
    sentences = []
    seq_len = []
    for row in sentences_list:
        try:
            sentences.append(np.vstack(
                sentence2sequence(row.lower(), glove_word2idx)))
            seq_len.append(len(row))
        except ValueError:
            continue
    
    padded = [fit_to_size(x, max_length)
                      for x in sentences]

    sentences = np.stack(padded)
                                 
    return sentences, seq_len


class Dataset(object):
    def __init__(self,sentences,glove_wordmap, glove_idx2word, glove_word2idx, pc1, pc2, word_weights):
        self.x, self.seqlen = split_data_into_scores(sentences, glove_word2idx)
        n_samples = len(self.seqlen)
        self.maxlen = 30
        self.batch_id = 0
        self.glove_wordmap = glove_wordmap
        self.glove_idx2word = glove_idx2word
        self.glove_word2idx = glove_word2idx
        self.pc1 = pc1
        self.pc2 = pc2
        self.word_weights = word_weights

    def get_word_avg(self, X):
        X_avg = []
        for x in X:
            lst = [0.0]*vector_size
            for i in range(len(x)):
                if x[i] in self.glove_idx2word and self.glove_idx2word[x[i]] in self.word_weights:
                    lst = np.add(lst,[float(num)*self.word_weights[self.glove_idx2word[x[i]]] for num in self.glove_wordmap[x[i]]])
            X_avg.append(lst-self.pc2)
        return X_avg

    def next(self, batch_size=None):
        if(batch_size is None):
            batch_size = len(self.x)
        if self.batch_id == len(self.x):
            self.batch_id = 0
        batch_x1 = (self.x[self.batch_id:min(self.batch_id + batch_size, len(self.x))])
        
        batch_x2 = batch_x1

        batch_x1_sif = (self.get_word_avg(self.x[self.batch_id:min(self.batch_id + batch_size, len(self.x))]))
        
        batch_x2_sif = batch_x1_sif

        batch_seqlen1 = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.x))])
        batch_seqlen2 = batch_seqlen1

        self.batch_id = min(self.batch_id + batch_size, len(self.x))

        return batch_x1, batch_x2, batch_x1_sif, batch_x2_sif, batch_seqlen1, batch_seqlen2, self.maxlen


# Parameters
learning_rate = 0.01
batch_size = 128
display_step = 50
lstm_steps = 30

# Network Parameters
n_hidden = 300 # hidden layer num of features
n_classes = 3 # linear sequence or not


def extract_embeddings(sentences):
    glove_wordmap, glove_idx2word, glove_word2idx, pc1, pc2, word_weights = build_glove()
    test_data = Dataset(sentences, glove_wordmap, glove_idx2word, glove_word2idx, pc1, pc2, word_weights)
    # test_data = Dataset(mode='test')

    checkpoint_file = tf.train.latest_checkpoint("./tmp_sif_reg/")
    print("Checkpoint file: ", checkpoint_file)


    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            x1 = graph.get_operation_by_name("x1").outputs[0]
            x2 = graph.get_operation_by_name("x2").outputs[0]
            x1_sif = graph.get_operation_by_name("x1_sif").outputs[0]
            x2_sif = graph.get_operation_by_name("x2_sif").outputs[0]
            # correct_scores = graph.get_operation_by_name("scores").outputs[0]

            seqlen1 = graph.get_operation_by_name("seqlen1").outputs[0]
            seqlen2 = graph.get_operation_by_name("seqlen2").outputs[0]

            
            maxlen = graph.get_operation_by_name("maxlen").outputs[0]
            multiple = graph.get_operation_by_name("multiple").outputs[0]

            # accuracy = graph.get_operation_by_name("accuracy").outputs[0]
            # cost = graph.get_operation_by_name("cost").outputs[0]
            sentemb1 = graph.get_operation_by_name("sentemb1").outputs[0]
            sentemb2 = graph.get_operation_by_name("sentemb2").outputs[0]


            # Run the initializer
            batch_x1, batch_x2, batch_x1_sif, batch_x2_sif, batch_seqlen_1, batch_seqlen_2, maxlength = test_data.next()
            batch_x1 = np.asarray(batch_x1)
            batch_x2 = np.asarray(batch_x2)
            batch_x1_sif = np.asarray(batch_x1_sif)
            batch_x2_sif = np.asarray(batch_x2_sif)
            batch_seqlen_1 = np.asarray(batch_seqlen_1)
            batch_seqlen_2 = np.asarray(batch_seqlen_2)
            # Run optimization op (backprop)
            emb = sess.run([sentemb1], feed_dict={x1: batch_x1,
                                           x2: batch_x2, 
                                           x1_sif: batch_x1_sif,
                                           x2_sif: batch_x2_sif,
                                           seqlen1: batch_seqlen_1,
                                           seqlen2: batch_seqlen_2,
                                           maxlen: maxlength})
            emb = np.asarray(emb)
            return emb.reshape((-1))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences_file")
    args = parser.parse_args()
    return args

def main():
    args  = parse_args()
    sentences = args.sentences_file
    embeddings = extract_embeddings(sentences)

if __name__ == '__main__':
    main()