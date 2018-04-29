import pickle
import data_io, SIF_embedding, params
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import numpy as np
import config

params = params.params()
params.rmpc = 1

def get_similarity_scores(We, x1, w1, x2, w2):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
    :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
    :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
    :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: scores, scores[i] is the matching score of the pair i
    """

    print("Getting similarity scores")
    emb1 = SIF_embedding.SIF_embedding(We, x1, w1, params, file_name="s1.train_")
    with open("s1_train_sentence_emb.pickle", "wb") as file:
        pickle.dump(emb1, file)
    print("embeddings for s1.train fetched")
    emb2 = SIF_embedding.SIF_embedding(We, x2, w2, params, file_name="s2.train_")
    with open("s2_train_sentence_emb.pickle", "wb") as file:
        pickle.dump(emb2, file)
    print("embeddings for s2.train fetched")

    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm
    print("scores computed")
    with open("snli_scores.pickle", "wb") as file:
        pickle.dump(scores, file)
    return scores

def get_emb(We, x1, w1):
    emb1 = SIF_embedding.get_weighted_average(We, x1, w1)

    with open(config.url_snli_pc1, "rb") as file:
        pc1 = pickle.load(file);

    emb1 = emb1 - emb1.dot(pc1.transpose()) * pc1

    return [emb1]



class SIF():
    def __init__(self):
        self.weightfile = config.url_enwiki
        self.weightpara = 1e-3

        print("Getting embeddings from the Glove pickles")

        with open(config.url_glove_pickle_we_1, "rb") as file:
            We_1 = pickle.load(file)

        with open(config.url_glove_pickle_words_1,"rb") as file:
            words_1 = pickle.load(file)


        with open(config.url_glove_pickle_we_2, "rb") as file:
            We_2 = pickle.load(file)

        with open(config.url_glove_pickle_words_2,"rb") as file:
            words_2 = pickle.load(file)

        with open(config.url_glove_pickle_we_3, "rb") as file:
            We_3 = pickle.load(file)

        with open(config.url_glove_pickle_words_3,"rb") as file:
            words_3 = pickle.load(file)

        self.We = []
        self.We.extend(We_1)
        self.We.extend(We_2)
        self.We.extend(We_3)

        self.words = {}
        self.words.update(words_1)
        self.words.update(words_2)
        self.words.update(words_3)

        with open(config.url_snli_pc1, "rb") as file:
            self.snli_pc_1 = pickle.load(file)

        with open(config.url_snli_pc2, "rb") as file:
            self.snli_pc_2 = pickle.load(file)

        print("Successfully got the embeddings from the pickle")

        self.word2weight = data_io.getWordWeight(self.weightfile, self.weightpara) # word2weight['str'] is the weight for the word 'str'
        self.weight4ind = data_io.getWeight(self.words, self.word2weight) # weight4ind[i] is the weight for the i-th word



    def compute_sif_emb(self, sentences):
        # load sentences
        x1, m = data_io.sentences2idx(sentences, self.words)
        w1 = data_io.seq2weight(x1, m, self.weight4ind)

        result = get_emb(self.We, x1, w1)
        return result
