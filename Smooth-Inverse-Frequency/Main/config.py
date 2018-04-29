import os.path
"""
URLS
"""
cwd = os.path.abspath(os.path.dirname(__file__))
url_glove_dataset = os.path.join(cwd, "../data/glove.840B.300d.txt")

url_glove_pickles = url_glove_pickle_we_1 = os.path.join(cwd, "Glove_Pickles/")

url_glove_pickle_we_1 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_1000000_We.pickle")
url_glove_pickle_we_2 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_2000000_We.pickle")
url_glove_pickle_we_3 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_Last_We.pickle")

url_glove_pickle_words_1 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_1000000_words.pickle")
url_glove_pickle_words_2 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_2000000_words.pickle")
url_glove_pickle_words_3 = os.path.join(cwd, "Glove_Pickles/glove_embeddings_Last_words.pickle")

url_enwiki = os.path.join(cwd, "../data/enwiki_vocab_min200.txt")

url_snli_1 = os.path.join(cwd, "../../InferSent/dataset/SNLI/s1.train")
url_snli_2 = os.path.join(cwd, "../../InferSent/dataset/SNLI/s2.train")

url_snli_pc1 = os.path.join(cwd, "SNLI_PC/s1.train_pc.pickle")
url_snli_pc2 = os.path.join(cwd, "SNLI_PC/s2.train_pc.pickle")
