import pickle

with open("SIF_Pickles/s1.train_pc.pickle", "rb") as fp:
    pc1 = pickle.load(fp)

with open("SIF_Pickles/s2.train_pc.pickle", "rb") as fp:
    pc2 = pickle.load(fp)

with open("SIF_Pickles/word2weight.pickle", "rb") as fp:
    word_weights = pickle.load(fp)

with open("SIF_Pickles/s1.train_pc.pickle", "wb") as fp:
    pickle.dump(pc1, fp, protocol=2)

with open("SIF_Pickles/s2.train_pc.pickle", "wb") as fp:
    pickle.dump(pc2, fp, protocol=2)

with open("SIF_Pickles/word2weight.pickle", "wb") as fp:
    pickle.dump(word_weights, fp, protocol=2)
