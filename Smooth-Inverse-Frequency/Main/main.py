import data_io, config

# input
wordfile = config.url_glove_dataset
weightfile = config.url_enwiki

weightpara = 1e-3
rmpc = 1
sentences = []

# load word vectors

print("Creating the glove pickles")
#(words, We) = data_io.lokesh_getWordmap(wordfile)
data_io.create_glove_pickles(wordfile)
