import sys
import os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cwd, "Main"))
import Compute_sentence_emb as main

class SIF_API():
    def __init__(self):
        self.main_emb = main.SIF()
    def get_emb(self, sent):
        result = self.main_emb.compute_sif_emb(sent)
        return result
