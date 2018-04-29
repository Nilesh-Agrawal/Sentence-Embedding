import sys
import os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cwd)
import fetch_sent_emb_snli_sif as snli_emb

sif = snli_emb.SIF_API()

sent = [[this is a sentence], [this is another sentence], [Yet another sentence],[i am the king], [i am the lord], [I am everything]]

emb = sif.get_emb(sent)
