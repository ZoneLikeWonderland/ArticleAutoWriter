#%%
import numpy as np
import re
import json
from tqdm import tqdm
import pickle
#%%
charset=json.load(open("charset.json",encoding="utf8"))

uri=r"D:\DOWNLOAD!!!\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
f=open(uri,encoding="utf8").read()
f=f.split("\n")[1:-1]
        
MAXLEN=5
wordlist=[]
matrix=[]
for i in tqdm(f):
    char=i[:i.find(" ")]
    if char not in charset:
        continue
    if len(char)>MAXLEN or re.search(r"[0-9a-zA-Z-+=_\.\*]",char):
        continue
    wordlist.append(char)
    j=np.fromstring(i[i.find(" "):],sep=" ")
    matrix.append(j)
matrix=np.array(matrix)
del f
pickle.dump((wordlist,matrix),open("wordlist&matrix@{}.dump".format(len(wordlist)),"wb"))
#%%
