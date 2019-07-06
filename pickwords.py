# %%
from tqdm import tqdm
import json
import os

openfolder = "./TXT_origin/"
savefolder = "./TXT_processed/"
# %%
for i in [i for i in os.listdir(openfolder) if i[-4:] == ".txt"]:
    try:
        open(savefolder+i+".words", "r", encoding="utf8")
        continue
    except:
        pass
    seq = set()
    try:
        article = open(openfolder+i, encoding="utf8").read()
    except:
        article = open(openfolder+i, encoding="gbk").read()
    k = 0
    with tqdm(total=len(article)) as pbar:
        while True:
            if k >= len(article):
                break
            for l in range(5, 0, -1):
                if article[k:k+l] not in seq and article[k:k+l] in wordlist:
                    seq.add(article[k:k+l])
                    k += l
                    pbar.update(l)
                    break
                if l == 1:
                    k += 1
                    pbar.update()
    json.dump(list(seq), open(savefolder+i+".words",
                              "w", encoding="utf8"), ensure_ascii=False)


# %%
wordset=set()
for i in [i for i in os.listdir(savefolder) if i[-6:] == ".words"]:
    print(i)
    wordset.update(json.load(open(savefolder+i,encoding="utf8")))
json.dump(list(wordset),open("charset.json","w",encoding="utf8"),ensure_ascii=False)