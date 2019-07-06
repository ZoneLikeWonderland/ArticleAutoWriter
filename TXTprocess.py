# %%
import pickle
import threading
import os
window = 10
# %%


def articletonum(article):
    k = 0
    seq = []
    with tqdm(total=len(article)) as pbar:
        while True:
            if k >= len(article):
                break
            for l in range(5, 0, -1):
                try:
                    seq.append(wordlist.index(article[k:k+l]))
                    k += l
                    pbar.update(l)
                    break
                except:
                    pass
                if l == 1:
                    k += 1
                    pbar.update()
    return seq
# %%


def handleF(f):
    try:
        open(savefolder+f+".set", "rb")
        return
    except:
        pass
    try:
        s = open(openfolder+f, encoding="utf8").read()
    except:
        s = open(openfolder+f, encoding="gbk").read()

    article = re.sub("\s", "", s).replace("\n\n", "\n")
    seq = articletonum(article)
    tset = []
    tans = []
    for i in range(len(seq)-window):
        tset.append(np.array(seq[i:i+window]))
        tans.append(np.array(seq[i+window]))

    pickle.dump(np.array(tset), open(savefolder+f+".set", "wb"))
    pickle.dump(np.array(tans), open(savefolder+f+".ans", "wb"))


def makeset(fp):
    for f in fp:
        handleF(f)


openfolder = "./TXT_origin/"
savefolder = "./TXT_processed/"


trainlist = [i for i in os.listdir(openfolder) if i[-4:] == ".txt"]

makeset(trainlist)


# %%
