# %%
import json
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# %% Set args

wordlist,matrix=pickle.load(open("wordlist&matrix@8017.dump","rb"))
charlen = len(wordlist)
window = 10
width = 512
dropout = 0.25
learnrate = 0.001
dimension = 300
charlen
# %% Build model
input_tensor = Input(shape=(window,))
embedd = Embedding(charlen, dimension,
                   weights=[matrix],
                   input_length=window,
                   trainable=False)(input_tensor)
lstm = Bidirectional(GRU(width, return_sequences=True))(embedd)

flatten = Flatten()(lstm)
dense = Dense(charlen, activation='softmax')(flatten)
# dense = Dense(dimension, activation="linear")(flatten)
model = Model(inputs=input_tensor, outputs=dense)
optimizer = Adam(lr=learnrate)
model.compile(loss='categorical_crossentropy',
                   optimizer=optimizer, metrics=['accuracy'])

model.summary()
# %% Generate training set

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


openfolder = "./TXT_origin/"
savefolder = "./TXT_processed/"


def get(tlist):
    settocat = []
    anstocat = []
    for i in tlist:
        settocat.append(pickle.load(open(savefolder+i+".set", "rb")))
        anstocat.append(pickle.load(open(savefolder+i+".ans", "rb")))

    tset = np.concatenate(settocat)
    tans = np.concatenate(anstocat)
    return tset, tans


trainlist = [i for i in os.listdir(openfolder) if i[-4:] == ".txt"]
trainset, trainans = get(trainlist)
testset, testans = get(["1.txt"])

trainans = to_categorical(trainans, charlen)
testans = to_categorical(testans, charlen)

print(trainset.shape, trainans.shape)
print(testset.shape, testans.shape)

# %%
history = model.fit(
    trainset,
    trainans,
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_data=(testset, testans)
)

model.save("GRU_FixedEmbedding_{}_{}.h5".format(
    history.history["loss"][-1], history.history["acc"][-1]))
# %%

model = load_model(
    "GRU_FixedEmbedding_0.057953879376296084_0.9953821686562302.h5")
# %%


def sample(preds, temperature=1.0):
    '''
    当temperature=1.0时，模型输出正常
    当temperature=0.5时，模型输出比较open
    当temperature=1.5时，模型输出比较保守
    在训练的过程中可以看到temperature不同，结果也不同
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# %%


sentence = "左世纪晚上拉屎不冲厕所，还不坚守"
print(sentence, end="")
sentence = articletonum(sentence)[-window:]
for i in range(3000):

    r = model.predict(np.array(sentence)[None, ...])

    c = sample(r[0],0.01)
    print(wordlist[c], end="")

    sentence = sentence[1:]+[c]
# %%
