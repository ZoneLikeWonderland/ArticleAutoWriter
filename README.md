# ArticleAutoWriter
Write articles by machine learning, reciting almost.

# Files' use

## charset.json

>`charset`: list of all words from origin texts.
`["得奖", "脚步", "身患", "爱做", "那只", ...]`



## weights_process.py

generate `wordlist` and embedding `matrix` from `charset`.

>`wordlist`: list of all words in use (in order).
`['，', '的', '。', '、', ',' ...]`

>`matrix`: np.array of word vectors.
`matrix.shape == (8017, 300)`



## TXTprocess.py

generate `.set` and `.ans` files from origin texts.

> `.set`: training set

> `.ans`: training answer



## model.py

main part of the model.

