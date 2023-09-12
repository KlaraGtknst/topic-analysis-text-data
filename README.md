# Topic analysis text data
*Project name* is a module that collects different sentence embedding methods in a database to make a large text corpus searchable and accessable.

We provide our pre-trained English sentence encoder from [our paper](https://arxiv.org/abs/1705.02364) and our [SentEval](https://github.com/facebookresearch/SentEval) evaluation toolkit.

**Recent changes**: 
TODO

## Dependencies

This code is written in Python. Dependencies include:

* Python 2/3
* [Pytorch](http://pytorch.org/) (recent version)
* NLTK >= 3
* Look at requirements.txt

## Download word vectors

Create a environment
(TODO)
```bash
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/
```

## Use our sentence encoder
Get started with the following steps: (TODO)

*0.0) Download our InferSent models (V1 trained with GloVe, V2 trained with fastText)[147MB]:*
```bash
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
```
Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer) and infersent2 is trained with fastText (which have been trained on text preprocessed with the MOSES tokenizer). The latter also removes the padding of zeros with max-pooling which was inconvenient when embedding sentences outside of their batches.

*0.1) Make sure you have the NLTK tokenizer by running the following once:*
```python
import nltk
nltk.download('punkt')
```

*1) [Load our pre-trained model](https://github.com/facebookresearch/InferSent/blob/master/encoder/demo.ipynb) (in encoder/):*
```python
from models import InferSent
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
```

*2) Set word vector path for the model:*
```python
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)
```

*3) Build the vocabulary of word vectors (i.e keep only those needed):*
```python
infersent.build_vocab(sentences, tokenize=True)
```
where *sentences* is your list of **n** sentences. You can update your vocabulary using *infersent.update_vocab(sentences)*, or directly load the **K** most common English words with *infersent.build_vocab_k_words(K=100000)*.
If **tokenize** is True (by default), sentences will be tokenized using NTLK.

*4) Encode your sentences (list of *n* sentences):*
```python
embeddings = infersent.encode(sentences, tokenize=True)
```
This outputs a numpy array with *n* vectors of dimension **4096**. Speed is around *1000 sentences per second* with batch size 128 on a single GPU.

*5) Visualize the importance that our model attributes to each word:*

We provide a function to visualize the importance of each word in the encoding of a sentence:
```python
infersent.visualize('A man plays an instrument.', tokenize=True)
```
![Model](https://dl.fbaipublicfiles.com/infersent/visualization.png)


## Reference
TODO
Please consider citing [[1]](https://arxiv.org/abs/1705.02364) if you found this code useful.

### Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (EMNLP 2017)

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

```
@InProceedings{conneau-EtAl:2017:EMNLP2017,
  author    = {Conneau, Alexis  and  Kiela, Douwe  and  Schwenk, Holger  and  Barrault, Lo\"{i}c  and  Bordes, Antoine},
  title     = {Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {670--680},
  url       = {https://www.aclweb.org/anthology/D17-1070}
}
```

### Related work TODO
* [J. R Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun, S. Fidler - SkipThought Vectors, NIPS 2015](https://arxiv.org/abs/1506.06726)


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
