import time
from enum import Enum
from collections import defaultdict
import numpy as np
from scipy.special import softmax, expit as sigmoid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize, MinMaxScaler
from numba import jit, njit

DETERMINISTIC = 42

np.seterr(all='raise')
np.random.seed(DETERMINISTIC)

@jit
def tokenizer(corpus):
    phrases = corpus.strip().lower().split(".")
    return [words.strip().split() for words in phrases if len(words) > 0]


@njit(parallel=True)
def layer(x, w, b):
    return np.dot(w.T, x) + b

@njit(parallel=True)
def gradient(m, yh, y):
    return (1/m) * (yh - y)


@jit(parallel=True, forceobj=True)
def backprop(dZ2, h, u, bx, w2):
    dW2 = np.outer(h, dZ2)
    dB2 = np.sum(dZ2, axis=0)
    dZ1 = np.dot(w2, dZ2)
    dW1 = np.outer(bx, dZ1)
    dB1 = np.sum(dZ1, axis=0)
    return dW1, dB1, dW2, dB2


class Word2VecModel(Enum):
    SKIP_GRAM = 1
    CBOW = 2


class Word2Vec:
    
    def __init__(self, 
                 model=Word2VecModel.SKIP_GRAM,
                 learning_rate=1e-1,
                 window=3,
                 latent_space=5,
                 norm=True
    ):
        self.model = model
        self.window = window
        self.lr = learning_rate
        self.lt = latent_space
        self.norm = norm
    
    def tokenizer(self, corpus):
        return tokenizer(corpus)

    def __generate_indexes(self, corpus):
        word_counts = defaultdict(int)
        for phrases in corpus:
            for word in phrases:
                word_counts[word] += 1
        v_count = len(word_counts.keys())

        words_list = list(word_counts.keys())
        word_index = dict((word, i) for i, word in enumerate(words_list))
        index_word = dict((i, word) for i, word in enumerate(words_list))
        return v_count, word_index, index_word
    
    def generate_embedding(self, corpus):
        self.v_count, self.word_index, self.index_word = self.__generate_indexes(corpus)

        keys = list(self.word_index.keys())
        oneHotEncoded = pd.get_dummies(keys)

        X, y = [], []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context += [oneHotEncoded[sentence[j]].to_numpy()]
                if len(w_context) > 0:
                    X += [oneHotEncoded[word].to_numpy()]
                    y += [w_context]
        return np.asarray(X).astype(np.float), y
    
    def __forward(self, bx):
        h = layer(bx, self.w1, self.b1)
        u = layer(h, self.w2, self.b2)
        return h, u, softmax(u, axis=0)
    
    def __backprop(self, dZ2, h, u, bx):
        return backprop(dZ2, h, u, bx.astype(np.float), self.w2)

    def __loss(self, u, bx, by):
        if self.model == Word2VecModel.SKIP_GRAM:
            k = [u[label == 1] for label in by]
            k = k if len(k) > 0 else [0]
            return -np.sum(k) + len(by) * np.log(np.sum(np.exp(u + 1e-10)))
        elif self.model == Word2VecModel.CBOW:
            return -float(u[bx == 1]) + np.log(np.sum(np.exp(u + 1e-10)))
    
    def __producer(self, m):
        for i in range(m):
            yield self.X[i], np.asarray(self.y[i])
    
    def train(self, X, y, epochs=150, show_iter_err=50):
        m, n = X.shape
        self.X = X
        self.y = y
        
        self.w1 = np.random.randn(n, self.lt) * np.sqrt(2 / (n + self.lt))
        self.w2 = np.random.randn(self.lt, n) * np.sqrt(2 / (self.lt + n))
        self.b1 = np.zeros((self.lt, ))
        self.b2 = np.zeros((n, ))
        
        _loss = []
        start = time.time()
        for epoch in range(epochs):
            loss = 0
            weights = []
            for bx, by in self.__producer(m):
                if self.model == Word2VecModel.SKIP_GRAM:
                    # Forward
                    h, u, yh = self.__forward(bx)
                    # Backpropagation
                    dZ2 = np.sum(gradient(m, yh, by), axis=0)
                    # Update weights latter
                    weights += [self.__backprop(dZ2, h, u, bx)]
                elif self.model == Word2VecModel.CBOW:
                    # Forward
                    h, u, yh = self.__forward(np.mean(by, axis=0))
                    # Backpropagation
                    dZ2 = gradient(m, yh, bx)
                    # Update weights latter
                    weights += [self.__backprop(dZ2, h, u, by)]
                # Cost / Loss function
                loss += self.__loss(u, bx, by)
            # Update weights
            for weight in weights:
                dW1, dB1, dW2, dB2 = weight
                self.w1 -= self.lr * dW1
                self.b1 -= self.lr * dB1
                self.w2 -= self.lr * dW2
                self.b2 -= self.lr * dB2
            _loss += [(1/m) * loss]
            if epoch % show_iter_err == 0 and epoch > 1:
                print(f"Epoch {epoch + 1}/{epochs}, Time: {round(time.time()-start, 3)} ====> Loss: {np.round(_loss[-1], 5)}")
                start = time.time()
        print(f"Epoch {epochs}/{epochs} ====> Loss: {np.round(_loss[-1], 5)}")
        
        if self.norm:
            self.wv = MinMaxScaler().fit_transform(self.w1)
        else:
            self.wv = np.copy(self.w1)
        return _loss

    def find_similar(self, word, topn=10, metric="cosine"):
        word = self.word_index.get(word, None)
        if word is not None:
            vector = self.wv[word, :]
            if metric == "cosine":
                sim = cosine_similarity(self.wv, [vector])
            elif metric == "euclidian":
                sim = euclidean_distances(self.wv, [vector], squared=True)
            else:
                raise Exception("Not an acceptable metric... Choose cosine or euclidian.")
            words = list(zip(self.index_word.values(), sim.tolist()))
            words = sorted(words, key=lambda x: x[1], reverse=True)
            return words[:topn]
        return None
    
    def find_word(self, vector, topn=2):
        sim = euclidean_distances(self.wv, [vector], squared=True)
        return sorted(list(zip(self.index_word.values(), sim.tolist())), key=lambda x: x[1], reverse=False)[:topn]

if __name__ == "__main__":
    corpus = """
king man.
queen woman.
man male.
woman female.
"""
    
    w2v = Word2Vec(latent_space=2)
    corpus = w2v.tokenizer(corpus)
    X, y = w2v.generate_embedding(corpus)
    loss = w2v.train(X, y, epochs=300)

    # Vector space
    wv = w2v.wv
    word_index = w2v.word_index
    index_word = w2v.index_word
    
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(1, 2, 1) #, projection='3d')
    plt.title(f"Last loss: {np.round(loss[-1], 2)}")
    plt.plot(loss)
    
    ax = plt.subplot(1, 2, 2) #, projection='3d')
    plt.title("Vector Space")
    plt.scatter(wv[:, 0], wv[:, 1], marker='o', color='C0')
    for i, txt in enumerate(w2v.word_index):
        plt.annotate(txt, (wv[i, 0], wv[i, 1]))

    c_wv = wv[word_index["man"], :] + wv[word_index["queen"], :] - wv[word_index["woman"], :]
    plt.scatter(c_wv[0], c_wv[1], marker='x', color='C1')
    c_wv = wv[word_index["woman"], :] - wv[word_index["female"], :]
    plt.scatter(c_wv[0], c_wv[1], marker='+', color='C2')
    plt.show()
    
    # Compare
    c_wv = wv[word_index["man"], :] + wv[word_index["queen"], :] - wv[word_index["woman"], :]
    print(w2v.find_word(c_wv))

    print()
    c_wv = wv[word_index["woman"], :] - wv[word_index["female"], :]
    print(w2v.find_word(c_wv))

    print()
    word = "king"
    print(f"Closest words to: {word}")
    print(w2v.find_similar(word, topn=2))