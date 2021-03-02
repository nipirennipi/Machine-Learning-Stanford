# author : Lee
# date   : 2021/2/19 14:28

import re
import string
from nltk.stem.porter import PorterStemmer
import numpy as np
import scipy.io as sio
from sklearn import svm

vocab = {}
with open(r'./data/vocab.txt') as f:
    for line in f:
        line = line.split()
        vocab[line[1]] = int(line[0])


def pre_process(path):
    with open(path) as f:
        text = f.read()
    # Lower-casing
    text = text.lower()
    # Normalizing URLs
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "httpaddr", text)
    # Normalizing Email Addresses
    text = re.sub(r"[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+", "emailaddr", text)
    # Normalizing Numbers
    text = re.sub(r"\d+", "number", text)
    # Normalizing Dollars
    text = text.replace('$', 'dollar')
    # Removal of punctuation
    text = (text.encode().translate(None, string.punctuation.encode())).decode()
    # Word Stemming
    text = text.split()
    stemmer = PorterStemmer()
    text = [stemmer.stem(w) for w in text]
    # Removal of non-words & map
    global vocab
    word_indices = []
    for w in text:
        if w in vocab.keys():
            word_indices.append(vocab[w])
    return word_indices


def email_feature(word_indices):
    global vocab
    feature = np.array([0] * len(vocab))
    for i in word_indices:
        feature[i] = 1
    return feature


# word_indices = pre_process(r'./data/emailSample1.txt')
# print(word_indices)
# feature = email_feature(word_indices)
# print(feature)
# print(len(feature))
# print(np.sum(feature, axis=0))

train_data = sio.loadmat(r'./data/spamTrain.mat')
test_data = sio.loadmat(r'./data/spamTest.mat')
svc = svm.SVC()
svc.fit(train_data['X'], train_data['y'].ravel())
print("training accuracy:", svc.score(train_data['X'], train_data['y'].ravel()))
print("test accuracy:", svc.score(test_data['Xtest'], test_data['ytest'].ravel()))
