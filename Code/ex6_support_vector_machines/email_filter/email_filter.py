# author : Lee
# date   : 2021/2/20 20:28

import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn import svm

# stop_words = set(stopwords.words('english'))


def pre_process(file_path, save_path):
    with open(file_path) as f:
        text = f.read()
    # Lower-casing
    text = text.lower()
    # Filter Stop-Words
    text = text.split()
    global stop_words
    text = [w for w in text if not w in stop_words]
    text = ' '.join(text)
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
    text = ' '.join(text)
    file = open(save_path, 'w')
    file.write(text)


def batch_pre_process(path):
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(filepath, filename)
            save_path = file_path.split('\\')
            save_path = save_path[1::1]
            save_path = r"./pre_process/" + '/'.join(save_path)
            dir_path, _ = os.path.split(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            pre_process(file_path, save_path)


# batch_pre_process(r'./lingspam_public')


def word_frequency_count(path):
    fre_dict = {}
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(filepath, filename)
            with open(file_path) as f:
                text = f.read()
            text = text.split()
            for key in text:
                fre_dict[key] = fre_dict.get(key, 0) + 1
    items = sorted(fre_dict.items(), key=lambda x: x[1], reverse=True)
    with open(r'word_frequency.txt', 'w') as f:
        for item in items:
            f.writelines(str(item[0]) + '\t' + str(item[1]) + '\n')


# word_frequency_count(r'./pre_process')


def vocab_table():
    with open("word_frequency.txt") as f:
        word_fre = f.readlines()
    file = open("vocab.txt", 'w')
    index = 0
    for wf in word_fre:
        temp = wf.split()
        if int(temp[1]) < 100:
            break
        file.write(str(index) + '\t' + temp[0] + '\n')
        index += 1


# vocab_table()
# vocab = {}
# with open(r'./vocab.txt') as f:
#     for line in f:
#         line = line.split()
#         vocab[line[1]] = int(line[0])


def email_feature(path, data_type="train"):
    """
    提取邮件特征，存储到 .csv 文件中，最后一列为邮件标签，1:spam
    :param path: 预处理过的邮件路径
    :param data_type: 特征集的名称:train,cross vilidation,test
    :return:
    """
    global vocab
    X = np.array([0] * len(vocab))
    y = np.array([0])
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(filepath, filename)
            with open(file_path) as f:
                text = f.read()
            text = text.split()
            feature = np.array([0] * len(vocab))
            for w in text:
                if w in vocab.keys():
                    feature[vocab[w]] = 1
            X = np.vstack((X, feature))
            if 'spmsg' in filename:
                y = np.vstack((y, np.array([1])))
            else:
                y = np.vstack((y, np.array([0])))
    X = np.delete(X, [0], axis=0)
    y = np.delete(y, [0], axis=0)
    np.savetxt(data_type + ".csv", np.hstack((X, y)), delimiter=",")


# email_feature(r'./pre_process')
# email_feature(r'./part10', 'test')


train_data = np.loadtxt(r'./train.csv', delimiter=',')
train_X = train_data[:, :-1]
train_y = train_data[:, -1]
test_data = np.loadtxt(r'./test.csv', delimiter=',')
test_X = test_data[:, :-1]
test_y = test_data[:, -1]
svc = svm.SVC()
svc.fit(train_X, train_y)
print("training accuracy:", svc.score(train_X, train_y))
print("test accuracy:", svc.score(test_X, test_y))
