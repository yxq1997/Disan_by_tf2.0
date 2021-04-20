import numpy as np
import tensorflow as tf


def from_txt_get_vocab(dir1='new_vocab.txt', encode='ISO-8859-1'):
    vocab = {}  # word2id
    with open(dir1, 'r', encoding=encode)as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split()
            vocab[line[0]] = int(line[1])
        f.close()
    return vocab

def to_index(words, word_to_index):
    """
    将一个由word组成的list转化为glove里对应的index
    """
    for i in range(len(words)):
        word = words[i]
        if word in word_to_index.keys():
            index = word_to_index[word]
        else:
            index = 0
        words[i] = index
    indexes = words
    return indexes


def process_txt(dir1, encode=None):
    words = []
    labels = []
    if encode is None:
        encode = 'utf-8'
    with open(dir1, 'r', encoding=encode) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().strip('\n').split('\t')
            line_word = line[0].lower().strip().split()
            words.append(line_word)
            label = int(line[1])
            labels.append(label)
        file.close()
        return words, labels


def get_data(dir1, word_to_index, encode=None):
    train_words, train_labels = process_txt(dir1, encode)
    for words in train_words:
        data = to_index(words, word_to_index)
        words = data
    return train_words, train_labels


def mask(pad_data):
    mask_row_numble = []
    for i in range(pad_data.shape[0]):  # i个样本
        count = 0
        for j in range(pad_data[i].shape[0]):
            if pad_data[i][j] != 0:
                count += 1
        mask_row_numble.append(count)
    return np.reshape(np.array(mask_row_numble), [-1, 1])


def process_mask(mask_length, max_len):
    pad_data_tile = np.zeros([mask_length.shape[0], max_len, max_len])
    for i in range(len(mask_length)):
        m = np.squeeze(mask_length[i])
        a = np.ones(shape=(m, m))
        pad_data_tile[i][:m, :m] = a
    return pad_data_tile

def change_train_data(x_train, x_dev, x_test, max_len):
    a = tf.range(max_len)
    b = tf.constant(a, shape=(1, max_len))
    c_train = tf.tile(b, [x_train.shape[0], 1]).numpy()
    c_dev = tf.tile(b, [x_dev.shape[0], 1]).numpy()
    c_test = tf.tile(b, [x_test.shape[0], 1]).numpy()
    mask_length_train = mask(pad_data=x_train)
    mask_length_dev = mask(pad_data=x_dev)
    mask_length_test = mask(pad_data=x_test)

    mask_length_train = process_mask(mask_length_train, max_len)
    mask_length_dev = process_mask(mask_length_dev, max_len)
    mask_length_test = process_mask(mask_length_test, max_len)

    x_train = [x_train, c_train, mask_length_train]
    x_dev = [x_dev, c_dev, mask_length_dev]
    x_test = [x_test, c_test, mask_length_test]
    return x_train, x_dev, x_test

def to_word(indexes, index_word):
    """
    translate a indexes'list to words by index_word

    input: indexes: 索引列表
           index_word: index_word形式的词表  来自于数据集的vocab
    output: words: 索引对应的words
    """
    indexes = indexes.tolist()
    for i in range(len(indexes)):
        index = indexes[i]
        word = index_word[index]
        indexes[i] = word
    words = indexes
    return words