import os
import json
import random
import numpy as np


class Data:
    def __init__(self, pretrain_embedding=None):
        self.vocab_map = {'word2id': {'': 0, '<UNK>': 1}, 'id2word': {0: '', 1: '<UNK>'}}
        if os.path.isfile('data/vocab_map.json'):
            self.vocab_map = json.load(open('data/vocab_map.json'))
            self.vocab_map['id2word'] = {int(k): v for k, v in self.vocab_map['id2word'].items()}
        self.label_map = {'label2id': {'O': 0}, 'id2label': {0: 'O'}}
        if os.path.isfile('data/label_map.json'):
            self.label_map = json.load(open('data/label_map.json'))
            self.label_map['id2label'] = {int(k): v for k, v in self.label_map['id2label'].items()}
        self.char_map = {'char2id': {'': 0, '<UNK>': 1}, 'id2char': {0: '', 1: '<UNK>'}}
        if os.path.isfile('data/char_map.json'):
            self.char_map = json.load(open('data/char_map.json'))
            self.char_map['id2char'] = {int(k): v for k, v in self.char_map['id2char'].items()}

        self.train_data = self.load_data('data/train.tsv')
        self.dev_data = self.load_data('data/dev.tsv')
        self.test_data = self.load_data('data/test.tsv')

        self.vocab_num = len(self.vocab_map['word2id'])
        self.label_num = len(self.label_map['label2id'])
        self.char_num = len(self.char_map['char2id'])

        self.embedding = None
        if pretrain_embedding:
            filename = 'data/' + pretrain_embedding.split('/')[-1] + '.npy'
            if not os.path.isfile(filename):
                w = [[]] * self.vocab_num
                found = 0
                dim = 0
                with open(pretrain_embedding) as f:
                    for l in f:
                        l = l.strip().split(' ')
                        word = l[0]
                        if word in self.vocab_map['word2id']:
                            w[self.vocab_map['word2id'][word]] = [float(i) for i in l[1:]]
                            dim = len(w[self.vocab_map['word2id'][word]])
                            found += 1
                w[0] = [0] * dim
                for i in range(1, self.vocab_num):
                    if len(w[i]) == 0:
                        w[i] = np.random.uniform(-np.sqrt(3 / dim), np.sqrt(3 / dim), [dim])

                np.save(filename, np.array(w, dtype=np.float32))
                self.embedding = w

                print('Pretrained Embedding File: {0}'.format(pretrain_embedding))
                print('Pretrained Embedding Size: {0}'.format(dim))
                print('Found Embedding: {0} / {1}'.format(found, self.vocab_num))
            else:
                self.embedding = np.load(filename)

        json.dump(self.vocab_map, open('data/vocab_map.json', 'w'), ensure_ascii=False)
        json.dump(self.label_map, open('data/label_map.json', 'w'), ensure_ascii=False)
        json.dump(self.char_map, open('data/char_map.json', 'w'), ensure_ascii=False)

        self.trained_num = 0
        np.random.shuffle(self.train_data)

    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for l in f:
                label, txt = l[:-1].split('\t')
                label_split = label.split(' ')
                for i in label_split:
                    if i == '':
                        print(l)
                    if i not in self.label_map['label2id']:
                        self.label_map['label2id'][i] = len(self.label_map['label2id'])
                        self.label_map['id2label'][len(self.label_map['id2label'])] = i
                word_split = txt.split(' ')
                for i in word_split:
                    if i.lower() not in self.vocab_map['word2id']:
                        self.vocab_map['word2id'][i.lower()] = len(self.vocab_map['word2id'])
                        self.vocab_map['id2word'][len(self.vocab_map['id2word'])] = i.lower()
                    for j in i:
                        if j not in self.char_map['char2id']:
                            self.char_map['char2id'][j] = len(self.char_map['char2id'])
                            self.char_map['id2char'][len(self.char_map['id2char'])] = j
                data.append([label_split, word_split])
        return data

    def get_train_batch(self, batch_size):
        data = self.train_data[self.trained_num: self.trained_num + batch_size]
        self.trained_num += batch_size
        if self.trained_num > len(self.train_data):
            self.trained_num = 0
            np.random.shuffle(self.train_data)
        labels = [d[0] for d in data]
        words = [d[1] for d in data]
        return words, labels

    def get_dev_batches(self, batch_size):
        labels = []
        words = []
        i = 0
        while i < len(self.dev_data):
            labels.append([d[0] for d in self.dev_data[i: i + batch_size]])
            words.append([d[1] for d in self.dev_data[i: i + batch_size]])
            i += batch_size
        return words, labels

    def get_test_batches(self, batch_size):
        labels = []
        words = []
        i = 0
        while i < len(self.test_data):
            labels.append([d[0] for d in self.test_data[i: i + batch_size]])
            words.append([d[1] for d in self.test_data[i: i + batch_size]])
            i += batch_size
        return words, labels


if __name__ == '__main__':
    d = Data()
    data = d.get_test_batches(20)
    print(len(data[0]))
    print(len(data[0][0]))
    print(data[0][0][0])
