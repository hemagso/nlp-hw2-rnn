import os
import numpy as np
import torch


class DataPreprocessing(object):
    def __init__(self, path):
        self._load_data(path)
        self._process_vocab()
        self._process_tags()
        self.train_data_oh, self.train_tags_id = self._preprocess( self.train_data, self.train_tags)
        self.test_data_oh, self.test_tags_id = self._preprocess(self.test_data, self.test_tags)

    def train_iterator(self):
        return zip(self.train_data_oh, self.train_tags_id)

    def test_iterator(self):
        return zip(self.test_data_oh, self.test_tags_id)

    def _preprocess(self, sentences, tags):
        sentences_oh = [self.sent_to_onehot(sentence) for sentence in sentences]
        tags_id = [self.tag_to_id(tag_sequence) for tag_sequence in tags]
        return sentences_oh, tags_id

    def _process_vocab(self):
        self.vocab_set = list(set([word for sent in self.train_data for word in sent])) + ['UNK']
        self.vocab_size = len(self.vocab_set)
        self.vocab2id = {v: i for i, v in enumerate(self.vocab_set)}
        self.id2vocab = {i: v for i, v in enumerate(self.vocab_set)}
        print("Number of word types, including 'UNK': %d" % self.vocab_size)

    def _process_tags(self):
        self.tag_set = list(set([tag for tag_seq in self.train_tags for tag in tag_seq]))
        self.tag_size = len(self.tag_set)
        self.tag2id = {t: i for i, t in enumerate(self.tag_set)}
        self.id2tag = {i: t for i, t in enumerate(self.tag_set)}
        print("Number of tag types: %d" % self.tag_size)
        print('These are the tag types: ' + str(self.tag_set))

    def _load_data(self, path):
        self.train_data = DataPreprocessing.read_file(os.path.join(path, "train.dat"))
        self.train_tags = DataPreprocessing.read_file(os.path.join(path, "train.tag"))
        self.test_data = DataPreprocessing.read_file(os.path.join(path, "test.dat"))
        self.test_tags = DataPreprocessing.read_file(os.path.join(path, "test.tag"))

        print('Total amount of training samples: %d' % len(self.train_data))
        print('Total amount of testing samples: %d' % len(self.test_data))
        print('Average sentence length in training data: %f' % (np.mean([len(sent) for sent in self.train_data])))
        print('\nExample:')
        print('The first sentence is: ' + str(self.train_data[0]))
        print('Its corresponding name entity sequence is: ' + str(self.train_tags[0]))

    def tag_to_id(self, tag_seq):
        return torch.LongTensor([self.tag2id[x] for x in tag_seq])

    def word_to_id(self, sentence):
        return torch.LongTensor([self.vocab2id.get(x, self.vocab2id["UNK"]) for x in sentence]).view(-1, 1)

    def sent_to_onehot(self, sentence):
        one_hot = torch.zeros(len(sentence), self.vocab_size)
        one_hot.scatter_(1, self.word_to_id(sentence), 1)
        return one_hot

    @staticmethod
    def read_file(f_name):
        data = []
        with open(f_name, 'r') as f:
            for line in f:
                data.append(line.strip().split())
        return data


if __name__ == "__main__":
    dp = DataPreprocessing("../../data")
    print(dp.sent_to_onehot(['2', 'start', 'restaurants', 'with', 'inside', 'dining']))