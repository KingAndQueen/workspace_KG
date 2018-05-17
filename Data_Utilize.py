import os
import numpy as np
import nltk
import re
import pdb
unlegal='[^A-Za-z0-9\ \']'
NAME_MAP_ID = {'Chandler': 0, 'Joey': 1, 'Monica': 2, 'Phoebe': 3, 'Rachel': 4, 'Ross': 5, 'others': 6, 'pad': 7}
class Vocab():
    def __init__(self, word2vec=None, embed_size=0):
        self.word2idx = {'<eos>': 0, '<go>': 1, '<pad>': 2, '<unk>': 3}
        self.idx2word = {0: '<eos>', 1: '<go>', 2: '<pad>', 3: '<unk>'}
        self.embed_size = embed_size

    def add_vocab(self, words):
        if isinstance(words, (list, np.ndarray)):
            for word in words:
                if word not in self.word2idx:
                    index = len(self.word2idx)
                    self.word2idx[word] = index
                    self.idx2word[index] = word
        else:
            if words not in self.word2idx:
                # print('adding new word',words)
                index = len(self.word2idx)
                self.word2idx[words] = index
                self.idx2word[index] = words

    def word_to_index(self, word):

        self.add_vocab(word)
        return self.word2idx[word]
        # for rl
        # if word in self.word2idx:
        #     return self.word2idx[word]
        # else:
        #     return self.word2idx['<unk>']

    def index_to_word(self, index):
        if index in self.idx2word:
            return self.idx2word[index]
        else:
            return '<unk>'

    @property
    def vocab_size(self):
        return len(self.idx2word)

def get_data(data_path, vocabulary, sentence_size):
    train_data_path = os.path.join(data_path, 'train.txt')
    test_data_path = os.path.join(data_path, 'test.txt')
    valid_data_path = os.path.join(data_path, 'validation.txt')
    return read_file(train_data_path, vocabulary, sentence_size), \
           read_file(valid_data_path, vocabulary, sentence_size), \
           read_file(test_data_path, vocabulary, sentence_size)

def read_file(data_path, vocabulary, sentence_size):

    f = open(data_path, 'r')
    # f = open(data_path, 'r', encoding='utf-8', errors='surrogateescap e')
    # f = codecs.open(data_path, 'r', 'utf-8')
    scene = {}
    scenes = []
    answer = ''
    questioner=''
    for lines in f:
        # lines = lines.strip()[2:-5]
        if len(lines) > 2:
            name = lines[:lines.index(':')]
            if name not in ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']:
                name = 'others'
            sentence = lines[lines.index(':') + 1:]
            sentence = re.sub(unlegal, ' ', sentence)
            sentence = sentence.lower()
            sentence=nltk.word_tokenize(sentence)
            # sentence = sentence.split()
            sentence_id = [vocabulary.word_to_index(word) for word in sentence]
            sentence_id = sentence_id[:sentence_size - 1]
            sentence_id.append(vocabulary.word_to_index('<eos>'))
            padding_len = max(sentence_size - len(sentence_id), 0)
            for i in range(padding_len):
                sentence_id.append(vocabulary.word_to_index('<pad>'))
            scene[name] = sentence_id
            questioner=answer
            answer = name

        else:

            if questioner == '':
                continue
            ans = scene[answer]
            ans.pop()  # pop out the last word IN answer
            ans.insert(0, vocabulary.word_to_index('<go>'))  # padding <go>
            scene['answer'] = ans
            scene['question']= scene[questioner]
            weight = []
            name_list = []
            for id in scene[answer]:
                if id == vocabulary.word_to_index('<pad>'):
                    weight.append(0.0)
                else:
                    weight.append(1.0)
            scene['weight'] = weight
            scene['speaker']=NAME_MAP_ID[answer]
            scenes.append(scene)
            scene = {}

    f.close()
    return scenes