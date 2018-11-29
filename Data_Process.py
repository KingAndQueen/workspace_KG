#encoding=utf8
import os, sys
import codecs
# from PIL import Image

import pdb
# open a pipe from a command
# import tensorflow as tf
# import time
import numpy as np
from datetime import datetime
# unlegal='[^A-Za-z\ \']'
import pickle
import copy
from PIL import Image
def pic_video(file_path, time_ss):
    if not os.path.exists(file_path):
        print ('data_dir is not exist!')
        return None
    a, b, c = os.popen3("ffmpeg -i " + file_path)
    # pdb.set_trace()
    out = c.read()
    dp = out.index("Duration: ")
    duration = out[dp + 10:dp + out[dp:].index(",")]
    hh, mm, ss = map(float, duration.split(":"))
    # total time ss
    total = (hh * 60 + mm) * 60 + ss
    season=os.path.basename(file_path)
    if time_ss < total:
        t = time_ss

        # t is seconds in the video
        os.system("ffmpeg -i " + file_path + " -y -f mjpeg -s 350x240 -ss %s -t 1 %s_frame_%i.jpg" % (t, season, t))
    return True


def read_srt(data_dir):
    # data_dir = data_dir + '/srt/'
    if not os.path.exists(data_dir):
        print ('data_dir is not exist!')
        return None
    # filelist = []
    # for root, dirs, files in os.walk(data_dir):
    #     for name in files:
    #         file_name = os.path.splitext(os.path.join(root, name))
    #         if file_name[1] == '.json':
    #             filelist.append(os.path.join(root, name))

    # files = os.listdir(data_dir)
    # files = [os.path.join(data_dir, f) for f in files]
    # s = 'Friends.{}'.format(season)
    # one_file = [f for f in files if s in f and 'train' in f][0]
    # for file_ in train_file:

    # pdb.set_trace()
    f = codecs.open(data_dir,'r')
    lines = f.readlines()
    f.close()
    times,sentences=[],[]
    TIME_FORMAT='%H:%M:%S,%f'
    # pdb.set_trace()
    # start=False
    for no,line in enumerate(lines):
        line=line.strip()
        # if line.find('http://bbs.btbbt.com')>0:
        #     start=True
        if len(line)>0 :
            if ' --> 'in line:
                if '（' in lines[no+1] or '<i>♪' in lines[no+1]:
                    # pdb.set_trace()
                    continue
                time_begin=line[:line.index(' --> ')]
                time_end=line[line.index(' --> ')+5:]
                # pdb.set_trace()
                time_begin=datetime.strptime(time_begin,TIME_FORMAT)
                time_end = datetime.strptime(time_end, TIME_FORMAT)
                relate_time=time_end - time_begin
                if relate_time.seconds>1:
                    time_=time_begin+(time_end-time_begin)/2
                else:
                    time_=time_end
                times.append(time_)
                sent=lines[no+1]
                sent=sent.replace('-','')
                sent =sent.strip()
                if ':' in sent:
                    if len(sent)-sent.index(':')<=1:
                        sent=sent[:sent.index(':')]
                    else:
                        # pdb.set_trace()
                        sent=sent[sent.index(':')+1:]
                sent = sent.strip()
                sentences.append(sent)
    # pdb.set_trace()
    if len(times)==len(sentences):
        print('length of frames:',len(times))
        return times,sentences
    else:
        pdb.set_trace()
        print('process srt error!')
        return None

def map_time(times, sentences,video_file_path,output_path):
    TIME_FORMAT = '%H:%M:%S,%f'
    output=open(output_path,'w')
    for no, time_ in enumerate(times):
        time_orig = datetime.strptime('00:00:00,0', TIME_FORMAT)
        relate_time = time_ - time_orig
        time_s = relate_time.seconds
        output_time_sents = str(time_) + '\t'+str(time_s)+'\t'+ sentences[no]+'\n'
        output.write(output_time_sents)
    output.close()
    # pdb.set_trace()
    for no,time_ in enumerate(times):
        time_orig = datetime.strptime('00:00:00,0', TIME_FORMAT)
        relate_time = time_ - time_orig
        time_s=relate_time.seconds
        # pdb.set_trace()
        pic_video(video_file_path,time_s)


if __name__ == "__main__":
    video_file_path = './data/video/S01E01.mp4'
    # pic_video(video_file_path, 32)
    # pdb.set_trace()
    srt_file_path='./data/subtitle/Friends.S01E01.1994.BD.x264-10bit.720P.AAC.Mysilu.eng&chs.srt'
    times,sentences=read_srt(srt_file_path)
    # pdb.set_trace()
    map_time(times,sentences,video_file_path,'./data/frame/S01E01.txt')



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


def read_txt_file_1E(data_path, vocabulary, sentence_size):
    f = open(data_path, 'r')
    sents=f.readlines()
    f.close()
    sents_idx=[]
    weights=[]
    for sent in sents:
        sent_idx=[]
        weight=[]
        sent=sent.strip()
        sent=sent[:sentence_size-1]
        for word in sent:
            idx=vocabulary.word_to_index(word)
            weight.append(1)
            sent_idx.append(idx)
        sent_idx.append(vocabulary.word_to_index('<eos>'))
        weight.append(0)
        padding_len = max(sentence_size - len(sent_idx), 0)
        for i in range(padding_len):
            weight.append(0)
            sent_idx.append(vocabulary.word_to_index('<pad>'))

        sents_idx.append(copy.copy(sent_idx))
        weights.append(copy.copy(weight))
    return sents_idx,weights

def read_pic_file_1E(data_path):
    files_name = os.listdir(data_path)
    files = [os.path.join(data_path, f) for f in files_name if 'txt' not in f]
    sec_map_pic={}
    pic_output=[]
    for no,file in enumerate(files):
        # image_raw = tf.gfile.FastGFile(file, 'rb').read()
        # img = tf.image.decode_jpeg(image_raw,channels=3)
        try:
            img=np.asarray(Image.open(file))
        except:
            pdb.set_trace()
        # print(img.shape)
        # pdb.set_trace()
        file_name=os.path.basename(file)
        sec=int(file_name[file_name.rindex('_')+1:file_name.rindex('.')])
        sec_map_pic[sec]=img

    for key in sorted(sec_map_pic.keys()):
        pic_output.append(sec_map_pic[key])
    # pdb.set_trace()
    return pic_output,sorted(sec_map_pic.keys())


def get_input_output_data(data_path, vocabulary,sentence_size):
    files_name = os.listdir(data_path)
    data_path_txt=[file_name for file_name in files_name if 'txt' in file_name ]
    txt,weights=read_txt_file_1E(data_path+data_path_txt[0],vocabulary,sentence_size)
    pic,times=read_pic_file_1E(data_path)
    input_data_txt=copy.copy(txt)
    _=input_data_txt.pop()
    output_data_txt = copy.copy(txt)
    _=output_data_txt.pop(0)

    output_weights=copy.copy(weights)
    output_weights.pop(0)

    input_data_pic=copy.copy(pic)
    _=input_data_pic.pop()
    output_data_pic=copy.copy(pic)
    _=output_data_pic.pop(0)
    # pdb.set_trace()
    assert len(output_weights)==len(output_data_txt)
    assert len(output_data_pic)==len(input_data_pic)
    assert len(input_data_txt)==len(output_data_txt)

    return input_data_txt,output_data_txt,input_data_pic,output_data_pic, output_weights,times

def vectorize_batch(input_data_txt,output_data_txt,input_data_pic,output_data_pic,weights,batch_size):
    batches_data=[]
    for _ in range(0, len(input_data_txt), batch_size):
        input_batch_txt=input_data_txt[_:_+batch_size]
        output_batch_txt=output_data_txt[_:_+batch_size]
        input_batch_pic=input_data_pic[_:_+batch_size]
        output_batch_pic=output_data_pic[_:_+batch_size]
        weight_batch_txt=weights[_:_+batch_size]
        batches_data.append([input_batch_txt,output_batch_txt,input_batch_pic,output_batch_pic,weight_batch_txt])
    return batches_data

def store_vocab(vocab, data_path):
    data_path = data_path + 'vocab.pkl'
    f = open(data_path, 'wb')
    pickle.dump(vocab, f)
    f.close()


def get_vocab(data_path):
    data_path = data_path + 'vocab.pkl'
    if os.path.exists(data_path):
        f = open(data_path, 'rb')
        vocab = pickle.load(f)
        f.close()
    else:
        print('<<<<<<< vocab is not exist >>>>>>>')
        vocab = None
    return vocab