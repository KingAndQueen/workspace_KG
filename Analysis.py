import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import pdb
import numpy as np


def run_tm(func):
    def inner(*N):
        start = time.time()
        res = func(*N)
        end = time.time()
        tm = end - start
        # print('time : %f' % tm)
        return res, tm

    return inner


def drew_output_pic(data_patch, case_name, save_path, gray=False):
    # pdb.set_trace()
    # for i in range(data_patch.shape[0]):
    plt.figure(figsize=(16, 32))
    if gray:
        plt.imshow(np.squeeze(data_patch), cmap='gray')
    else:
        plt.imshow(data_patch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + str(case_name) + '.png')
    plt.close('all')


def drew_seq(times, data_seq_batch, save_path,gray=False):
    data_seq = []
    for pic_batch in data_seq_batch:
        for pic in pic_batch:
            data_seq.append(pic)
    assert len(times) == len(data_seq)
    for idx, pic in enumerate(data_seq):
        name = times[idx]
        drew_output_pic(pic, name, save_path,gray)


def write_sents(times, data_seq_batch, save_path, vocab):
    data_seq = []
    # pdb.set_trace()
    for txt_batch in data_seq_batch:
        for txt in txt_batch:
            data_seq.append(txt)

    assert len(times) == len(data_seq)
    f = open(save_path + 'test_output.txt', 'w')
    for idx, txt2 in enumerate(data_seq):
        # for txt in txt2:
        sent = [vocab.index_to_word(word) + ' ' for word in txt2]
        f.writelines(sent)
        f.write('\n')
    f.close()

