import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import pdb
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


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


def drew_seq(times, data_seq_batch, save_path, gray=False):
    data_seq = []
    for pic_batch in data_seq_batch:
        for pic in pic_batch:
            data_seq.append(pic)
    assert len(times) == len(data_seq)
    for idx, pic in enumerate(data_seq):
        name = times[idx]
        drew_output_pic(pic, name, save_path, gray)


def write_sents(times, data_seq_batch, target_sents, save_path, vocab, show_matric=True):
    data_seq_pred, data_seq_target, data_seq_target_bleu = [], [], []
    # pdb.set_trace()
    if not len(data_seq_batch) == len(target_sents):
        pdb.set_trace()

    for idx, txt_batch in enumerate(data_seq_batch):
        for txt in txt_batch:
            txt = list(txt)
            if 1 in txt:
                txt = txt[:txt.index(1)]
            data_seq_pred.append(txt)
        for target_sent in target_sents[idx]:
            target_sent = list(target_sent)
            if 1 in target_sent:
                target_sent = target_sent[:target_sent.index(1)]
            data_seq_target_bleu.append([target_sent])
            data_seq_target.append(target_sent)

    score = corpus_bleu(data_seq_target_bleu, data_seq_pred)
    print("Bleu Score = " + str(score))

    assert len(times) == len(data_seq_pred)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + 'test_output.txt', 'w')
    for idx, txt2 in enumerate(data_seq_pred):
        # for txt in txt2:
        sent_pred = [vocab.index_to_word(word) + ' ' for word in txt2]
        sent_target = [vocab.index_to_word(word) + ' ' for word in data_seq_target[idx]]
        f.writelines(sent_target)
        f.write('\n')
        if '<eos>' in sent_pred:
            sent_pred = sent_pred[:sent_pred.index('<eos>') + 1]
        f.writelines(sent_pred)
        f.write('\n\n')

    f.write("Bleu Score = " + str(score))
    f.close()

    if show_matric:
        try:
            # from nlgeval import compute_metrics
            # metrics_dict = compute_metrics(hypothesis=data_seq_pred,
            #                                references=data_seq_target_bleu)
            from nlgeval import NLGEval
            nlgeval = NLGEval()  # loads the models
            metrics_dict = nlgeval.compute_metrics(data_seq_target_bleu, data_seq_pred)
            print(metrics_dict)

        except:
            print('please install nlgeval first')


def write_sents_1(batch_id,sents_id, data_seq_batch, save_path, vocab,times,batch_size):
    data_seq_pred = []
    for idx, txt_batch in enumerate(data_seq_batch):
        txt = list(txt_batch)
        if 1 in txt:
            txt = txt[:txt.index(1)]
        data_seq_pred.append(txt)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + 'test_output.txt', 'a+')
    for idx, txt2 in enumerate(data_seq_pred):
        name = str(times[batch_id * batch_size + idx]) + '_sID' + str(sents_id)
        sent_pred = [vocab.index_to_word(word) + ' ' for word in txt2]
        f.write(name+':')
        f.writelines(sent_pred)
        f.write('\n')
    f.close()


def drew_seq_1(batch_id,sents_id, data_seq_batch, save_path,times,batch_size, gray=False):
    # assert len(times) == len(data_seq_batch)
    for idx, pic in enumerate(data_seq_batch):
        name = str(times[batch_id*batch_size+idx])+'_sID'+str(sents_id)
        drew_output_pic(pic, name, save_path, gray)


def write_process(times, data_seq_batch, save_path, vocab, batch_size):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for id_batch, batch_data in enumerate(data_seq_batch):
        for id_sents, [sents, images] in enumerate(batch_data):
            # time = []
            assert len(sents) == len(images)
            # for i in range(len(sents)):
            #     time.append(times[id_batch * batch_size + id_sents]+'_sID'+str(id_sents))
            # pdb.set_trace()
            write_sents_1(id_batch,id_sents, sents, save_path, vocab,times,batch_size)
            drew_seq_1(id_batch,id_sents, images, save_path,times,batch_size)
