import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import pdb
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from cider_scorer import CiderScorer

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
    rouge = Rouge()
    cider = Cider()
    #rouge_socre,arr = rouge.compute_score(data_seq_target,data_seq_pred)
    print("Bleu Score = " + str(score))

    assert len(times) == len(data_seq_pred)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + 'test_output.txt', 'w')
    rouge_res = []
    cider_res = []
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
        rouge_score = rouge.calc_score(sent_target,sent_pred)
        rouge_res.append(rouge_score)
        cider_score,_ = cider.compute_score(sent_target,sent_pred)
        cider_res.append(cider_score)

    avg_rouge_res = np.mean(np.array(rouge_res))
    avg_cider_res = np.mean(np.array(cider_res))
    print("Rouge Score = "+str(avg_rouge_res))
    print("cider Score= "+str(avg_cider_res))
    f.write("Bleu Score = " + str(score)+"\n")
    f.write("Rouge Score = "+str(avg_rouge_res))
    f.write("Bleu Score = " + str(score)+"\n")
    f.write("cider Score= "+str(avg_cider_res))
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
def write_sents_viDial(data_seq_batch, target_sents, save_path, vocab, show_matric=False):
    data_seq_pred, data_seq_target, data_seq_target_bleu = [], [], []
    # pdb.set_trace()
    if not len(data_seq_batch) == len(target_sents):
        pdb.set_trace()

    for idx, txt_batch in enumerate(data_seq_batch):
        for txt in txt_batch:
            txt = list(txt)
            if 1 in txt:
                txt = txt[:txt.index(2)]
            data_seq_pred.append(txt)
        for target_sent in target_sents[idx]:
            target_sent = list(target_sent)
            if 1 in target_sent:
                target_sent = target_sent[:target_sent.index(2)]
            data_seq_target_bleu.append([target_sent])
            data_seq_target.append(target_sent)

    score = corpus_bleu(data_seq_target_bleu, data_seq_pred)
    rouge = Rouge()
    cider = Cider()
    #rouge_socre,arr = rouge.compute_score(data_seq_target,data_seq_pred)
    print("Bleu Score = " + str(score))
    pdb.set_trace()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + 'test_output.txt', 'w')
    rouge_res = []
    cider_res = []
    for idx, txt2 in enumerate(data_seq_pred):
        # for txt in txt2:
        sent_pred = [vocab.index2word(word) + ' ' for word in txt2]
        sent_target = [vocab.index2word(word) + ' ' for word in data_seq_target[idx]]
        f.writelines(sent_target)
        f.write('\n')
        if '<eos>' in sent_pred:
            sent_pred = sent_pred[:sent_pred.index('<eos>') + 1]
        f.writelines(sent_pred)
        f.write('\n\n')
        rouge_score = rouge.calc_score(sent_target,sent_pred)
        rouge_res.append(rouge_score)
        cider_score,_ = cider.compute_score(sent_target,sent_pred)
        cider_res.append(cider_score)

    avg_rouge_res = np.mean(np.array(rouge_res))
    avg_cider_res = np.mean(np.array(cider_res))
    print("Rouge Score = "+str(avg_rouge_res))
    print("cider Score= "+str(avg_cider_res))
    f.write("Bleu Score = " + str(score)+"\n")
    f.write("Rouge Score = "+str(avg_rouge_res))
    f.write("Bleu Score = " + str(score)+"\n")
    f.write("cider Score= "+str(avg_cider_res))
    f.close()


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


class Rouge:
    """
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    """
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        # assert(len(candidate) == 1)
        # assert(len(refs) > 0)
        prec = []
        rec = []
        # split into tokens
        if len(candidate)>0 and len(refs)>0:

            token_c = candidate

            #for reference in refs:
                # split into tokens
            token_r = refs
                # compute the longest common subsequence
            lcs = self.my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max != 0:
                score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
            else:
                score = 0.0

        else:
            score = 0.0

        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param gts: dict : ground_truth
        :param res: dict : results of predict
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        score = []
        # score = self.calc_score(gts, res)
        # for idx in sorted(gts.keys()):
        for idx in range(len(res)):
            hypo = res[idx]
            ref = gts[idx]
            score.append(self.calc_score(hypo, ref))

            # Sanity check
            # assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            # assert(type(ref) is list)
            # assert(len(ref) > 0)
        #
        average_score = np.mean(np.array(score))

        # convert to percentage
        return 100 * average_score, np.array(score)
        # return score

    def my_lcs(self,string, sub):
        """
        Calculates longest common subsequence for a pair of tokenized strings
        :param string : list of str : tokens from a string split using whitespace
        :param sub : list of str : shorter string, also split using whitespace
        :returns: length (list of int): length of the longest common subsequence between the two strings
        Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
        """
        if len(string) < len(sub):
            sub, string = string, sub

        lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

        return lengths[len(string)][len(sub)]


    @staticmethod
    def method():
        return "Rouge"



class Cider(object):
    """
    Main Class to compute the CIDEr metric
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  res: dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
        :param  gts: dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float): computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        # for idx in sorted(gts.keys()):
        #     hypo = res[idx]
        #     ref = gts[idx]

            # Sanity check.
            # assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            # assert(type(ref) is list)
            # assert(len(ref) > 0)
            # cider_scorer += (hypo[0], ref)
        hypo = res
        ref = gts
        cider_scorer += (hypo, ref)
        (score, scores) = cider_scorer.compute_score()

        return score, scores

    @staticmethod
    def method():
        return "CIDEr"