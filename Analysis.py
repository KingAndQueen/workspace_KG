
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb
import numpy as np

def drew_output_pic(data_patch,case_name,save_path):
    # pdb.set_trace()
    for i in range(data_patch.shape[0]):
        plt.figure(figsize=(160, 320))
        plt.imshow(data_patch[i])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+str(case_name)+'.png')
        plt.close('all')

def drew_seq(times,data_seq,save_path):
    assert len(times)==len(data_seq)
    for idx,pic in enumerate(data_seq):
        name=times[idx]
        drew_output_pic(pic,name,save_path)

def write_sents(times,data_seq,save_path,vocab):
    assert len(times)==len(data_seq)
    f=open(save_path+'test_output.txt')
    for idx,txt in enumerate(data_seq):
        sent=[vocab.idx2word(word) for word in txt]
        f.writelines(sent)
    f.close()
