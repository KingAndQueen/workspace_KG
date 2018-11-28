
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def drew_output_pic(data_patch,case_name,save_path):
    plt.figure(figsize=(35, 15))
    plt.imshow(data_patch)
    if not os.path.exists('./imgs'):
        os.makedirs('./imgs')
    plt.savefig(save_path+case_name+'.png')
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
