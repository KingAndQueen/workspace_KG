from sklearn import model_selection
import os
import tensorflow as tf
import numpy as np
import pdb
import Data_Process
import Model
import random
import Analysis
from VGG import run_candidates
import pickle as pkl
from dataset import VisDialDataset
import scipy.io
import pickle as pkl
from tqdm import tqdm
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from math import exp
tf.flags.DEFINE_float("learn_rate", 0.001, "Learning rate for adam.")
# tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
# tf.flags.DEFINE_float("anneal_stop_epoch", 50, "Epoch number to end annealed lr schedule.")
# tf.flags.DEFINE_float("learning_rate_decay_factor", 0.5, 'if loss not decrease, multiple the lr with factor')
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 10 * 10, "Batch size for training.")  # should consider the size of validation set
tf.flags.DEFINE_integer("head", 8, "head number of attention")
tf.flags.DEFINE_integer("epochs", 500, "Number of epochs to train for.")
tf.flags.DEFINE_integer('check_epoch', 10, 'evaluation times')
tf.flags.DEFINE_integer("layers", 3, "the num layers of RNN.")
tf.flags.DEFINE_integer("recurrent_dim", 64, "Embedding size for neural networks.")
tf.flags.DEFINE_string("data_dir", "data/friends/", "Directory containing tasks")
tf.flags.DEFINE_integer('sentence_size', 30, 'length of word in a sentence')
tf.flags.DEFINE_integer('stop_limit', 5, 'number of evaluation loss is greater than train loss  ')
tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_bool("is_training", True, "whether to train or test model")
tf.flags.DEFINE_integer('img_feature_layer', 36, 'generate pic size in X')
tf.flags.DEFINE_integer('img_feature_vector', 2048, 'generate pic size in Y')
tf.flags.DEFINE_integer('noise_dim', 64, 'dim in noise')
tf.flags.DEFINE_integer('convolution_dim', 256, 'dim in the first layer pic decoder')
tf.flags.DEFINE_bool('gray', False, 'picture is gray or not, placeholder also should be changed')
tf.flags.DEFINE_integer('num_identical', 6, 'number of encode transformers')
tf.flags.DEFINE_bool('qa_transpose', False, 'whether to train model in AQ with QA training')
tf.flags.DEFINE_bool('pre_training', False, 'whether to train model in AQ with QA training')
tf.flags.DEFINE_integer('pretrain_epochs', 100, 'epoch for pre-training')
tf.flags.DEFINE_integer('round', 10, 'dialogue round in a image')
config = tf.flags.FLAGS


def clean_data(data_class, keys, name='valid_idx.pkl'):
    if not os.path.exists('data/vds/' + name):
        # pdb.set_trace()
        print('cleaning dataset ............')
        f = open('data/vds/' + name, 'wb')
        valid_ids, error_ids = [], []

        # id_image = random.choice(keys)
        for i in tqdm(range(len(keys))):
            id_image = keys[i]
            sample = data_class[id_image]
            if 'ans_in' in sample.keys() and "ans_out" in sample.keys() and "ques" in sample.keys() and 'img_feat' in sample.keys():
                valid_ids.append(id_image)
            else:
                print('error data_ids:', id_image)
                error_ids.append(id_image)
            del sample
        pkl.dump(valid_ids, f)
        print('total error ids:', error_ids)
    else:
        f = open('data/vds/' + name, 'rb')
        valid_ids = pkl.load(f)
    f.close()
    print('valid training data: ', len(valid_ids))
    return valid_ids


def get_batch_data(data_class, valid_ids, batch_size,real_batch_factor=10):
    if len(valid_ids)<int(batch_size/real_batch_factor):
        print('datset is not long enough for one batch')
        return None
    batch_size=int(batch_size/real_batch_factor)
    # batch_txt_ans_input, batch_txt_ans_output, batch_pic_input, batch_txt_query = [], [], [], []
    # data_batch = []
    # print('batch the data....')
    # tf_writer=tf.python_io.TFRecordWriter('./data/train_dataset_batch.tf')
    idxs=iter(range(len(data_class.image_ids)))
    # pdb.set_trace()
    try:
        while True:
            batch_txt_ans_input, batch_txt_ans_output, batch_txt_query, batch_pic_input = [], [], [], []
            for i in range(batch_size):
                sample=data_class[data_class.image_ids[idxs.__next__()]]
                if 'ans_in' in sample.keys() and "ans_out" in sample.keys() and "ques" in sample.keys() and 'img_feat' in sample.keys():
                    batch_txt_ans_input.extend(sample["ans_in"])
                    batch_txt_ans_output.extend(sample["ans_out"])
                    batch_txt_query.extend(sample["ques"])
                    batch_pic_input.extend(10 * [sample['img_feat']])
                    del sample
                else:
                    pdb.set_trace()
                    print('data structure is error, id :', data_class.image_ids[idxs])
            data_batch = [batch_txt_query,batch_txt_ans_input, batch_txt_ans_output,batch_pic_input]
            yield data_batch
    except StopIteration:
        return None

    # for idx in tqdm(range(int(len(valid_ids) / batch_size))):
    #     batch_txt_ans_input,batch_txt_ans_output,batch_txt_query,batch_pic_input=[],[],[],[]
    #     for i in range(batch_size):
    #         if idx * batch_size + i < len(valid_ids):
    #             ture_id = valid_ids[idx * batch_size + i]
    #             # print(ture_id)
    #             # pdb.set_trace()
    #             sample = data_class[ture_id]
    #             if 'ans_in' in sample.keys() and "ans_out" in sample.keys() and "ques" in sample.keys() and 'img_feat' in sample.keys():
    #                 batch_txt_ans_input.extend(sample["ans_in"])
    #                 batch_txt_ans_output.extend(sample["ans_out"])
    #                 batch_txt_query.extend(sample["ques"])
    #                 batch_pic_input.extend(10 * [sample['img_feat']])
    #                 del sample
    #             else:
    #                 pdb.set_trace()
    #                 print('data structure is error, id :', ture_id)
    #                 # batch_txt_ans_input.extend(10 * [config.sentence_size * [0]])
    #                 # batch_txt_ans_output.extend(10 * [config.sentence_size * [0]])
    #                 # batch_txt_query.extend(10 * [config.sentence_size * [0]])
    #                 # batch_pic_input.extend(10 * [np.zeros((config.img_feature_layer, config.img_feature_vector))])
    #         else:
    #             break
    #     # data_batch.append([copy.copy(batch_txt_query), copy.copy(batch_txt_ans_input), copy.copy(batch_txt_ans_output), copy.copy(batch_pic_input)])
    #     # pdb.set_trace()
    #     # example=tf.train.Example()
    #     # feature=example.features.feature
    #     # feature['batch_txt_query'].int64_list.value.append(batch_txt_query)
    #     # feature['batch_txt_ans_input'].int64_list.value.append(batch_txt_ans_input)
    #     # feature['batch_txt_ans_output'].int64_list.value.append(batch_txt_ans_output)
    #     # feature['batch_pic_input'].float_list.value.append(batch_pic_input) #100*36*2048 float
    #     # tf_writer.write(example.SerializeToString())
    # # tf_writer.close()

    # return data_batch


def train_model(sess, model, train_data, valid_data):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 1
    train_losses = []

    epoch = config.epochs
    print('training....')
    checkpoint_path = os.path.join(config.checkpoint_path, 'visual_dialog.ckpt')
    train_summary_writer = tf.summary.FileWriter(config.summary_path, sess.graph)
    global_steps = 0

    # keys_train=clean_data(train_data,keys_train,name='train_idx.pkl')
    # keys_valid=valid_data.image_ids
    # keys_valid = clean_data(valid_data, keys_valid,name='valid_idx.pkl')
    iter_copy_train=copy.deepcopy(train_data)

    iter_copy_valid = copy.deepcopy(valid_data)
    while current_step <= epoch:
        # print('current_step:', current_step)
        for train_data_batch in train_data:
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
            # pdb.set_trace()
            # train_data_batch_id = []
            # pdb.set_trace()

            if train_data_batch is None:
                train_summary_writer.add_summary(summary, global_steps)
                train_data=iter_copy_train
                break
            train_loss_, summary = model.steps(sess, train_data_batch, step_type='train',
                                               qa_transpose=config.qa_transpose)
            global_steps += 1
            print('training step', global_steps)

            # g_step = sess.run(model.global_step)
            # if global_steps % len(train_data) == 0:
            #     train_summary_writer.add_summary(summary, global_steps)


        if current_step % config.check_epoch == 0:
            eval_losses = 0
            train_losses.append(train_loss_)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training loss:', train_loss_)
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])

            for in_valid in valid_data:
                if in_valid is None:
                    valid_data=iter_copy_valid
                    break
                eval_loss, _, = model.steps(sess, in_valid, step_type='test')
                eval_losses += eval_loss

            print('evaluation ap:', eval_losses / len(valid_data))

            model.saver.save(sess, checkpoint_path, global_step=current_step)
            # if len(eval_losses_all) > 0 and eval_loss > eval_losses_all[-1]:
            #     print('decay learning rate....')
            #     sess.run(model.learning_rate_decay_op)
            #     model.saver.save(sess, checkpoint_path, global_step=current_step)
            # if len(eval_losses_all) > config.stop_limit and eval_loss > sum(eval_losses_all[-1 * config.stop_limit:])/float(config.stop_limit):
            #     print('----End training for evaluation increase----')
            #     break

        current_step += 1
    print(' current step %d finished' % current_step)


def test_model(sess, model, test_data, vocab):
    test_loss = 0.0
    print('begin testing...')
    encoding_pics, pred_txts, target_txt, processing_data, acc_pics = [], [], [], [], []
    # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
    # current_step = 0
    for batch_id in tqdm(range(len(test_data))):
        data_test=test_data[batch_id]
    # for batch_id, data_test in enumerate(test_data):
        # pred_txt, encoding_pic, word_defined_image, acc_img = model.steps(sess, data_test,
        #                                                                   step_type='test')
        # pred_txt, encoding_pic, word_defined_image, acc_img = model.steps(sess, data_test,
        #                                                                   step_type='test')
        loss,pred_txt=model.steps(sess, data_test, step_type='test')
        test_loss += loss

        # pred_pics.append(pred_pic)
        pred_txts.append(pred_txt)
        target_txt.append(data_test[2])
        # encoding_pics.append(encoding_pic)
        # processing_data.append(word_defined_image)
        # acc_pics.append(acc_img)
        # if current_step % 1 == 0:
        #     print("current_step", current_step)
        #     print("pred_txt", pred_txt)
        #     print("target_txt", data_test[1])
        # current_step += 1
    # print('img choosing accuracy:', np.mean(acc_pics))
    # Analysis.drew_seq(times, encoding_pics, './result/', config.gray)
    Analysis.write_sents_viDial(pred_txts, target_txt, './result/', vocab, show_matric=False)
    # Analysis.write_process(times,processing_data,'./result/process/',vocab,batch_size=config.batch_size)
    test_loss=test_loss / len(test_data)
    print('test perplexity:', np.exp(test_loss))
    print('test is finished!')
    return None

# def pretrain_model(sess, model, train_data):
#     current_step = 1
#     train_losses = []
#
#     epoch = config.pretrain_epochs
#     print('pre-training....')
#     # train_summary_writer = tf.summary.FileWriter(config.summary_path, sess.graph)
#     # global_steps = 0
#     while current_step <= epoch:
#         #  print ('current_step:',current_step)
#         for i in range(len(train_data)):
#             # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
#             # pdb.set_trace()
#             train_loss_, _ = model.steps(sess, random.choice(train_data), step_type='train',
#                                          qa_transpose=config.qa_transpose)
#             # global_steps += 1
#             # g_step = sess.run(model.global_step)
#             # if global_steps % len(train_data) == 0:
#             #     train_summary_writer.add_summary(summary, global_steps)
#         if current_step % config.check_epoch == 0:
#             train_losses.append(train_loss_)
#             print('-------------------------------')
#             print('pre-train current_step:', current_step)
#             print('pre-training loss:', train_loss_)
#             # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
#         current_step += 1
#     print(' pre-train current step %d finished' % current_step)


def main(_):
    data_config = {
        'image_features_train_h5': 'data/vds/features_faster_rcnn_x101_train.h5',
        'image_features_val_h5': 'data/vds/features_faster_rcnn_x101_val.h5',
        'image_features_test_h5': 'data/vds/features_faster_rcnn_x101_test.h5',
        'word_counts_json': 'data/vds/visdial_1.0_word_counts_train.json',

        'img_norm': 1,
        'concat_history': True,
        'max_sequence_length': config.sentence_size,
        'vocab_min_count': 5}
    sess = tf.Session()
    if config.is_training:
        train_dataset = VisDialDataset(data_config, 'data/vds/visdial_1.0_train.json',
                                       dense_annotations_jsonpath=None, overfit=False, in_memory=True,
                                       return_options=True, add_boundary_toks=True)
        # print('original train dataset length :', len(train_dataset.dialogs_reader))
        valid_dataset = VisDialDataset(data_config, 'data/vds/visdial_1.0_val.json', 'data/vds/visdial_1.0_val_dense_annotations.json', False, True, True, True)
        # print('original valid dataset length :', len(valid_dataset.dialogs_reader))
        keys_train = train_dataset.image_ids
        print('training dataset length :',len(keys_train))
        train_dataset_batch = get_batch_data(train_dataset, keys_train, config.batch_size)
        # pdb.set_trace()
        # if os.path.exists('./data/train_dataset_batch.pkl'):
        #     with open('./data/train_dataset_batch.pkl','r') as f:
        #         train_dataset_batch =pkl.load(f)
        # else:
        #     train_dataset_batch = get_batch_data(train_dataset, keys_train, config.batch_size)
        #     with open('./data/train_dataset_batch.pkl', 'w') as f:
        #         pkl.dump(train_dataset_batch,f)
        keys_valid=valid_dataset.image_ids
        print('valid dataset length :', len(keys_valid))
        valid_dataset_batch=get_batch_data(valid_dataset,keys_valid,config.batch_size)
        print('establish the model...')
        model = Model.seq_pic2seq_pic(config, train_dataset.vocabulary)
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, train_dataset_batch, valid_dataset_batch)



    else:
        valid_dataset = VisDialDataset(data_config, 'data/vds/visdial_1.0_val.json',
                                       'data/vds/visdial_1.0_val_dense_annotations.json', False, True, True, True)
        keys_valid = valid_dataset.image_ids
        print('valid dataset length :', len(keys_valid))
        valid_dataset_batch = get_batch_data(valid_dataset, keys_valid, config.batch_size)
        model = Model.seq_pic2seq_pic(config, valid_dataset.vocabulary)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print('vocabulary len:', len(valid_dataset.vocabulary))
        test_model(sess, model, valid_dataset_batch, valid_dataset.vocabulary)
        # pdb.set_trace()

        # test dataset has no target to count result currently
        # test_dataset = VisDialDataset(data_config, 'data/vds/visdial_1.0_test.json', None, False, True, True, False)
        # keys_test=test_dataset.image_ids
        # test_dataset_batch = get_batch_data(test_dataset, keys_test, config.batch_size)
        # print('Test model.......')
        # print('establish the model...')

        # model = Model.seq_pic2seq_pic(config, test_dataset.vocabulary)
        # print('Reload model from checkpoints.....')
        # ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
        # model.saver.restore(sess, ckpt.model_checkpoint_path)
        # print('vocabulary len:',len(test_dataset.vocabulary))
        # test_model(sess, model, test_dataset_batch, test_dataset.vocabulary)


if __name__ == "__main__":
    tf.app.run()