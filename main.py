from sklearn import model_selection
import os
import tensorflow as tf
import numpy as np
import pdb
import Data_Process
import Model
import random
import Analysis

# from math import exp
tf.flags.DEFINE_float("learn_rate", 0.00001, "Learning rate for SGD.")
# tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
# tf.flags.DEFINE_float("anneal_stop_epoch", 50, "Epoch number to end annealed lr schedule.")
# tf.flags.DEFINE_float("learning_rate_decay_factor", 0.5, 'if loss not decrease, multiple the lr with factor')
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 4, "Batch size for training.")  # should consider the size of validation set
tf.flags.DEFINE_integer("head", 8, "head number of attention")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer('check_epoch',20, 'evaluation times')
tf.flags.DEFINE_integer("layers", 3, "the num layers of RNN.")
tf.flags.DEFINE_integer("recurrent_dim",64, "Embedding size for neural networks.")
tf.flags.DEFINE_string("data_dir", "data/friends/", "Directory containing tasks")
tf.flags.DEFINE_integer('sentence_size', 30, 'length of word in a sentence')
tf.flags.DEFINE_integer('stop_limit', 5, 'number of evaluation loss is greater than train loss  ')
tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_bool("is_training", False, "whether to train or test model")
tf.flags.DEFINE_integer('img_size_x',160,'generate pic size in X')
tf.flags.DEFINE_integer('img_size_y',320,'generate pic size in Y')
tf.flags.DEFINE_integer('noise_dim',64,'dim in noise')
tf.flags.DEFINE_integer('convolution_dim',256,'dim in the first layer pic decoder')
tf.flags.DEFINE_bool('gray',False,'picture is gray or not, placeholder also should be changed')
tf.flags.DEFINE_integer('num_identical',6,'number of encode transformers')
tf.flags.DEFINE_bool('qa_transpose',False,'whether to train model in AQ with QA training')
tf.flags.DEFINE_bool('pre_training',False,'whether to train model in AQ with QA training')
tf.flags.DEFINE_integer('pretrain_epochs',100,'epoch for pre-training')
config = tf.flags.FLAGS

def train_model(sess, model, train_data, valid_data):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 1
    train_losses = []

    epoch=config.epochs
    print('training....')
    checkpoint_path = os.path.join(config.checkpoint_path, 'visual_dialog.ckpt')
    train_summary_writer= tf.summary.FileWriter(config.summary_path, sess.graph)
    global_steps=0
    while current_step <= epoch:
        #  print ('current_step:',current_step)
        for i in range(len(train_data)):
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
            # pdb.set_trace()
            train_loss_, summary = model.steps(sess, random.choice(train_data),step_type='train',qa_transpose=config.qa_transpose)
            global_steps+=1
            # g_step = sess.run(model.global_step)
            if global_steps%len(train_data)==0:
                train_summary_writer.add_summary(summary,global_steps)
        if current_step % config.check_epoch == 0:
            eval_losses = 0
            train_losses.append(train_loss_)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training loss:', train_loss_)
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])

            for i in range(len(valid_data)):
                eval_loss, _ ,= model.steps(sess, random.choice(valid_data),step_type='train')
                eval_losses+=eval_loss

            print('evaluation loss:', eval_losses/len(valid_data))

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

def test_model(sess, model, test_data, vocab,times):
    # test_loss = 0.0
    print('begin testing...')
    encoding_pics,pred_txts,processing_data = [],[],[]
    z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
    for batch_id, data_test in enumerate(test_data):
        pred_txt,encoding_pic,word_defined_image = model.steps(sess, data_test,z_noise, step_type='test',img_affect_testing=2)
        # test_loss += loss
        # pred_pics.append(pred_pic)
        pred_txts.append(pred_txt)
        encoding_pics.append(encoding_pic)
        # processing_data.append(word_defined_image)
    Analysis.drew_seq(times,encoding_pics,'./result/',config.gray)
    Analysis.write_sents(times,pred_txts,'./result/',vocab,show_matric=False,test_data=test_data)
    # Analysis.write_process(times,processing_data,'./result/process/',vocab,batch_size=config.batch_size)
    # test_loss=test_loss / len(test_data)
    # print('test total loss:', test_loss)K
    print('test is finished!')

def pretrain_model(sess, model, train_data):
    current_step = 1
    train_losses = []

    epoch = config.pretrain_epochs
    print('pre-training....')
    # train_summary_writer = tf.summary.FileWriter(config.summary_path, sess.graph)
    # global_steps = 0
    while current_step <= epoch:
        #  print ('current_step:',current_step)
        for i in range(len(train_data)):
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
            # pdb.set_trace()
            train_loss_, _ = model.steps(sess, random.choice(train_data), step_type='train',
                                               qa_transpose=config.qa_transpose)
            # global_steps += 1
            # g_step = sess.run(model.global_step)
            # if global_steps % len(train_data) == 0:
            #     train_summary_writer.add_summary(summary, global_steps)
        if current_step % config.check_epoch == 0:
            train_losses.append(train_loss_)
            print('-------------------------------')
            print('pre-train current_step:', current_step)
            print('pre-training loss:', train_loss_)
            # z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])
        current_step += 1
    print(' pre-train current step %d finished' % current_step)


def main(_):
    vocab = Data_Process.Vocab()
    if config.pre_training:
        input_data_txt_p,output_data_txt_p, input_data_pic_p, output_data_pic_p,weights_p, = Data_Process.get_pretrain_data(config.data_dir,vocab,config.sentence_size,config.gray)
        prtrain_batches_data = Data_Process.vectorize_batch(input_data_txt_p, output_data_txt_p, input_data_pic_p, output_data_pic_p,
                                                    weights_p, config.batch_size)

    input_data_txt, output_data_txt, input_data_pic, output_data_pic,weights,times = Data_Process.get_input_output_data(config.data_dir, vocab, config.sentence_size,config.gray)
    # config.img_size_x =input_data_pic.values()[0].shape[0]
    # config.img_size_y=input_data_pic.values()[0].shape[1]
    # pdb.set_trace()
    print('total sentences:',len(input_data_txt))
    batches_data,ignore_number=Data_Process.vectorize_batch(input_data_txt,output_data_txt,input_data_pic,output_data_pic,weights,config.batch_size)
    print('data processed,vocab size:', vocab.vocab_size)

    # train_data,valid_test_data=model_selection.train_test_split(batches_data,test_size=0.2,shuffle=False)
    # valid_data, test_data = model_selection.train_test_split(valid_test_data, test_size=0.5, shuffle=False)
    train_data, valid_test_data =batches_data[:int(round(0.8*len(batches_data)))],batches_data[-int(round(0.2*len(batches_data))):]
    valid_data, test_data =valid_test_data[:int(round(0.5*len(valid_test_data)))],valid_test_data[-int(round(0.5*len(valid_test_data))):]

    # pdb.set_trace()
    sess = tf.Session()
    test_ignore_len=len(times)-len(batches_data)*config.batch_size
    if test_ignore_len>0:
        assert ignore_number==test_ignore_len
        times_test=times[-(len(test_data)*config.batch_size+test_ignore_len):-test_ignore_len]
    else:
        times_test=times[-len(test_data)*config.batch_size:]
    # pdb.set_trace()
    if config.is_training:
        print('establish the model...')
        model = Model.seq_pic2seq_pic(config, vocab)
        # pdb.set_trace()
        sess.run(tf.global_variables_initializer())
        if config.pre_training:
            pretrain_model(sess, model,prtrain_batches_data)
        train_model(sess, model, train_data, valid_data)
        # config.model_type = 'test'
        test_model(sess, model, test_data[:len(times_test)/config.batch_size], vocab,times_test) ###

    else:
        print('Test model.......')
        print('establish the model...')
        # config.batch_size = len(test_data)

        model = Model.seq_pic2seq_pic(config, vocab)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(sess, model,test_data[:len(times_test)/config.batch_size], vocab,times_test) ################## test_data

if __name__ == "__main__":
    tf.app.run()
