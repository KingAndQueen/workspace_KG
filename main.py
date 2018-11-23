from sklearn import model_selection
import Model
import tensorflow as tf
import numpy as np
import pdb
import Data_Process
import Model
import random
from math import exp
tf.flags.DEFINE_float("learn_rate", 0.001, "Learning rate for SGD.")
# tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
# tf.flags.DEFINE_float("anneal_stop_epoch", 50, "Epoch number to end annealed lr schedule.")
# tf.flags.DEFINE_float("learning_rate_decay_factor", 0.5, 'if loss not decrease, multiple the lr with factor')
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 20, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")  # should consider the size of validation set
tf.flags.DEFINE_integer("head", 3, "head number of attention")
tf.flags.DEFINE_integer("epochs", 2000, "Number of epochs to train for.")
tf.flags.DEFINE_integer('check_epoch',20, 'evaluation times')
tf.flags.DEFINE_integer("layers", 3, "the num layers of RNN.")
tf.flags.DEFINE_integer("neurons", 100, "Embedding size for neural networks.")
tf.flags.DEFINE_string("data_dir", "data/", "Directory containing tasks")
tf.flags.DEFINE_integer('sentence_size', 20, 'length of word in a sentence')
tf.flags.DEFINE_integer('stop_limit', 5, 'number of evaluation loss is greater than train loss  ')
# tf.flags.DEFINE_integer("knowledge_memory", 6, "size of additional info from KB . at least above 6")
# tf.flags.DEFINE_integer("random_state", None, "Random state.")
#tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
#tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_string("model_type", "train", "whether to train or test model")
config = tf.flags.FLAGS

def train_model(sess, model, train_data, valid_data):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 1
    train_losses = []

    epoch=config.epochs
    print('training....')
    eval_losses_all = []
    while current_step <= epoch:
        #  print ('current_step:',current_step)
        for i in range(len(train_data)):
            z_noise = np.random.uniform(-1, 1, [config.batch_size, config.noise_dim])

            train_loss_, _ = model.step(sess, random.choice(train_data),z_noise,step_type='train')

        if current_step % config.check_epoch == 0:
            eval_losses = []
            train_losses.append(train_loss_)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training loss:', train_loss_)



            eval_loss, _ = model.step(sess, valid_data)
            print('evaluation loss:', eval_loss)

            # model.saver.save(sess, checkpoint_path, global_step=current_step)
            # if len(eval_losses_all) > 0 and eval_loss > eval_losses_all[-1]:
            #     print('decay learning rate....')
            #     sess.run(model.learning_rate_decay_op)
            #     model.saver.save(sess, checkpoint_path, global_step=current_step)
            if len(eval_losses_all) > config.stop_limit and eval_loss > sum(eval_losses_all[-1 * config.stop_limit:])/float(config.stop_limit):
                print('----End training for evaluation increase----')
                break
            eval_losses_all.append(eval_loss)
        current_step += 1
    print(' current step %d finished' % current_step)

def test_model(sess, model, test_data, vocab):
    test_loss = 0.0
    predicts = []

    for batch_id, data_test in enumerate(test_data):
        loss, predict = model.step(sess, data_test, step_type='test')
        test_loss += loss
        predicts.append(predict)

    test_loss=test_loss / len(test_data)
    print('test total loss:', test_loss)



def main(_):
    vocab = Data_Process.Vocab()
    input_data_txt, output_data_txt, input_data_pic, output_data_pic = Data_Process.get_input_output_data(config.data_dir, vocab, config.sentence_size)
    config.img_size_x=input_data_pic[0][0]
    config.img_size_y=input_data_pic[0][1]

    batches_data=Data_Process.vectorize(input_data_txt,output_data_txt,input_data_pic,output_data_pic,vocab,config.batch_size)
    print('data processed,vocab size:', vocab.vocab_size)
    train_data,valid_test_data=model_selection.train_test_split(batches_data,0.2)
    valid_data,test_data=model_selection.train_test_split(valid_test_data,0.5)

    sess = tf.Session()

    if config.model_type == 'train':
        print('establish the model...')
        model = Model.seq2seq(config, vocab)
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, train_data, valid_data)
        # config.model_type = 'test'
        test_model(sess, model, test_data, vocab)

    if config.model_type == 'test' :
        print('Test model.......')
        print('establish the model...')
        # config.batch_size = len(test_data)
        config.trained_emb = False
        model = Model.seq2seq(config, vocab)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(sess, model, test_data, vocab)

if __name__ == "__main__":
    tf.app.run()
