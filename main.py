from sklearn import model_selection
import Model
import tensorflow as tf
import numpy as np
import pdb
import Data_Utilize
import Model
import random
from math import exp
tf.flags.DEFINE_float("learn_rate", 0.0001, "Learning rate for SGD.")
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
# tf.flags.DEFINE_integer("knowledge_memory", 6, "size of additional info from KB . at least above 6")
# tf.flags.DEFINE_integer("random_state", None, "Random state.")
#tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
#tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_string("model_type", "train", "whether to train or test model")
config = tf.flags.FLAGS

def train_model(sess, model, train_data, valid_data):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 1
    data_input_train = model.get_batch(train_data)
    data_input_eval = model.get_batch(valid_data)
    train_losses = []

    epoch=config.epochs
    print('training....')
    eval_losses_all = []
    while current_step <= epoch:
        #  print ('current_step:',current_step)
        for i in range(len(data_input_train)):
            train_loss_, _ = model.step(sess, random.choice(data_input_train),step_type='train')

        if current_step % config.check_epoch == 0:
            eval_losses = []
            train_losses.append(train_loss_)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training loss:', train_loss_)


            for eval_data in data_input_eval:
                eval_loss_, _, summary_eval = model.step(sess, eval_data)
                eval_losses.append(eval_loss_)
            eval_loss=float(sum(eval_losses)) / len(eval_losses)

            print('evaluation loss:', eval_loss)
            if eval_loss<300 and train_loss_ <300:
                print('train perplex:',exp(train_loss_))
                print('evaluation perplex:',exp(eval_loss))

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
    data_input_test = model.get_batch(test_data)
    test_loss = 0.0
    predicts = []

    for batch_id, data_test in enumerate(data_input_test):
        loss, predict = model.step(sess, data_test, step_type='test')
        test_loss += loss
        predicts.append(predict)

    test_loss=test_loss / len(data_input_test)
    print('test total loss:', test_loss)
    if test_loss<300:
        print ('test perplex:',exp(test_loss))


def main(_):
    vocab = Data_Utilize.Vocab()
    train_data, valid_data, test_data = Data_Utilize.get_data(config.data_dir, vocab, config.sentence_size)
    # initiall model from new parameters or checkpoints
    print('data processed,vocab size:', vocab.vocab_size)
    sess = tf.Session()
    print(' train data set %d, valid data set %d, test data set %d' % (
          len(train_data), len(valid_data), len(test_data)))

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
