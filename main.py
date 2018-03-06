from sklearn import model_selection
import Model
import tensorflow as tf
import numpy as np
import pdb

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 50, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("learning_rate_decay_factor", 0.5, 'if loss not decrease, multiple the lr with factor')
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 200, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")  # should consider the size of validation set
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 2000, "Number of epochs to train for.")
tf.flags.DEFINE_integer("rnn_layers", 3, "the num layers of RNN.")
tf.flags.DEFINE_integer("neurons", 1000, "Embedding size for neural networks.")

tf.flags.DEFINE_integer("knowledge_memory", 6, "size of additional info from KB . at least above 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_string("process_type", "train", "whether to train or test model")


FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    tf.app.run()