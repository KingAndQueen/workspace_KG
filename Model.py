# import tensorflow as tf
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import embedding_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import rnn
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.python.util import nest
# import pdb
import convolution
# from VGG import build_vgg19
from transformer import *


class seq_pic2seq_pic():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32, name='learn_rate')

        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        # self._opt = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._opt = tf.train.AdamOptimizer(learning_rate=self._learn_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)

        self._embedding_size = config.recurrent_dim
        self._layers = config.layers
        self.img_size_x = config.img_size_x
        self.img_size_y = config.img_size_y
        # self._gf_dim=config.gf_dim
        self._max_grad_norm = config.max_grad_norm
        self._cov_size = config.convolution_dim
        self._noise_dim = config.noise_dim
        self.model_type = config.model_type
        self._num_identical = config.num_identical
        self._build_inputs()
        self._head = config.head
        self.g_bn0 = convolution.batch_norm(name='g_bn0')
        self.g_bn1 = convolution.batch_norm(name='g_bn1')
        self.g_bn2 = convolution.batch_norm(name='g_bn2')
        self.g_bn3 = convolution.batch_norm(name='g_bn3')
        if self.model_type == 'train':
            is_training = True
        else:
            is_training = False

        with tf.variable_scope("encode_txt"):
            self.enc = embedding(self._question,
                                 vocab_size=vocab.vocab_size,
                                 num_units=self._embedding_size,
                                 scale=True,
                                 scope="enc_embed")

            # Position Encoding(use range from 0 to len(inpt) to represent position dim of each words)
            # tf.tile(tf.expand_dims(tf.range(tf.shape(self.inpt)[1]), 0), [tf.shape(self.inpt)[0], 1]),
            self.enc += positional_encoding(self._question,
                                            vocab_size=self._sentence_size,
                                            num_units=self._embedding_size,
                                            zero_pad=False,
                                            scale=False,
                                            scope="enc_pe")

            # Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=0.1,
                                         training=tf.convert_to_tensor(is_training))

            # Identical
            for i in range(self._num_identical):
                with tf.variable_scope("num_identical_{}".format(i)):
                    # Multi-head Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._head,
                                                   dropout_rate=0.1,
                                                   is_training=is_training,
                                                   causality=False)

                    self.enc = feedforward(self.enc, num_units=[4 * self._embedding_size, self._embedding_size])

        # pdb.set_trace()
        def _encoding_pic_frame(frame, name=''):
            with tf.variable_scope('encoding_frame_' + name):
                # resident net
                resnet_output, end_points = convolution.resnet_v2_50(frame)
                # resnet_output=end_points['resnet_v2_50' + '/block4'] # test2 to check the full connections effect
                # encode_output=end_points['encoding_frame_'+name+'/resnet_v2_50/block4']
                # pdb.set_trace()
                w_pic = tf.get_variable('w', [1, 2048, 64], initializer=tf.random_normal_initializer(stddev=0.02))
                resnet_output = tf.squeeze(resnet_output, 1)
                encoding_pic_output = tf.nn.conv1d(resnet_output, w_pic, 1, 'SAME')
                encoding_pic_output = tf.tile(encoding_pic_output, [1, 30, 1])
                return encoding_pic_output

        # pdb.set_trace()
        encoder_pic_output = _encoding_pic_frame(self._input_pic, 'one')

        with tf.variable_scope('embed_decode_input'):

            decoder_input = tf.concat((tf.ones_like(self._response[:, :1]) * 2, self._response[:, :-1]), -1)
            dec = embedding(self._response,
                            vocab_size=vocab.vocab_size,
                            num_units=self._embedding_size,
                            scale=True,
                            scope="dec_embed")

            # Position Encoding(use range from 0 to len(inpt) to represent position dim)
            dec += positional_encoding(decoder_input,
                                       vocab_size=self._sentence_size,
                                       num_units=self._embedding_size,
                                       zero_pad=False,
                                       scale=False,
                                       scope="dec_pe")

        with tf.variable_scope('merge_txt_pic'):
            decoder_input = tf.concat((encoder_pic_output, dec), -1)
            w_merge = tf.get_variable('w', [1, 128, 64], initializer=tf.random_normal_initializer(stddev=0.02))
            # pdb.set_trace()
            self.dec =tf.nn.conv1d(decoder_input, w_merge, 1, 'SAME')


        with tf.variable_scope('decode_txt'):
            # Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=0.1,
                                         training=tf.convert_to_tensor(is_training))
            # Identical
            # pdb.set_trace()

            for i in range(self._num_identical):
                with tf.variable_scope("num_identical_{}".format(i)):
                    # Multi-head Attention(self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._head,
                                                   dropout_rate=0.1,
                                                   is_training=is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Multi-head Attention(vanilla-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._head,
                                                   dropout_rate=0.1,
                                                   is_training=is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    self.dec = feedforward(self.dec, num_units=[4 * self._embedding_size, self._embedding_size])

                    # def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
                    #     with tf.variable_scope(name):
                    #         # filter : [height, width, output_channels, in_channels]
                    #         w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                    #                             initializer=tf.random_normal_initializer(stddev=stddev))
                    #
                    #         deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    #                                         strides=[1, d_h, d_w, 1])
                    #
                    #         biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                    #         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
                    #
                    #         if with_w:
                    #             return deconv, w, biases
                    #         else:
                    #             return deconv


                    # with tf.variable_scope('decoder_pic'):
                    #     s = self.img_size_x
                    #     y = self.img_size_y
                    #     s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
                    #     y2, y4, y8, y16 = int(y / 2), int(y / 4), int(y / 8), int(y / 16)
                    #     # encoder_pic_output_reshape = tf.reshape(encoder_pic_output, [self._batch_size, -1])
                    #     # response_txt_reshape=tf.reshape(response_txt,[self._batch_size,-1])
                    #     # encoder_txt_output_reshape=tf.reshape(encoder_txt_output,[self._batch_size,-1])
                    #     # all_infor=tf.concat([response_txt_reshape,encoder_txt_output_reshape,encoder_pic_output_reshape],1)
                    #     # try more input method to replace all_infor # test1
                    #     # pdb.set_trace()
                    #     # reduced_text_embedding = lrelu(linear(all_infor, self._cov_size, 'g_embedding'))
                    #     # reduced_text_embedding = lrelu(all_infor)
                    #     # z_concat = tf.concat([self._random_z, reduced_text_embedding],1)
                    #     # z_ = linear(encoder_pic_output_reshape, self._cov_size * 8 * s16 * y16, 'g_h0_lin')
                    #     z_=encoder_pic_output
                    #     # pdb.set_trace()
                    #     h0 = tf.reshape(z_, [-1, s16, y16, self._cov_size * 8])
                    #     h0 = tf.nn.relu(self.g_bn0(h0, type=self.model_type))
                    #
                    #     h1 = deconv2d(h0, [self._batch_size, s8, y8, self._cov_size * 4], name='g_h1')
                    #     h1 = tf.nn.relu(self.g_bn1(h1, type=self.model_type))
                    #
                    #     h2 = deconv2d(h1, [self._batch_size, s4, y4, self._cov_size * 2], name='g_h2')
                    #     h2 = tf.nn.relu(self.g_bn2(h2, type=self.model_type))
                    #
                    #     h3 = deconv2d(h2, [self._batch_size, s2, y2, self._cov_size * 1], name='g_h3')
                    #     h3 = tf.nn.relu(self.g_bn3(h3, type=self.model_type))
                    #
                    #     h4 = deconv2d(h3, [self._batch_size, s, y, 1], name='g_h4')
                    #
                    #     predict_pic = tf.tanh(h4) / 2. + 0.5

                    # print '$$$$$$$$$$$$$$$$$$$$'
                    # print h4.get_shape().as_list()
                    # print '$$$$$$$$$$$$$$$$$$$$'

        # pdb.set_trace()
        # def compute_error(real, fake):
        #     return tf.reduce_mean(tf.abs(fake - real))
            # diversity loss

            # with tf.variable_scope('loss_function_pic'):
            # pdb.set_trace()
            # cov_input=convolution.deeplab_v3(predict_pic)
            # cov_output=convolution.deeplab_v3(self._output_pic)
            # sp=self.img_size_x

            # vgg_real = build_vgg19(self._real_pic)
            # vgg_fake = build_vgg19(predict_pic, reuse=True)
            # p0 = compute_error(vgg_real['input'], vgg_fake['input'])
            # p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 1.6
            # p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 2.3
            # p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 1.8
            # p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 2.8
            # p5 = compute_error(vgg_real['conv5_2'],
            #                    vgg_fake['conv5_2']) * 10 / 0.8  # weights lambda are collected at 100th epoch
            # content_loss = p0 + p1 + p2 + p3 + p4 + p5

            # pdb.set_trace()
            # G_loss = tf.reduce_sum(tf.reduce_min(content_loss)) * 0.999 + tf.reduce_sum(
            #     tf.reduce_mean(content_loss)) * 0.001

            # pic_loss=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self._real_pic,predict_pic),name='pic_loss'),[1,2,3]))
            # pic_loss=tf.reduce_mean(pic_loss,name='l2_mean_loss_pic')
            # self.predict_pic = predict_pic

        with tf.variable_scope('loss_function_txt'):
            # with tf.device('/device:GPU:1'):
            # Our targets are decoder inputs shifted by one.
            # _, labels = tf.split(self._response, [1, -1], 1)  # sign 'go' for train decoder
            # labels = tf.concat([labels, _], axis=1)  # remove 'go' to compute loss
            #
            # cross_entropy_sentence = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response_txt,
            #                                                                         labels=labels,
            #                                                                         name="cross_entropy_sents")
            # cross_entropy_sentence = tf.multiply(cross_entropy_sentence, self._weight)  # batch_size * sents_size
            #
            # cross_entropy_sentence = tf.reduce_sum(cross_entropy_sentence, axis=1)
            # weight_sum = tf.reduce_sum(self._weight, axis=1)
            # cross_entropy_sentence = cross_entropy_sentence / weight_sum
            # txt_loss = tf.reduce_mean(cross_entropy_sentence, name="cross_entropy_sentences")
            self.logits = tf.layers.dense(self.dec, vocab.vocab_size)
            preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.predict_txt = preds
            self.istarget = tf.to_float(tf.not_equal(self._response, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(preds, self._response)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
            self.y_smoothed = label_smoothing(tf.one_hot(self._response, depth=vocab.vocab_size))
            # pdb.set_trace()
            txt_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            mean_txt_loss = tf.reduce_sum(txt_loss * self.istarget) / (tf.reduce_sum(self.istarget))
            # all_loss = pic_loss + 0.0*txt_loss
            # pdb.set_trace()

            # all_loss=G_loss+txt_loss
            # loss_=tf.concat([tf.expand_dims(G_loss,-1),tf.expand_dims(cross_entropy_sentence,-1)],1)
            # all_loss=linear(loss_,1)

            self.losses = mean_txt_loss #+ pic_loss
        # pdb.set_trace()

        # grads_and_vars_txt = self._opt.compute_gradients(mean_txt_loss)
        # grads_and_vars_pic =self._opt.compute_gradients(pic_loss)
        # # pdb.set_trace()
        # def _merge_gradients(grads_vars_lists):
        #     """Merges the given losses into one tensor."""
        #     merged_grads=[]
        #     with tf.variable_scope('merge_gradients'):
        #         for idx,(grad, var) in enumerate(grads_vars_lists[0]):
        #             # pdb.set_trace()
        #             if var==grads_vars_lists[1][idx][1]:
        #                 if grad is not None and grads_vars_lists[1][idx][0] is not None:
        #                     merged_grads.append((tf.add_n([grad,grads_vars_lists[1][idx][0]]),var))
        #                 else:
        #                     if grad is None:
        #                         merged_grads.append((grads_vars_lists[1][idx][0],var))
        #                     if grads_vars_lists[1][idx][0] is None:
        #                         merged_grads.append((grad, var))
        #                     # if grad is None and grads_vars_lists[1][idx][0] is None:
        #                     #     print(var)
        #                         # pdb.set_trace()
        #             else:
        #                 print(var)
        #                 # print(grads_vars_lists[1][idx][0])
        #                 # pdb.set_trace()
        #     # pdb.set_trace()
        #     return merged_grads
        #
        # losses_grads=_merge_gradients([grads_and_vars_txt,grads_and_vars_pic])
        # pdb.set_trace()
        # grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        # grads_and_vars = [(self.add_gradient_noise(g), v) for g, v in grads_and_vars]

        # self.train_op = self._opt.apply_gradients(grads_and_vars=losses_grads, name='train_op')
        # pdb.set_trace()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with tf.variable_scope('output_information'):
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer
            # self.train_op = self._opt.minimize(self.losses, global_step=self.global_step)
            grads_and_vars = self._opt.compute_gradients(self.losses)
            self.train_ops=self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')
            tf.summary.scalar('mean_loss', self.losses)
            self.merged = tf.summary.merge_all()

    def add_gradient_noise(self, t, stddev=1e-3, name=None):
        """
        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2]..
        """
        with tf.name_scope("add_gradient_noise") as name:
            t = tf.convert_to_tensor(t, name="t")
            gn = tf.random_normal(tf.shape(t), stddev=stddev)
            return tf.add(t, gn, name=name)

    def _build_inputs(self):
        self._question = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Question')
        self._response = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Response')
        self._weight = tf.placeholder(tf.float32, [self._batch_size, self._sentence_size], name='weight')
        self._input_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, 1],
                                         name='frame_input')
        # self._real_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, 1],
        #                                 name='frame_output')
        # self._random_z=tf.placeholder(tf.float32,[self._batch_size,self._noise_dim],name='noise')

    def steps(self, sess, data_dict, noise, step_type='train'):
        self.model_type = step_type
        input_batch_txt = data_dict[0]
        output_batch_txt = data_dict[1]
        input_batch_pic = data_dict[2]
        # output_batch_pic = data_dict[3]
        weight_batch_txt = data_dict[4]
        # pdb.set_trace()
        feed_dict = {self._response: output_batch_txt,
                     self._question: input_batch_txt,
                     self._weight: weight_batch_txt,
                     self._input_pic: input_batch_pic}
                     # self._real_pic: output_batch_pic}
        # self._random_z:noise}

        if step_type == 'train':
            output_list = [self.losses, self.train_ops,self.merged]
            try:
                loss, _,summary = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, summary

        if step_type == 'test':
            output_list = [self.losses, self.predict_txt]#, self.predict_pic
            try:
                loss, txt = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, txt

        print('step_type is wrong!>>>')
        return None
