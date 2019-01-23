import pdb
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
        self.is_training = config.is_training
        self._num_identical = config.num_identical
        self._build_inputs()
        self._head = config.head

        # if self.model_type == 'train':
        #     self.is_training = True
        # else:
        #     self.is_training = False

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
                                         training=tf.convert_to_tensor(self.is_training))

            # Identical
            for i in range(self._num_identical):
                with tf.variable_scope("num_identical_{}".format(i)):
                    # Multi-head Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._head,
                                                   dropout_rate=0.1,
                                                   is_training=self.is_training,
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
                w_pic = tf.get_variable('w', [1, 2048, self._embedding_size],
                                        initializer=tf.random_normal_initializer(stddev=0.02))
                resnet_output = tf.squeeze(resnet_output, 1)
                encoding_pic_output = tf.nn.conv1d(resnet_output, w_pic, 1, 'SAME')
                encoding_pic_output = tf.tile(encoding_pic_output, [1, self._sentence_size, 1])
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
            w_merge = tf.get_variable('w', [1, 2 * self._embedding_size, self._embedding_size],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
            # pdb.set_trace()
            self.dec = tf.nn.conv1d(decoder_input, w_merge, 1, 'SAME')

        with tf.variable_scope('decode_txt'):
            # Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=0.1,
                                         training=self.is_training)
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
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Multi-head Attention(vanilla-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._head,
                                                   dropout_rate=0.1,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    self.dec = feedforward(self.dec, num_units=[4 * self._embedding_size, self._embedding_size])

        with tf.variable_scope('loss_function_txt'):

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

            self.losses = mean_txt_loss  # + pic_loss

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with tf.variable_scope('output_information'):
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer
            self.train_ops = self._opt.minimize(self.losses)
            # grads_and_vars = self._opt.compute_gradients(self.losses)
            # self.train_ops=self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')
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
        # self._weight = tf.placeholder(tf.float32, [self._batch_size, self._sentence_size], name='weight')
        self._input_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, 1],
                                         name='frame_input')
        # self._real_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, 1],
        #                                 name='frame_output')
        # self._random_z=tf.placeholder(tf.float32,[self._batch_size,self._noise_dim],name='noise')

    def steps(self, sess, data_dict, noise, step_type='train'):
        self.is_training = step_type
        input_batch_txt = data_dict[0]
        output_batch_txt = data_dict[1]
        input_batch_pic = data_dict[2]
        # output_batch_pic = data_dict[3]
        # weight_batch_txt = data_dict[4]
        # pdb.set_trace()
        # feed_dict = {self._response: output_batch_txt,
        #              self._question: input_batch_txt,
        #              # self._weight: weight_batch_txt,
        #              self._input_pic: input_batch_pic}
        # self._real_pic: output_batch_pic}
        # self._random_z:noise}

        if step_type == 'train':
            pdb.set_trace()
            feed_dict = {self._response: output_batch_txt,
                         self._question: input_batch_txt,
                         # self._weight: weight_batch_txt,
                         self._input_pic: input_batch_pic}
            output_list = [self.losses, self.train_ops, self.merged]
            try:
                loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, summary

        if step_type == 'test':
            output_batch_txt = np.zeros((self._batch_size, self._sentence_size), dtype=np.int32)

            for j in range(self._batch_size):
                _preds = sess.run(self.predict_txt,
                                  feed_dict={self._question: input_batch_txt, self._response: output_batch_txt,
                                             self._input_pic: input_batch_pic})
                output_batch_txt[:, j] = _preds[:, j]

            # feed_dict = {self._response: output_batch_txt,
            #              self._question: input_batch_txt,
            #              # self._weight: weight_batch_txt,
            #              self._input_pic: input_batch_pic}
            # output_list = [self.losses, self.predict_txt]#, self.predict_pic
            # try:
            #     loss, txt = sess.run(output_list, feed_dict=feed_dict)
            # except:
            #     pdb.set_trace()
            return output_batch_txt

        print('step_type is wrong!>>>')
        return None
