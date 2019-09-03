import pdb
import convolution
# from VGG import build_vgg19
from transformer import *
import copy


class seq_pic2seq_pic():
    def __init__(self, config, vocab, img_numb=None, candidates_vector_len=None):
        if config.gray:
            self._color_size = 1
        else:
            self._color_size = 3
        # self._vocab = vocab
        # self._img_numb = img_numb
        # self._candidates_vector_len = candidates_vector_len
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32, name='learn_rate')
        self._round = config.round
        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        # self._opt = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learn_rate, beta1=0.9, beta2=0.98,
                                                     epsilon=1e-8)

        self._embedding_size = config.recurrent_dim
        self._layers = config.layers
        self.img_feature_layer = config.img_feature_layer
        self.img_feature_vector = config.img_feature_vector
        # self._gf_dim=config.gf_dim
        self._max_grad_norm = config.max_grad_norm
        # self._cov_size = config.convolution_dim
        self._noise_dim = config.noise_dim
        self.is_training = config.is_training
        self._num_identical = config.num_identical
        self._build_inputs()
        self._head = config.head
        # self._candidates_pool = tf.Variable(self._candidates_pool_ph, trainable=False)

        with tf.compat.v1.variable_scope("encode_txt"):
            self.enc = embedding(self._question,
                                 vocab_size=len(vocab),
                                 num_units=self._embedding_size,
                                 scale=True,
                                 scope="enc_embed")

            # Position Encoding(use range from 0 to len(inpt) to represent position dim of each words)
            # tf.tile(tf.expand_dims(tf.range(tf.shape(self.inpt)[1]), 0), [tf.shape(self.inpt)[0], 1]),

            self.enc += positional_encoding(self._question, vocab_size=self._sentence_size,
                                            num_units=self._embedding_size, zero_pad=False, scale=False,
                                            scope="enc_pe")

            # self.enc += positional_encoding(self._question,
            #                                 vocab_size=self._sentence_size,
            #                                 num_units=self._embedding_size,
            #                                 zero_pad=False,
            #                                 scale=False,
            #                                 scope="enc_pe")

            # Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=0.1,
                                         training=self.is_training)

            # Identical
            for i in range(self._num_identical):
                with tf.compat.v1.variable_scope("num_identical_{}".format(i)):
                    # Multi-head Attention
                    self.enc = multihead_attention(queries=self.enc, keys=self.enc, num_units=self._embedding_size,
                                                   num_heads=self._head, dropout_rate=0.1,
                                                   is_training=self.is_training,
                                                   causality=False)
                    self.enc = feedforward(self.enc,
                                           num_units=[4 * self._embedding_size, self._embedding_size])  # 4 30 64

        # def conv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d", with_w=False):
        #     with tf.compat.v1.variable_scope(name):
        #         # pdb.set_trace()
        #         # filter : [height, width, output_channels, in_channels]
        #         w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_shape[-1]],
        #                             initializer=tf.random_normal_initializer(stddev=stddev))
        #
        #         deconv = tf.nn.conv2d(input_, filter=w, strides=[1, d_h, d_w, 1], padding="SAME")
        #
        #         biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        #
        #         if with_w:
        #             return deconv, w, biases
        #         else:
        #             return deconv
        #
        # def lrelu(x, leak=0.2, name="lrelu"):
        #     return tf.maximum(x, leak * x)

        # with tf.variable_scope('encoding_frame_cnn'):
        #     s = self.img_size_x
        #     y = self.img_size_y
        #     s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
        #     y2, y4, y8, y16 = int(y / 2), int(y / 4), int(y / 8), int(y / 16)
        #     # pdb.set_trace()
        #     h0_e,w0,b0 = conv2d(self._input_pic, [self._batch_size, s, y, self._cov_size * 1], name='e_h0',with_w=True)
        #     h0_e = lrelu(self.e_bn0(h0_e, type=self.is_training))
        #
        #     h1_e,w1,b1 = conv2d(h0_e, [self._batch_size, s2, y2, self._cov_size * 2], name='e_h1',with_w=True)
        #     h1_e = lrelu(self.e_bn1(h1_e, type=self.is_training))
        #
        #     h2_e,w2,b2 = conv2d(h1_e, [self._batch_size, s4, y4, self._cov_size * 4], name='e_h2',with_w=True)
        #     h2_e = lrelu(self.e_bn2(h2_e, type=self.is_training))
        #
        #     h3_e,w3,b3 = conv2d(h2_e, [self._batch_size, s8, y8, self._cov_size * 8], name='e_h3',with_w=True)
        #     h3_e = lrelu(self.e_bn3(h3_e, type=self.is_training))
        #
        #     h4_e= tf.reshape(h3_e, [self._batch_size, -1], name='e_h4')
        #     # pdb.set_trace()
        #     w_pic = tf.get_variable('w', [1, h4_e.get_shape()[-1], self._embedding_size],
        #                             initializer=tf.random_normal_initializer(stddev=0.02))
        #     encoding_pic_output = tf.nn.conv1d(tf.expand_dims(h4_e,1), w_pic, 1, 'SAME')
        #     pdb.set_trace()
        #     encoder_pic_output = tf.tile(encoding_pic_output, [1, self._sentence_size, 1])

        # with tf.compat.v1.variable_scope('encoding_frame_cnn'):
        #     # resident net
        #     # resnet_output, end_points = convolution.resnet_v2_50(self._input_pic)
        #     # # encode_output=end_points['encoding_frame/resnet_v2_50/block4']
        #     # self.pic_encoding = resnet_output
        #     # w_pic = tf.get_variable('w', [1, 2048, self._embedding_size],
        #     #                         initializer=tf.random_normal_initializer(stddev=0.02))
        #     # resnet_output = tf.squeeze(resnet_output, 1)
        #     # encoding_pic_output = tf.nn.conv1d(resnet_output, w_pic, 1, 'SAME')
        #     # encoder_pic_output = tf.tile(encoding_pic_output, [1, self._sentence_size, 1])
        #
        #     resnet_output, end_points = convolution.resnet_v2_50(self._input_pic)
        #     # resnet_output=end_points['resnet_v2_50' + '/block4'] # test2 to check the full connections effect
        #     encode_output = end_points['encoding_frame_cnn/resnet_v2_50/block4']
        #
        #     encode_output = tf.reshape(encode_output,
        #                                (self._batch_size, -1, 2048))
        #     trans_encode_out = tf.layers.dense(encode_output, 64)  # -> 4 50 512
        #
        #     # trans_encode_out = multihead_attention(
        #     #     queries=trans_encode_out,
        #     #     keys=trans_encode_out,
        #     #     num_units=512,
        #     #     num_heads=8,
        #     #     dropout_rate=0.1,
        #     #     # is_training=self.is_training,
        #     #     causality=False)
        #
        #     trans_encode_out = multihead_attention(
        #         queries=trans_encode_out,
        #         keys=trans_encode_out,
        #         num_units=self._embedding_size,
        #         num_heads=self._head,
        #         dropout_rate=0.1,
        #         is_training=self.is_training,
        #         causality=False)
        #     # trans_encode_out = feedforward(trans_encode_out,num_units=[4 * 512, 512],reuse=True)
        #     trans_encode_out = tf.reshape(trans_encode_out, (self._batch_size, -1))  # pic 4 * 5 * 10 * 512
        #     trans_encode_out = tf.layers.dense(trans_encode_out, 30 * 64)
        #     encoder_pic_output = tf.reshape(trans_encode_out, (self._batch_size, 30, 64))

        with tf.compat.v1.variable_scope('embed_decode_input'):
            # pdb.set_trace()
            # decoder_input = tf.concat((tf.ones_like(self._response[:, :1]) * 2, self._response[:, :-1]),
            #                           -1)  ## add 'go' sign
            decoder_input = self._response_in
            dec_input = embedding(decoder_input,  # self._response,
                                  vocab_size=len(vocab),
                                  num_units=self._embedding_size,
                                  scale=True,
                                  scope="dec_embed")

            # Position Encoding(use range from 0 to len(inpt) to represent position dim)
            dec_input += positional_encoding(decoder_input,
                                             vocab_size=self._sentence_size,
                                             num_units=self._embedding_size,
                                             zero_pad=False,
                                             scale=False,
                                             scope="dec_pe")

        with tf.compat.v1.variable_scope('merge_txt_pic'):
            # encoder_pic_output 4 5 10 512 -> 4 30 64
            # pic 30 64 > 1 64  sentence 30 * 64
            # pdb.set_trace()
            # encoder_pic_output = tf.reshape(self._input_pic, [self._batch_size, -1])

            encoder_pic_output = tf.layers.dense(self._input_pic, self._embedding_size)
            # encoder_pic_output = tf.expand_dims(encoder_pic_output, 1)  # batch size 1 64

            test_enc = tf.reshape(self.enc, [self._batch_size, -1])
            test_enc= tf.layers.dense(test_enc, self._embedding_size)
            test_enc = tf.expand_dims(test_enc, axis=1)
            cur_self_enc = tf.tile(test_enc, [1, self.img_feature_layer, 1])


            # pdb.set_trace()
            pic_attention = tf.nn.softmax(tf.multiply(encoder_pic_output, cur_self_enc))  # 8 1 64 \ 8 64 30 -> 8 1 30
            pic_attention = tf.reshape(pic_attention, [self._batch_size, -1])
            pic_attention= tf.expand_dims(pic_attention, axis=1)
            pic_attention=tf.tile(pic_attention,[1, self._sentence_size, 1])
            # # encoder_pic_output = tf.tile(encoder_pic_output, [1, 30, 1])  # batch size 30 64
            # encoder_pic_output = tf.matmul(encoder_pic_output, pic_attention)

            decoder_input = tf.concat((pic_attention, self.enc), -1)
            self.enc=tf.layers.dense(decoder_input, self._embedding_size)
            # w_merge = tf.compat.v1.get_variable('w', [1, 2 * self._embedding_size, self._embedding_size],
            #                                     initializer=tf.random_normal_initializer(stddev=0.02))
            #
            # self.enc = tf.nn.conv1d(decoder_input, w_merge, 1, 'SAME')



        with tf.compat.v1.variable_scope('decode_txt'):
            # Dropout
            dec_input = tf.layers.dropout(dec_input,
                                          rate=0.1,
                                          training=self.is_training)
            # Identical
            # pdb.set_trace()

            for i in range(self._num_identical):
                with tf.variable_scope("num_identical_{}".format(i)):
                    # Multi-head Attention(self-attention)
                    dec_input = multihead_attention(queries=dec_input,
                                                    keys=dec_input,
                                                    num_units=self._embedding_size,
                                                    num_heads=self._head,
                                                    dropout_rate=0.1,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    scope="self_attention")

                    # Multi-head Attention(vanilla-attention)
                    self.dec_output = multihead_attention(queries=dec_input,
                                                          keys=self.enc,
                                                          num_units=self._embedding_size,
                                                          num_heads=self._head,
                                                          dropout_rate=0.1,
                                                          is_training=self.is_training,
                                                          causality=False,
                                                          scope="vanilla_attention")

                    self.dec_output = feedforward(self.dec_output,
                                                  num_units=[4 * self._embedding_size, self._embedding_size])

        with tf.compat.v1.variable_scope('loss_function_txt'):

            self.logits = tf.layers.dense(self.dec_output, len(vocab))
            preds = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            self.predict_txt = preds
            self.istarget = tf.cast(tf.not_equal(self._response_out, 0), tf.float32)
            self.acc = tf.reduce_sum(tf.cast(tf.equal(preds, self._response_out), tf.float32) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
            self.y_smoothed = label_smoothing(tf.one_hot(self._response_out, depth=len(vocab)))
            # pdb.set_trace()
            txt_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            mean_txt_loss = tf.reduce_sum(txt_loss * self.istarget) / (tf.reduce_sum(self.istarget))
            # all_loss = pic_loss + 0.0*txt_loss
            # pdb.set_trace()

            self.losses = mean_txt_loss
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        with tf.compat.v1.variable_scope('output_information'):
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer
            self.train_ops = self._opt.minimize(self.losses)
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
        self._question = tf.compat.v1.placeholder(tf.int32, [self._batch_size, self._sentence_size],
                                                  name='Question')
        self._response_in = tf.compat.v1.placeholder(tf.int32, [self._batch_size, self._sentence_size],
                                                     name='Response_in')
        self._response_out = tf.compat.v1.placeholder(tf.int32, [self._batch_size, self._sentence_size],
                                                      name='Response_out')
        # self._weight = tf.placeholder(tf.float32, [self._batch_size, self._sentence_size], name='weight')
        self._input_pic = tf.compat.v1.placeholder(tf.float32,
                                                   [self._batch_size, self.img_feature_layer, self.img_feature_vector],
                                                   name='frame_input')
        # self._real_pic = tf.placeholder(tf.int32, [self._batch_size],
        #                                 name='frame_output')
        # self._candidates_pool_ph = tf.placeholder(tf.float32, [self._img_numb, self._candidates_vector_len],
        #                                           name='candidates_pool')
        # self._random_z=tf.placeholder(tf.float32,[self._batch_size,self._noise_dim],name='noise')

    def steps(self, sess, data_dict, noise=None, step_type='train', qa_transpose=False, img_affect_testing=None):
        # self.is_training = step_type
        input_batch_txt = data_dict[0]
        output_batch_txt_in = data_dict[1]
        output_batch_txt_out = data_dict[2]
        input_batch_pic = np.tile(data_dict[3],[self._batch_size,1,1])

        # if isinstance(img_affect_testing, int):
        #     input_batch_pic_temp = []
        #     for idx, pic in enumerate(input_batch_pic):
        #         id = (idx + img_affect_testing) % len(input_batch_pic)
        #         input_batch_pic_temp.append(input_batch_pic[id])
        #     input_batch_pic = input_batch_pic_temp
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
            # pdb.set_trace()
            feed_dict = {self._response_in: output_batch_txt_in,
                         self._response_out: output_batch_txt_out,
                         self._question: input_batch_txt,
                         # self._weight: weight_batch_txt,
                         self._input_pic: input_batch_pic,
                         # self._real_pic: output_batch_pic,
                         # self._candidates_pool:candidates_pool
                         }
            output_list = [self.losses, self.train_ops, self.merged]
            try:
                loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()

            if qa_transpose:
                feed_dict = {self._response_in: input_batch_txt,
                             self._question: output_batch_txt_in,
                             # self._weight: weight_batch_txt,
                             self._input_pic: input_batch_pic}
                try:
                    loss_t, _, summary = sess.run(output_list, feed_dict=feed_dict)
                    loss = (loss + loss_t) / 2
                except:
                    pdb.set_trace()
            return loss, summary

        # if step_type == 'valid':
        #     # pdb.set_trace()
        #     feed_dict = {self._response: output_batch_txt,
        #                  self._question: input_batch_txt,
        #                  # self._weight: weight_batch_txt,
        #                  self._input_pic: input_batch_pic}
        #     output_list = [self.losses, self.merged]
        #     try:
        #         loss,summary = sess.run(output_list, feed_dict=feed_dict)
        #     except:
        #         pdb.set_trace()
        #     return loss, summary

        if step_type == 'test':
            output_batch_txt = np.zeros((self._batch_size, self._sentence_size), dtype=np.int32)
            pic_encoding, acc_img = None, None
            word_defined_image = []
            for j in range(self._sentence_size):
                # txt_preds,pic_encoding,acc_img = sess.run([self.predict_txt,self.encoder_pic,self.acc_img],
                txt_preds = sess.run([self.predict_txt],
                                     feed_dict={self._question: input_batch_txt,
                                                self._response: output_batch_txt,
                                                self._input_pic: input_batch_pic, })
                # self._real_pic: output_batch_pic})
                # output_batch_txt[:, j] = txt_preds[:, j]
                output_batch_txt = txt_preds[0]
                # feed_dict = {self._response: output_batch_txt,
                #              self._question: input_batch_txt,
                #              # self._weight: weight_batch_txt,
                #              self._input_pic: input_batch_pic}
                # output_list = [self.losses, self.predict_txt]#, self.predict_pic
                # try:
                #     loss, txt = sess.run(output_list, feed_dict=feed_dict)
                # except:
                #     pdb.set_trace()
                word_defined_image.append([copy.copy(output_batch_txt), copy.copy(pic_encoding)])
            # return output_batch_txt,pic_encoding,word_defined_image,acc_img
            return output_batch_txt, pic_encoding, word_defined_image, acc_img
        print('step_type is wrong!>>>')
        return None
