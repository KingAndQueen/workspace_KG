import pdb
import convolution
# from VGG import build_vgg19
from transformer import *
import copy

class seq_pic2seq_pic():
    def __init__(self, config, vocab,img_numb,candidates_vector_len):
        if config.gray:
            self._color_size = 1
        else:
            self._color_size = 3
        self._vocab = vocab
        self._img_numb=img_numb
        self._candidates_vector_len=candidates_vector_len
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

        self.g_bn0 = convolution.batch_norm(name='g_bn0')
        self.g_bn1 = convolution.batch_norm(name='g_bn1')
        self.g_bn2 = convolution.batch_norm(name='g_bn2')
        self.g_bn3 = convolution.batch_norm(name='g_bn3')

        self.e_bn0 = convolution.batch_norm(name='e_bn0')
        self.e_bn1 = convolution.batch_norm(name='e_bn1')
        self.e_bn2 = convolution.batch_norm(name='e_bn2')
        self.e_bn3 = convolution.batch_norm(name='e_bn3')


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


        # with tf.variable_scope('encoding_frame_resnet'):
        #     # resident net
        #     resnet_output, end_points = convolution.resnet_v2_50(self._input_pic)
        #     # encode_output=end_points['encoding_frame/resnet_v2_50/block4']
        #     self.pic_encoding = resnet_output
        #     w_pic = tf.get_variable('w', [1, 2048, self._embedding_size],
        #                             initializer=tf.random_normal_initializer(stddev=0.02))
        #     resnet_output = tf.squeeze(resnet_output, 1)
        #     encoding_pic_output = tf.nn.conv1d(resnet_output, w_pic, 1, 'SAME')
        #     encoder_pic_output = tf.tile(encoding_pic_output, [1, self._sentence_size, 1])
        #     pdb.set_trace()
        def conv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d", with_w=False):
            with tf.variable_scope(name):
                # pdb.set_trace()
                # filter : [height, width, output_channels, in_channels]
                w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_shape[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))

                deconv = tf.nn.conv2d(input_, filter=w, strides=[1, d_h, d_w, 1], padding="SAME")

                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

                if with_w:
                    return deconv, w, biases
                else:
                    return deconv

        def lrelu(x, leak=0.2, name="lrelu"):
            return tf.maximum(x, leak * x)

        with tf.variable_scope('encoding_frame_cnn'):
            s = self.img_size_x
            y = self.img_size_y
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
            y2, y4, y8, y16 = int(y / 2), int(y / 4), int(y / 8), int(y / 16)
            # pdb.set_trace()
            h0_e,w0,b0 = conv2d(self._input_pic, [self._batch_size, s, y, self._cov_size * 1], name='e_h0',with_w=True)
            h0_e = lrelu(self.e_bn0(h0_e, type=self.is_training))

            h1_e,w1,b1 = conv2d(h0_e, [self._batch_size, s2, y2, self._cov_size * 2], name='e_h1',with_w=True)
            h1_e = lrelu(self.e_bn1(h1_e, type=self.is_training))

            h2_e,w2,b2 = conv2d(h1_e, [self._batch_size, s4, y4, self._cov_size * 4], name='e_h2',with_w=True)
            h2_e = lrelu(self.e_bn2(h2_e, type=self.is_training))

            h3_e,w3,b3 = conv2d(h2_e, [self._batch_size, s8, y8, self._cov_size * 8], name='e_h3',with_w=True)
            h3_e = lrelu(self.e_bn3(h3_e, type=self.is_training))

            h4_e= tf.reshape(h3_e, [self._batch_size, -1], name='e_h4')
            # pdb.set_trace()
            w_pic = tf.get_variable('w', [1, h4_e.get_shape()[-1], self._embedding_size],
                                    initializer=tf.random_normal_initializer(stddev=0.02))
            encoding_pic_output = tf.nn.conv1d(tf.expand_dims(h4_e,1), w_pic, 1, 'SAME')
            encoder_pic_output = tf.tile(encoding_pic_output, [1, self._sentence_size, 1])

        def deconv2d(input_, output_shape,weight_cnn,biase_cnn=None, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
            with tf.variable_scope(name):
                # filter : [height, width, output_channels, in_channels]
                # w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                #                     initializer=tf.random_normal_initializer(stddev=stddev))
                # w=tf.transpose(weight_cnn,[0,1,3,2])
                # pdb.set_trace()

                biase_cnn = tf.negative(biase_cnn)
                input_bias = tf.nn.bias_add(input_, biase_cnn)
                deconv = tf.nn.conv2d_transpose(input_bias, weight_cnn, output_shape=output_shape,strides=[1, d_h, d_w,1])
                # if biase_cnn is None:
                #     return deconv
                # biase_cnn = tf.negative(biase_cnn)
                # deconv_output = tf.nn.bias_add(deconv, biase_cnn)
                # biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

                return deconv

        with tf.variable_scope('decoder_pic'):
            # h0 = ztf.reshape(h3_e, [-1, s16, y16, self._cov_size * 8])
            h0 = tf.nn.relu(self.g_bn0(h3_e, type=self.is_training))
            # pdb.set_trace()
            h1 = deconv2d(h0, [self._batch_size, s8, y8, self._cov_size * 4],weight_cnn=w3,biase_cnn=b3, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, type=self.is_training))

            h2 = deconv2d(h1, [self._batch_size, s4, y4, self._cov_size * 2],weight_cnn=w2,biase_cnn=b2, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, type=self.is_training))

            h3 = deconv2d(h2, [self._batch_size, s2, y2, self._cov_size * 1],weight_cnn=w1,biase_cnn=b1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, type=self.is_training))

            h4 = deconv2d(h3, [self._batch_size, s, y, self._color_size],weight_cnn=w0,biase_cnn=b0, name='g_h4')

            self.encoder_pic =  tf.tanh(h4) / 2. + 0.5

        with tf.variable_scope('embed_decode_input'):
            # pdb.set_trace()
            decoder_input = tf.concat((tf.ones_like(self._response[:, :1]) * 2, self._response[:, :-1]), -1) ## add 'go' sign
            dec_input = embedding(decoder_input,                   #self._response,
                                 vocab_size=vocab.vocab_size,
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

        # with tf.variable_scope('merge_txt_pic'):
        #     decoder_input = tf.concat((encoder_pic_output, self.dec), -1)
        #     w_merge = tf.get_variable('w', [1, 2 * self._embedding_size, self._embedding_size],
        #                               initializer=tf.random_normal_initializer(stddev=0.02))
        #     # pdb.set_trace()
        #     self.dec = tf.nn.conv1d(decoder_input, w_merge, 1, 'SAME')

        with tf.variable_scope('merge_txt_pic'):
            # pdb.set_trace()
            decoder_input = tf.concat((encoder_pic_output, self.enc), -1)
            w_merge = tf.get_variable('w', [1, 2 * self._embedding_size, self._embedding_size],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
            self.enc = tf.nn.conv1d(decoder_input, w_merge, 1, 'SAME')

        with tf.variable_scope('decode_txt'):
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

                    self.dec_output = feedforward(self.dec_output, num_units=[4 * self._embedding_size, self._embedding_size])

        with tf.variable_scope('img_classification'):

            classfy_input = tf.concat((tf.expand_dims(encoder_pic_output,-1), tf.expand_dims(self.enc,-1),tf.expand_dims(self.dec_output,-1)), -1)

            with tf.variable_scope('text_input_cnn_classify'):
                # pdb.set_trace()
                w = tf.get_variable('w', [5, 5,3, 64],
                                    initializer=tf.random_normal_initializer(stddev=0.02))

                conv = tf.nn.conv2d(classfy_input, filter=w, strides=[1, 2, 2, 1], padding="SAME")

                biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
                conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            conv_classfy=tf.reshape(conv, [self._batch_size, -1], name='classify_reshape')

            # feedforward(conv_classfy,num_units=[conv_classfy.get_shape()[-1],self._cov_size],scope='classify_ff')
            context_img = tf.layers.dense(conv_classfy, self._img_numb)
            context_img_vgg=tf.sparse_softmax(context_img * self._candidates_pool)
            self.logits_img = tf.layers.dense(context_img_vgg, self._img_numb)
            self.predict_img = tf.to_int32(tf.argmax(self.logits_img, axis=-1))
            self.acc_img = tf.reduce_sum(tf.to_float(tf.equal(self.predict_img, self._real_pic)))
            tf.summary.scalar('acc_img', self.acc_img)
            self.img_real_smoothed = label_smoothing(tf.one_hot(self._real_pic, depth=self._img_numb))
            img_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_img, labels=self.img_real_smoothed)
            mean_img_loss=tf.reduce_mean(img_loss)
            self.losses_img=mean_img_loss

        with tf.variable_scope('loss_function_txt'):

            self.logits = tf.layers.dense(self.dec_output, vocab.vocab_size)
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

            self.losses = mean_txt_loss+ self.losses_img # + pic_loss

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
        self._input_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, self._color_size],
                                         name='frame_input')
        self._real_pic = tf.placeholder(tf.int32, [self._batch_size],
                                        name='frame_output')
        self._candidates_pool=tf.placeholder(tf.float32, [self._img_numb,self._candidates_vector_len],
                                         name='candidates_pool')
        # self._random_z=tf.placeholder(tf.float32,[self._batch_size,self._noise_dim],name='noise')

    def steps(self, sess, data_dict,candidates_pool, noise=None, step_type='train',qa_transpose=False, img_affect_testing=None):
        # self.is_training = step_type
        input_batch_txt = data_dict[0]
        output_batch_txt = data_dict[1]
        input_batch_pic = data_dict[2]
        if isinstance(img_affect_testing ,int):
            input_batch_pic_temp=[]
            for idx,pic in enumerate(input_batch_pic):
                id=(idx+img_affect_testing)%len(input_batch_pic)
                input_batch_pic_temp.append(input_batch_pic[id])
            input_batch_pic=input_batch_pic_temp
        output_batch_pic = data_dict[3]
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
            feed_dict = {self._response: output_batch_txt,
                         self._question: input_batch_txt,
                         # self._weight: weight_batch_txt,
                         self._input_pic: input_batch_pic,
                         self._real_pic:output_batch_pic,
                         self._candidates_pool:candidates_pool
                         }
            output_list = [self.losses, self.train_ops, self.merged]
            try:
                loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()

            if qa_transpose:
                feed_dict = {self._response: input_batch_txt,
                             self._question: output_batch_txt,
                             # self._weight: weight_batch_txt,
                             self._input_pic: output_batch_pic}
                try:
                    loss_t, _, summary = sess.run(output_list, feed_dict=feed_dict)
                    loss=(loss+loss_t)/2
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
            pic_encoding,acc_img=None,None
            word_defined_image=[]
            for j in range(self._sentence_size):
                txt_preds,pic_encoding,acc_img = sess.run([self.predict_txt,self.encoder_pic,self.acc_img],
                                  feed_dict={self._question: input_batch_txt, self._response: output_batch_txt,
                                             self._input_pic: input_batch_pic,self._real_pic:output_batch_pic})
                output_batch_txt[:, j] = txt_preds[:, j]

            # feed_dict = {self._response: output_batch_txt,
            #              self._question: input_batch_txt,
            #              # self._weight: weight_batch_txt,
            #              self._input_pic: input_batch_pic}
            # output_list = [self.losses, self.predict_txt]#, self.predict_pic
            # try:
            #     loss, txt = sess.run(output_list, feed_dict=feed_dict)
            # except:
            #     pdb.set_trace()
                word_defined_image.append([copy.copy(output_batch_txt),copy.copy(pic_encoding)])
            return output_batch_txt,pic_encoding,word_defined_image,acc_img

        print('step_type is wrong!>>>')
        return None
