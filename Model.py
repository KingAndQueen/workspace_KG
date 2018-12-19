import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.util import nest
import pdb
import convolution
from VGG import build_vgg19


class seq_pic2seq_pic():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32, name='learn_rate')

        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        self._opt = tf.train.AdamOptimizer(learning_rate=self._learn_rate)

        self._embedding_size = config.recurrent_dim
        self._layers = config.layers
        self.img_size_x = config.img_size_x
        self.img_size_y = config.img_size_y
        # self._gf_dim=config.gf_dim
        self._max_grad_norm = config.max_grad_norm
        self._cov_size = config.convolution_dim
        self._noise_dim = config.noise_dim
        self.model_type = config.model_type
        self._build_inputs()

        self.g_bn0 = convolution.batch_norm(name='g_bn0')
        self.g_bn1 = convolution.batch_norm(name='g_bn1')
        self.g_bn2 = convolution.batch_norm(name='g_bn2')
        self.g_bn3 = convolution.batch_norm(name='g_bn3')

        with tf.variable_scope('embedding'):
            self._word_embedding = tf.get_variable(name='embedding_word',
                                                   shape=[self._vocab.vocab_size, config.recurrent_dim])
            _Question = tf.unstack(self._question, axis=1)
            question_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Question]
            _Response = tf.unstack(self._response, axis=1)
            response_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Response]

        def _encoding_txt_sentence(person_emb, name='', GPU_id=0):
            # with tf.device('/device:GPU:%d' %GPU_id):
            with tf.variable_scope('encoding_TXT_' + name):
                # encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.recurrent_dim, reuse=tf.get_variable_scope().reuse)
                # encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                # , reuse=tf.get_variable_scope().reuse)
                encoding_single_layer = [tf.nn.rnn_cell.GRUCell(config.recurrent_dim) for n in range(config.layers)]
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell(encoding_single_layer)
                encoding_cell = tf.contrib.rnn.DropoutWrapper(encoding_cell, 0.8, 0.8, 0.8)
                # for future test
                # output, state_fw, state_bw = rnn.static_bidirectional_rnn(cell_fw=encoding_cell, cell_bw=encoding_cell,
                #                                                           inputs=person_emb, dtype=tf.float32)
                output, state_fw = rnn.static_rnn(encoding_cell, person_emb, dtype=tf.float32)
                # state = tf.concat([state_fw, state_bw], -1)
                # state = tf.matmul(state, tf.get_variable('Wi', [3, 2 * self._embedding_size, self._embedding_size],
                #                                          dtype=tf.float32, trainable=True))

                top_output = [array_ops.reshape(o, [-1, 1, config.recurrent_dim]) for o in output]
                # pdb.set_trace()
                attention_states = array_ops.concat(top_output, 1)
                return attention_states, state_fw

        encoder_txt_output, question_state = _encoding_txt_sentence(question_emb,
                                                                    name='question')  # monica_sate.shape=layers*[batch_size,neurons]

        # pdb.set_trace()
        def _encoding_pic_frame(frame, name='', GPU_id=0):
            with tf.variable_scope('encoding_frame_' + name):
                # resident net
                resnet_output, end_points = convolution.resnet_v2_50(frame)
                # resnet_output=end_points['resnet_v2_50' + '/block4'] # test2 to check the full connections effect
                encode_output=end_points['encoding_frame_'+name+'/resnet_v2_50/block4']
                # pdb.set_trace()

                return encode_output

        # pdb.set_trace()
        encoder_pic_output = _encoding_pic_frame(self._input_pic,'one')


        # pdb.set_trace()
        def decoder_txt_atten(encoder_state, attention_states, ans_emb, model_type='train'):
            with tf.variable_scope('speaker'):
                num_heads = 1
                batch_size = ans_emb[0].get_shape()[0]
                attn_length = attention_states.get_shape()[1].value
                attn_size = attention_states.get_shape()[2].value
                hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
                # (128,20,100)--> hidden= (128,20,1,100)
                hidden_features = []
                v = []
                attention_vec_size = attn_size
                for a in range(num_heads):
                    k = tf.get_variable('AttnW_%d' % a, [1, 1, attn_size, attention_vec_size])
                    hidden_features.append(
                        nn_ops.conv2d(hidden, k, [1, 1, 1, 1], 'SAME'))  # hidden_features=(128,20,1,100)
                    v.append(tf.get_variable('AttnV_%d' % a, [attention_vec_size]))  # [100]

                # pdb.set_trace()
                def attention(query):
                    ds = []
                    if nest.is_sequence(query):
                        query_list = nest.flatten(query)
                        for q in query_list:
                            ndims = q.get_shape().ndims
                            if ndims:
                                assert ndims == 2
                        query = array_ops.concat(query_list, 1)
                    for a in range(num_heads):
                        with tf.variable_scope('Attention_%d' % a):
                            y = linear(query, attention_vec_size, True)
                            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                            s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                                    [2, 3])  # shape=(128, 20)
                            a = nn_ops.softmax(s)  # shape=(128, 20)
                            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                                                    [1, 2])  # (128,100)
                            ds.append(array_ops.reshape(d, [-1, attn_size]))
                            # pdb.set_trace()
                    return ds

                def extract_argmax_and_embed(prev, _):
                    """Loop_function that extracts the symbol from prev and embeds it."""
                    prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
                    return embedding_ops.embedding_lookup(self._word_embedding, prev_symbol)

                if model_type == 'test':
                    loop_function = extract_argmax_and_embed
                else:
                    loop_function = None

                linear = core_rnn_cell._linear
                batch_attn_size = array_ops.stack([batch_size, attn_size])
                attns = [array_ops.zeros(batch_attn_size, dtype=tf.float32) for _ in range(num_heads)]
                # pdb.set_trace()
                for a in attns:
                    a.set_shape([None, attn_size])

                with tf.variable_scope("rnn_decoder"):
                    # single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    # cell_de = tf.nn.rnn_cell.MultiRNNCell([single_cell_de] * self._layers)
                    # $ cell_de = single_cell_de
                    # if self._layers > 1:
                    single_cell = [tf.nn.rnn_cell.GRUCell(self._embedding_size) for n in range(self._layers)]
                    cell_de = tf.nn.rnn_cell.MultiRNNCell(single_cell)
                    cell_de = tf.contrib.rnn.DropoutWrapper(cell_de, 0.5, 1, 0.5)
                    # cell_de = core_rnn_cell.OutputProjectionWrapper(cell_de, self._vocab_size)
                    outputs = []
                    prev = None
                    #   pdb.set_trace()
                    state = encoder_state
                    for i, inp in enumerate(ans_emb):
                        if loop_function is not None and prev is not None:
                            with tf.variable_scope("loop_function", reuse=True):
                                inp = array_ops.stop_gradient(loop_function(prev, i))
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()

                        num_emb_in = inp.get_shape()[1]
                        weights_initializer_emb = tf.truncated_normal_initializer(
                            stddev=0.1)
                        regularizer_emb = tf.contrib.layers.l2_regularizer(0.1)
                        weights_emb = tf.get_variable('weights',
                                                      shape=[num_emb_in, self._embedding_size / 2],
                                                      initializer=weights_initializer_emb,
                                                      regularizer=regularizer_emb)
                        biases_emb = tf.get_variable('biases',
                                                     shape=[self._embedding_size / 2],
                                                     initializer=tf.zeros_initializer)
                        inp = tf.nn.xw_plus_b(inp, weights_emb, biases_emb)
                        # inp = tf.concat([inp, speaker_embedding], 1)
                        inp = linear([inp] + attns, self._embedding_size, True)
                        output, state = cell_de(inp, state)
                        # pdb.set_trace()
                        attns = attention(state)

                        with tf.variable_scope('AttnOutputProjecton'):
                            output = linear([output] + attns, self._vocab.vocab_size, True)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = array_ops.stop_gradient(output)

                outputs = tf.transpose(outputs, perm=[1, 0, 2])
                return outputs

        # pdb.set_trace()

        response_txt = decoder_txt_atten(question_state, encoder_txt_output, response_emb, self.model_type)

        def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
            with tf.variable_scope(name):
                # filter : [height, width, output_channels, in_channels]
                w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))

                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                strides=[1, d_h, d_w, 1])

                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

                if with_w:
                    return deconv, w, biases
                else:
                    return deconv

        def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
            shape = input_.get_shape().as_list()


            with tf.variable_scope(scope or "Linear"):
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    # pdb.set_trace()
                    return tf.matmul(input_, matrix) + bias

        def lrelu(x, leak=0.2, name="lrelu"):
            return tf.maximum(x, leak * x)

        with tf.variable_scope('decoder_pic'):
            s = self.img_size_x
            y = self.img_size_y
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
            y2, y4, y8, y16 = int(y / 2), int(y / 4), int(y / 8), int(y / 16)
            # encoder_pic_output_reshape = tf.reshape(encoder_pic_output, [self._batch_size, -1])
            # response_txt_reshape=tf.reshape(response_txt,[self._batch_size,-1])
            # encoder_txt_output_reshape=tf.reshape(encoder_txt_output,[self._batch_size,-1])
            # all_infor=tf.concat([response_txt_reshape,encoder_txt_output_reshape,encoder_pic_output_reshape],1)
            # try more input method to replace all_infor # test1
            # pdb.set_trace()
            # reduced_text_embedding = lrelu(linear(all_infor, self._cov_size, 'g_embedding'))
            # reduced_text_embedding = lrelu(all_infor)
            # z_concat = tf.concat([self._random_z, reduced_text_embedding],1)
            # z_ = linear(encoder_pic_output_reshape, self._cov_size * 8 * s16 * y16, 'g_h0_lin')
            z_=encoder_pic_output
            # pdb.set_trace()
            h0 = tf.reshape(z_, [-1, s16, y16, self._cov_size * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, type=self.model_type))

            h1 = deconv2d(h0, [self._batch_size, s8, y8, self._cov_size * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, type=self.model_type))

            h2 = deconv2d(h1, [self._batch_size, s4, y4, self._cov_size * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, type=self.model_type))

            h3 = deconv2d(h2, [self._batch_size, s2, y2, self._cov_size * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, type=self.model_type))

            h4 = deconv2d(h3, [self._batch_size, s, y, 1], name='g_h4')

            predict_pic = tf.tanh(h4) / 2. + 0.5

            # print '$$$$$$$$$$$$$$$$$$$$'
            # print h4.get_shape().as_list()
            # print '$$$$$$$$$$$$$$$$$$$$'

        # pdb.set_trace()
        def compute_error(real, fake):
            return tf.reduce_mean(tf.abs(fake - real))
            # diversity loss

        with tf.variable_scope('loss_function_pic'):
            # pdb.set_trace()
            # cov_input=convolution.deeplab_v3(predict_pic)
            # cov_output=convolution.deeplab_v3(self._output_pic)
            # sp=self.img_size_x

            vgg_real = build_vgg19(self._real_pic)
            vgg_fake = build_vgg19(predict_pic, reuse=True)
            p0 = compute_error(vgg_real['input'], vgg_fake['input'])
            p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 1.6
            p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 2.3
            p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 1.8
            p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 2.8
            p5 = compute_error(vgg_real['conv5_2'],
                               vgg_fake['conv5_2']) * 10 / 0.8  # weights lambda are collected at 100th epoch
            content_loss = p0 + p1 + p2 + p3 + p4 + p5

            # pdb.set_trace()
            vgg_loss = tf.reduce_sum(tf.reduce_min(content_loss)) * 0.999 + tf.reduce_sum(
                tf.reduce_mean(content_loss)) * 0.001

            pic_loss=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self._real_pic,predict_pic),name='pic_loss'),[1,2,3]))
            pic_square_loss=tf.reduce_mean(pic_loss,name='l2_mean_loss_pic')

        with tf.variable_scope('loss_function_txt'):
            # with tf.device('/device:GPU:1'):
            # Our targets are decoder inputs shifted by one.
            # targets = [self.decoder_inputs[i + 1]
            #           for i in xrange(len(self.decoder_inputs) - 1)]
            _, labels = tf.split(self._response, [1, -1], 1)  # sign 'go' for train decoder
            labels = tf.concat([labels, _], axis=1)  # remove 'go' to compute loss

            cross_entropy_sentence = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response_txt,
                                                                                    labels=labels,
                                                                                    name="cross_entropy_sents")
            cross_entropy_sentence = tf.multiply(cross_entropy_sentence, self._weight)  # batch_size * sents_size

            cross_entropy_sentence = tf.reduce_sum(cross_entropy_sentence, axis=1)
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            cross_entropy_sentence = cross_entropy_sentence / weight_sum
            txt_loss = tf.reduce_mean(cross_entropy_sentence, name="cross_entropy_sentences")

        # all_loss = pic_loss + 0.0*txt_loss
        # pdb.set_trace()
        all_loss=0.2*pic_square_loss+0.3*vgg_loss+0.5*txt_loss
        # loss_=tf.concat([tf.expand_dims(G_loss,-1),tf.expand_dims(cross_entropy_sentence,-1)],1)
        # all_loss=linear(loss_,1)

        self.loss = all_loss
        # pdb.set_trace()
        grads_and_vars = self._opt.compute_gradients(all_loss)
        # pdb.set_trace()
        # grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        # grads_and_vars = [(self.add_gradient_noise(g), v) for g, v in grads_and_vars]

        self.train_op = self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')
        # pdb.set_trace()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with tf.variable_scope('output_information'):
            self.predict_pic = predict_pic
            self.predict_txt = tf.argmax(response_txt, axis=2)

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
        self._real_pic = tf.placeholder(tf.float32, [self._batch_size, self.img_size_x, self.img_size_y, 1],
                                        name='frame_output')
        # self._random_z=tf.placeholder(tf.float32,[self._batch_size,self._noise_dim],name='noise')

    def steps(self, sess, data_dict, noise, step_type='train'):
        self.model_type = step_type
        input_batch_txt = data_dict[0]
        output_batch_txt = data_dict[1]
        input_batch_pic = data_dict[2]
        output_batch_pic = data_dict[3]
        weight_batch_txt = data_dict[4]
        # pdb.set_trace()
        feed_dict = {self._response: output_batch_txt,
                     self._question: input_batch_txt,
                     self._weight: weight_batch_txt,
                     self._input_pic: input_batch_pic,
                     self._real_pic: output_batch_pic}
        # self._random_z:noise}

        if step_type == 'train':
            output_list = [self.loss, self.train_op]
            try:
                loss, _ = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, _

        if step_type == 'test':
            output_list = [self.loss, self.predict_pic, self.predict_txt]
            try:
                loss, pic, txt = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, pic, txt

        print('step_type is wrong!>>>')
        return None
