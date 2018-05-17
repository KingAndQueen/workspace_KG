import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import pdb


class seq2seq():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32,name='learn_rate')

        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        self._opt = tf.train.AdamOptimizer(learning_rate=self._learn_rate)

        self._embedding_size = config.neurons
        self._layers = config.layers
        self.model_type = config.model_type

        self._max_grad_norm = config.max_grad_norm
        self._build_inputs()


        with tf.variable_scope('embedding'):
            self._word_embedding = tf.get_variable(name='embedding_word',
                                                   shape=[self._vocab.vocab_size, config.neurons])
            self._name_embedding = tf.get_variable(name='embedding_name',
                                                   shape=[7 + 1, 0.5*config.neurons])

            speaker_embedding=embedding_ops.embedding_lookup(self._name_embedding, self._speaker)
            _Question = tf.unstack(self._question, axis=1)
            Question_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Question]
            _Response = tf.unstack(self._response, axis=1)
            response_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Response]

        def _encoding_roles(person_emb, name='', GPU_id=0):
            # with tf.device('/device:GPU:%d' %GPU_id):
            with tf.variable_scope('encoding_role_' + name):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neurons, reuse=tf.get_variable_scope().reuse)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                encoding_cell = tf.contrib.rnn.DropoutWrapper(encoding_cell, 0.8, 0.8, 0.8)
                # for future test
                # output, state_fw, state_bw = rnn.static_bidirectional_rnn(cell_fw=encoding_cell, cell_bw=encoding_cell,
                #                                                           inputs=person_emb, dtype=tf.float32)
                output, state_fw = rnn.static_rnn(encoding_cell, person_emb, dtype=tf.float32)
                # state = tf.concat([state_fw, state_bw], -1)
                # state = tf.matmul(state, tf.get_variable('Wi', [3, 2 * self._embedding_size, self._embedding_size],
                #                                          dtype=tf.float32, trainable=True))

                top_output = [array_ops.reshape(o, [-1, 1, encoding_single_layer.output_size]) for o in output]
                #pdb.set_trace()
                attention_states = array_ops.concat(top_output, 1)
                return  attention_states,state_fw

        question_output, question_state = _encoding_roles(Question_emb,'question_bi-encode')  # monica_sate.shape=layers*[batch_size,neurons]


        def speaker_atten(encoder_state, attention_states, ans_emb,speaker_embedding,model_type='train'):
            with tf.variable_scope('speaker'):
                num_heads = 3
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

                linear = rnn_cell_impl._linear
                batch_attn_size = array_ops.stack([batch_size, attn_size])
                attns = [array_ops.zeros(batch_attn_size, dtype=tf.float32) for _ in range(num_heads)]
                # pdb.set_trace()
                for a in attns:
                    a.set_shape([None, attn_size])

                with tf.variable_scope("rnn_decoder"):
                    single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    # $ cell_de = single_cell_de
                    # if self._layers > 1:
                    cell_de = tf.nn.rnn_cell.MultiRNNCell([single_cell_de] * self._layers)
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
                        inp = tf.concat([inp, speaker_embedding], 1)
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

        response = speaker_atten(question_state, question_output, response_emb,speaker_embedding,self.model_type)

        with tf.variable_scope('loss_function'):
            # with tf.device('/device:GPU:1'):
            # Our targets are decoder inputs shifted by one.
            # targets = [self.decoder_inputs[i + 1]
            #           for i in xrange(len(self.decoder_inputs) - 1)]
            _, labels = tf.split(self._response, [1, -1], 1)
            labels = tf.concat([labels, _], axis=1)

            cross_entropy_sentence = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response,
                                                                                    labels=labels,
                                                                                    name="cross_entropy_sents")
            cross_entropy_sentence = tf.multiply(cross_entropy_sentence, self._weight)  # batch_size * sents_size

            cross_entropy_sentence = tf.reduce_sum(cross_entropy_sentence, axis=1)
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            cross_entropy_sentence = cross_entropy_sentence / weight_sum
            cross_entropy_sentence_sum = tf.reduce_mean(cross_entropy_sentence, name="cross_entropy_sentences")
            self.loss=cross_entropy_sentence_sum

        grads_and_vars=self._opt.compute_gradients(cross_entropy_sentence_sum)
        # pdb.set_trace()
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(self.add_gradient_noise(g), v) for g, v in grads_and_vars]

        self.train_op = self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')
        self.response = tf.argmax(response, axis=2)

    def add_gradient_noise(self,t, stddev=1e-3, name=None):
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
        self._speaker= tf.placeholder(tf.int32, [self._batch_size], name='true_speaker')
    def get_batch(self, data_raw):
        # return a list of batches
        list_all_batch = []
        #    pdb.set_trace()
        for _ in range(0, len(data_raw), self._batch_size):
            if _ + self._batch_size > len(data_raw): continue
            data_batch = data_raw[_:_ + self._batch_size]
            Question,Response,weight,Speaker = [], [],[],[]
            for i in data_batch:
                Question.append(i.get('question'))
                Response.append(i.get('answer'))
                weight.append(i.get('weight'))
                Speaker.append(i.get('speaker'))
            list_all_batch.append({'Question': Question,'Response':Response,'weight': weight,'Speaker':Speaker})
        return list_all_batch

    def step(self, sess, data_dict, step_type='train'):
        self.model_type = step_type
        feed_dict = {self._response: data_dict['Response'],
                     self._question: data_dict['Question'],
                     self._weight:data_dict['weight'],
                     self._speaker:data_dict['Speaker']}
        if step_type == 'train':
            output_list = [self.loss, self.train_op]
            try:
                loss, _ = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, _

        if step_type == 'test':
            output_list = [self.loss, self.response]
            try:
                loss, response = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, response

        print('step_type is wrong!>>>')
        return None
