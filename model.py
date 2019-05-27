import re

import tensorflow as tf
from entity_performance import performance


class Model:
    def __init__(self, rnn_size, embedding_size, char_embedding_size, max_length, embedding_W,
                 vocab_num, label_num, char_num,
                 vocab_map, label_map, char_map,
                 abstract_word_size, orthogonal):
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.max_length = max_length
        self.embedding_W = embedding_W
        self.vocab_num = vocab_num
        self.label_num = label_num
        self.char_num = char_num
        self.vocab_map = vocab_map
        self.label_map = label_map
        self.char_map = char_map
        self.abstract_words_size = abstract_word_size
        self.is_orthogonal = orthogonal
        self.epoch = 0

        self.filter_num = 30

        # initialize W in fully connected layer.
        def fcl_weight_variable(shape):
            initial = tf.random_uniform(shape,
                                        -tf.sqrt(6 / (shape[0] + shape[1])),
                                        tf.sqrt(6 / (shape[0] + shape[1])))
            fcl_W = tf.Variable(initial, name='fcl_W')
            return fcl_W

        # initialize bias in fully connected layer.
        def fcl_bias_variable(shape):
            initial = tf.zeros(shape=shape)
            fcl_b = tf.Variable(initial, name='fcl_b')
            return fcl_b

        # initialize W in CNN.
        def conv_weight_variable(shape):
            initial = tf.random_uniform(shape, -tf.sqrt(6 / (tf.reduce_sum(tf.cast(shape, tf.float32)) + 1)),
                                        -tf.sqrt(6 / (tf.reduce_sum(tf.cast(shape, tf.float32)) + 1)))
            conv_W = tf.Variable(initial, name='conv_W', dtype=tf.float32)
            return conv_W

        # initialize bias in CNN.
        def conv_bias_variable(shape):
            initial = tf.zeros(shape=shape)
            conv_b = tf.Variable(initial, name='conv_b', dtype=tf.float32)
            return conv_b

        # compute convolution.
        def conv1d(x, conv_W, conv_b):
            conv = tf.nn.conv1d(x,
                                conv_W,
                                stride=1,
                                padding='SAME',
                                name='conv')
            h = tf.nn.tanh(tf.nn.bias_add(conv, conv_b), name='relu')
            return h

        # max-pooling.
        def max_pool(x):
            return tf.reduce_max(x, axis=1)

        self.x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.x_char = tf.placeholder(tf.int32, [None, None], name='input_x_char')
        self.y_ = tf.placeholder(tf.int32, [None, None], name='input_y')
        self.sl = tf.placeholder(tf.int32, [None], name='input_sl')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            if self.embedding_W is None:
                embedding_table = tf.Variable(
                    tf.random_uniform([vocab_num, self.embedding_size], -tf.sqrt(3 / self.embedding_size),
                                      tf.sqrt(3 / self.embedding_size)),
                    name='embedding_table',
                    trainable=True)
            else:
                embedding_table = tf.Variable(self.embedding_W,
                                              name='embedding_table',
                                              trainable=True)
            embedded_words = tf.nn.embedding_lookup(embedding_table, self.x)

        with tf.device('/cpu:0'), tf.variable_scope('char_embedding'):
            embedding_table = tf.Variable(
                tf.random_uniform([char_num, self.char_embedding_size], -tf.sqrt(3 / self.char_embedding_size),
                                  tf.sqrt(3 / self.char_embedding_size)),
                name='embedding_table',
                trainable=True)
            embedded_chars = tf.nn.embedding_lookup(embedding_table, self.x_char)

        embedded_chars = tf.nn.dropout(embedded_chars, keep_prob=self.keep_prob)

        batch_size = tf.shape(embedded_words)[0]
        seq_length = tf.shape(embedded_words)[1]

        with tf.variable_scope('char_CNN'):
            cnn_input = tf.reshape(embedded_chars, [batch_size * seq_length, -1, self.char_embedding_size])
            filter_shape = [3, self.char_embedding_size, self.filter_num]
            conv_W = conv_weight_variable(filter_shape)
            conv_b = conv_bias_variable([self.filter_num])
            conv = conv1d(cnn_input, conv_W, conv_b)
            pooled = max_pool(conv)
            embedded_chars = tf.reshape(pooled, [batch_size, seq_length, self.filter_num])

        embedded_words = tf.concat([embedded_words, embedded_chars], axis=-1)

        embedded_words = tf.nn.dropout(embedded_words, keep_prob=self.keep_prob)

        with tf.variable_scope('extract_words'):
            abstract_embedding_table = tf.Variable(
                tf.random_uniform([self.abstract_words_size, self.embedding_size + self.filter_num],
                                  -tf.sqrt(3 / (self.embedding_size + self.filter_num)),
                                  tf.sqrt(3 / (self.embedding_size + self.filter_num))),
                name='abstract_embedding_table',
                trainable=True)

            abstract_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                           num_units=self.rnn_size,
                                                           direction='bidirectional',
                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                           bias_initializer=tf.zeros_initializer())
            rnn_output, _ = abstract_lstm(tf.transpose(embedded_words, [1, 0, 2]))
            rnn_output = tf.transpose(rnn_output, [1, 0, 2])

            rnn_output = tf.nn.dropout(rnn_output, keep_prob=self.keep_prob)

            extracted_words_logits = tf.layers.dense(rnn_output, self.abstract_words_size,
                                                     kernel_initializer=tf.random_uniform_initializer(
                                                         -tf.sqrt(6 / (2 * self.rnn_size + self.abstract_words_size)),
                                                         tf.sqrt(6 / (2 * self.rnn_size + self.abstract_words_size))))
            extracted_words_probs = tf.nn.sigmoid(extracted_words_logits)

            self.extracted_words = tf.round(extracted_words_probs)
            self.extracted_words_probs = extracted_words_probs

            extracted_mask = tf.floor(extracted_words_probs + tf.random_uniform(tf.shape(extracted_words_probs), 0, 1))
            extracted_words = tf.expand_dims(extracted_mask, axis=3) * tf.expand_dims(
                tf.expand_dims(abstract_embedding_table, axis=0), axis=0)
            extracted_words = tf.nn.dropout(extracted_words, keep_prob=self.keep_prob)
            extracted_words += (1 - tf.expand_dims(extracted_mask, axis=3)) * tf.expand_dims(embedded_words, axis=2)

        with tf.variable_scope('BiLSTM_CRF'):
            self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                       num_units=self.rnn_size,
                                                       direction='bidirectional',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       bias_initializer=tf.zeros_initializer())
            self.crf_W = fcl_weight_variable([2 * self.rnn_size, self.label_num])
            self.crf_b = fcl_bias_variable([self.label_num])
            self.transition_params = tf.get_variable('transition_params', [self.label_num, self.label_num])

            all_examples = tf.concat([tf.expand_dims(embedded_words, axis=2), extracted_words], axis=2)
            all_examples_shape = tf.shape(all_examples)  # [b, l, n, e]
            per_choice_output, _ = self.lstm(
                tf.reshape(
                    tf.transpose(all_examples, [1, 0, 2, 3]),  # [l, b, n, e]
                    [all_examples_shape[1], all_examples_shape[0] * all_examples_shape[2],
                     self.embedding_size + self.char_embedding_size])  # [l, b * n, e]
            )  # [l, b * n, e]
            per_choice_output = tf.reshape(per_choice_output,
                                           [all_examples_shape[1], all_examples_shape[0], all_examples_shape[2],
                                            self.rnn_size * 2])  # [l, b, n, e]
            per_choice_output = tf.transpose(per_choice_output, [1, 2, 0, 3])  # [b, n, l, e]
            per_choice_logits = tf.reshape(per_choice_output, [-1, self.rnn_size * 2]) @ self.crf_W + self.crf_b
            per_choice_logits = tf.reshape(per_choice_logits, [-1, all_examples_shape[1], self.label_num])
            per_choice_logits = tf.nn.dropout(per_choice_logits, keep_prob=self.keep_prob)
            y_ = tf.reshape(tf.tile(tf.expand_dims(self.y_, axis=1), [1, self.abstract_words_size + 1, 1]),
                            [-1, all_examples_shape[1]])
            lengths = tf.reshape(tf.tile(tf.expand_dims(self.sl, axis=1), [1, self.abstract_words_size + 1]), [-1])

            per_choice_loss, _ = tf.contrib.crf.crf_log_likelihood(per_choice_logits,
                                                                   y_,
                                                                   lengths,
                                                                   self.transition_params)
            per_choice_loss = tf.reshape(-per_choice_loss, [-1, self.abstract_words_size + 1])
            origin_loss, abstract_words_loss = tf.split(per_choice_loss,
                                                        num_or_size_splits=[1, self.abstract_words_size], axis=1)

        self.reward = tf.stop_gradient(abstract_words_loss + origin_loss)
        reward_mean, reward_var = tf.nn.moments(self.reward, [0], keep_dims=True)
        self.reward = tf.nn.batch_normalization(self.reward, reward_mean, reward_var, None, None, 1e-8)

        self.abstract_words_loss = tf.reduce_mean(
            tf.reduce_mean(tf.expand_dims(self.reward, axis=1)
                           * (
                           extracted_words_probs * extracted_mask + (1 - extracted_words_probs) * (1 - extracted_mask)),
                           axis=1)
            + abstract_words_loss, axis=-1)

        origin_loss = tf.squeeze(origin_loss, axis=1)
        self.loss = tf.reduce_mean(origin_loss + self.abstract_words_loss)

        if self.is_orthogonal:
            abstract_embedding_table_normalized = tf.math.l2_normalize(abstract_embedding_table, axis=-1)
            self.loss_orthogonal = abstract_embedding_table_normalized @ tf.transpose(
                abstract_embedding_table_normalized, [1, 0])
            self.loss_orthogonal = self.loss_orthogonal * (1 - tf.eye(self.abstract_words_size))
            self.loss_orthogonal = tf.reduce_mean(self.loss_orthogonal)
            self.loss += self.loss_orthogonal

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [((tf.clip_by_norm(grad, 3)), var) for grad, var in gvs]
        self.train_step = optimizer.apply_gradients(capped_gvs)

        lstm_output, _ = self.lstm(tf.transpose(embedded_words, [1, 0, 2]))
        lstm_output = tf.transpose(lstm_output, [1, 0, 2])
        logits = tf.reshape(lstm_output, [-1, self.rnn_size * 2]) @ self.crf_W + self.crf_b
        logits = tf.reshape(logits, [tf.shape(lstm_output)[0], tf.shape(lstm_output)[1], self.label_num])
        self.tags, _ = tf.contrib.crf.crf_decode(logits,
                                                 self.transition_params,
                                                 self.sl)

    @staticmethod
    def padding(x):
        l = max([len(i) for i in x])
        new_x = []
        for i in range(len(x)):
            new_x.append([j for j in x[i]])
            while len(new_x[i]) < l:
                new_x[i].append(0)
        return new_x

    def train(self, sess, xs, ys):
        sl_batch = [min(len(x), self.max_length) for x in xs]
        x_batch = self.padding([[self.vocab_map['word2id'][i.lower()] for i in x[:self.max_length]] for x in xs])
        y_batch = self.padding([[self.label_map['label2id'][i] for i in y[:self.max_length]] for y in ys])

        x_char_batch = []
        for j in range(len(x_batch)):
            for i in range(len(x_batch[j])):
                if x_batch[j][i] == 0:
                    x_char_batch.append([0])
                else:
                    x_char_batch.append([self.char_map['char2id'][c] for c in xs[j][i]])
        x_char_batch = self.padding([[0, 0] + x + [0, 0] for x in x_char_batch])

        r = sess.run([self.train_step, self.loss], feed_dict={self.x: x_batch,
                                                              self.x_char: x_char_batch,
                                                              self.y_: y_batch,
                                                              self.sl: sl_batch,
                                                              self.keep_prob: 0.5,
                                                              self.lr: 0.1 / (1 + 0.05 * self.epoch)
                                                              })
        return r[1]

    def test(self, sess, x, y):
        prediction = []
        labels = []
        test_loss = 0
        for n in range(len(x)):
            sl_batch = [min(len(x), self.max_length) for x in x[n]]
            x_batch = self.padding([[self.vocab_map['word2id'][i.lower()] for i in x[:self.max_length]] for x in x[n]])
            y_batch = self.padding([[self.label_map['label2id'][i] for i in y[:self.max_length]] for y in y[n]])

            x_char_batch = []
            for j in range(len(x_batch)):
                for i in range(len(x_batch[j])):
                    if x_batch[j][i] == 0:
                        x_char_batch.append([0])
                    else:
                        x_char_batch.append([self.char_map['char2id'][c] for c in x[n][j][i]])
            x_char_batch = self.padding([[0, 0] + x + [0, 0] for x in x_char_batch])

            preds, loss = sess.run([self.tags, self.loss], feed_dict={self.x: x_batch,
                                                                      self.x_char: x_char_batch,
                                                                      self.y_: y_batch,
                                                                      self.sl: sl_batch,
                                                                      self.keep_prob: 1.0})
            prediction.extend(preds)
            labels.extend(y[n])
            test_loss += loss * len(x_batch)
        test_loss = test_loss / sum([len(i) for i in x])
        precision, recall, f1 = performance([[self.label_map['id2label'][j] for j in i] for i in prediction],
                                            labels)
        return test_loss, precision, recall, f1

    def get_abstract_words(self, sess, x, y, return_one=False):
        extracted_words_all = []
        for n in range(len(x)):
            sl_batch = [min(len(x), self.max_length) for x in x[n]]
            x_batch = self.padding([[self.vocab_map['word2id'][i.lower()] for i in x[:self.max_length]] for x in x[n]])
            y_batch = self.padding([[self.label_map['label2id'][i] for i in y[:self.max_length]] for y in y[n]])

            x_char_batch = []
            for j in range(len(x_batch)):
                for i in range(len(x_batch[j])):
                    if x_batch[j][i] == 0:
                        x_char_batch.append([0])
                    else:
                        x_char_batch.append([self.char_map['char2id'][c] for c in x[n][j][i]])
            x_char_batch = self.padding([[0, 0] + x + [0, 0] for x in x_char_batch])

            extracted_words, preds, extracted_words_probs = sess.run(
                [self.extracted_words, self.tags, self.extracted_words_probs], feed_dict={self.x: x_batch,
                                                                                          self.x_char: x_char_batch,
                                                                                          self.y_: y_batch,
                                                                                          self.sl: sl_batch,
                                                                                          self.keep_prob: 1.0})
            for i in range(len(x_batch)):
                es = []

                e = []
                for n in range(len(x_batch[i])):
                    if x_batch[i][n] > 0:
                        if y_batch[i][n] > 0 or preds[i][n] > 0:
                            e.append('[' + self.vocab_map['id2word'][x_batch[i][n]] + '|' + self.label_map['id2label'][
                                y_batch[i][n]] + '|' + self.label_map['id2label'][preds[i][n]] + ']')
                        else:
                            e.append(self.vocab_map['id2word'][x_batch[i][n]])
                es.append(' '.join(e) + '\n')

                for j in range(self.abstract_words_size):
                    e = []
                    p = []
                    for n in range(len(x_batch[i])):
                        if x_batch[i][n] > 0:
                            if extracted_words[i][n][j] > 0:
                                e.append('[' + self.vocab_map['id2word'][x_batch[i][n]] + ']')
                            else:
                                e.append(self.vocab_map['id2word'][x_batch[i][n]])
                            p.append(str(extracted_words_probs[i][n][j]))
                    es.append(' '.join(e) + '\n')
                    es.append(' '.join(p) + '\n')
                extracted_words_all.append(''.join(es))
            if return_one:
                return extracted_words_all[0]
        return extracted_words_all
