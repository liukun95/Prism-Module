import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from data import Data
from model import Model
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

config = {'max_step': 400,
          'step_size': 220,
          'rnn_size': 200,
          'embedding_size': 100,
          'char_embedding_size': 30,
          'batch_size': 64,
          'max_length': 2000,
          'abstract_word_size': 8,
          'orthogonal': True,
          }
config = OrderedDict(config)


def main(_):
    max_step = config['max_step']
    step_size = config['step_size']
    rnn_size = config['rnn_size']
    embedding_size = config['embedding_size']
    char_embedding_size = config['char_embedding_size']
    batch_size = config['batch_size']
    max_length = config['max_length']
    abstract_word_size = config['abstract_word_size']
    orthogonal = config['orthogonal']

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=False)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    data = Data(pretrain_embedding='glove.6B/glove.6B.100d.txt')
    model = Model(rnn_size=rnn_size,
                  embedding_size=embedding_size,
                  char_embedding_size=char_embedding_size,
                  max_length=max_length,
                  embedding_W=data.embedding,
                  vocab_num=data.vocab_num,
                  label_num=data.label_num,
                  char_num=data.char_num,
                  vocab_map=data.vocab_map,
                  label_map=data.label_map,
                  char_map=data.char_map,
                  abstract_word_size=abstract_word_size,
                  orthogonal=orthogonal,
                  )

    tf.global_variables_initializer().run()

    best_dev_loss_global = 1000000
    best_dev_f_global = 0
    best_dev_p = 0
    best_dev_r = 0
    best_dev_f = 0
    best_test_p = 0
    best_test_r = 0
    best_test_f = 0
    step = 0
    while step < max_step:
        start_time = time.time()
        step += 1
        train_loss = 0
        for _ in tqdm(range(step_size)):
            x_batch, y_batch = data.get_train_batch(batch_size)
            train_loss += model.train(sess, x_batch, y_batch)
        model.epoch += 1
        train_loss = train_loss / step_size
        dev_loss, dev_precision, dev_recall, dev_f1 = model.test(sess, *data.get_dev_batches(batch_size))
        test_loss, test_precision, test_recall, test_f1 = model.test(sess, *data.get_test_batches(batch_size))

        if best_dev_loss_global > dev_loss or best_dev_f_global < dev_f1:
            best_dev_p = dev_precision
            best_dev_r = dev_recall
            best_dev_f = dev_f1
            best_test_p = test_precision
            best_test_r = test_recall
            best_test_f = test_f1

        if best_dev_loss_global > dev_loss:
            best_dev_loss_global = dev_loss
        if best_dev_f_global < dev_f1:
            best_dev_f_global = dev_f1

        log_txt = [
            '  Step: {0:d} / {1:d}'.format(step, max_step),
            '  Step Time: {0:.8f}'.format(time.time() - start_time),
            '  Train Loss: {0:.8f}'.format(train_loss),
            '  Dev  Loss: {0:.8f}\t Precision: {1:.8f}\t Recall: {2:.8f}\t F1: {3:.8f}'.format(dev_loss, dev_precision,
                                                                                               dev_recall, dev_f1),
            '  Test Loss: {0:.8f}\t Precision: {1:.8f}\t Recall: {2:.8f}\t F1: {3:.8f}'.format(test_loss,
                                                                                               test_precision,
                                                                                               test_recall, test_f1),
            '  ################',
            '  Best Dev Precision: {0:.8f}\t Recall: {1:.8f}\t F1: {2:.8f}'.format(best_dev_p, best_dev_r, best_dev_f),
            '  Best Test Precision: {0:.8f}\t Recall: {1:.8f}\t F1: {2:.8f}'.format(best_test_p, best_test_r,
                                                                                    best_test_f),
            '  ################',
        ]

        for t in log_txt:
            tf.logging.info(t)
        tf.logging.info('')

        # It should be noted that CNN-BiLSTM-CRF model sometimes starts at a trivial location which makes the training
        # hard, and it needs to be restart manually. We find this phenomenon not only in the version we reproduce but
        # also in the several popular versions on GitHub.
        if dev_f1 < 0.85:
            tf.logging.info('  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            tf.logging.info('  A trivial location. Restart...')
            tf.logging.info('  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            tf.global_variables_initializer().run()
            model.epoch = 0

            best_dev_loss_global = 1000000
            best_dev_f_global = 0
            best_dev_p = 0
            best_dev_r = 0
            best_dev_f = 0
            best_test_p = 0
            best_test_r = 0
            best_test_f = 0
            step = 0

            continue


if __name__ == '__main__':
    tf.app.run()
