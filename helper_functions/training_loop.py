'''
This File consit of the main trainig loop for this model
Refer to the notebook  to see how to use these functions are used.

'''

from __future__ import absolute_import, division, print_function, unicode_literals

import io
import os
import re
import string
import time
import unicodedata
import pandas as pd
from unicodedata import normalize

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from sklearn.model_selection import train_test_split
import seaborn as sns

# DEFINES BATCH TRAINING PROCESS


def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # dec_input = tf.expand_dims([1]*BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# MODEL TRAINING
EPOCHS = 50
array_epochs, array_losses = [], []
for epoch in range(EPOCHS):

    array_epochs.append(epoch)

    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every epoch
    checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch_{} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    array_losses.append(total_loss / steps_per_epoch)
    print('Time taken for epoch_{} : {} sec\n'.format(
        epoch + 1, time.time() - start))

np.save("all_epoch_en_lmb_{}".format(EPOCHS), np.array(array_epochs))
np.save("all_losses_en_lmb_{}".format(EPOCHS), np.array(array_losses))
