# -*- coding: utf-8 -*-

import os
import random
import logging
import numpy as np
from dotenv import load_dotenv

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM

from utils.convert2format import convert
load_dotenv()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


random.seed(42)

logging.basicConfig(level=logging.INFO,
                    format='TRAIN - %(asctime)s :: %(levelname)s :: %(message)s')

DATA_PATH = os.environ.get('DATA_PATH')
CONVERT_PATH = os.environ.get('CONVERT_PATH')
DATASET_FILE = os.environ.get('DATASET_FILE')
QUERY_FILE = os.environ.get('QUERY_FILE')
LABEL_FILE = os.environ.get('LABEL_FILE')
GLOVE_DIR = os.environ.get('GLOVE_DIR')
EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM'))
MAX_SEQ_LEN = int(os.environ.get('MAX_SEQ_LEN'))
MODEL_DIR = os.environ.get('MODEL_DIR')

convert_path = os.path.join(DATA_PATH, CONVERT_PATH)
if not os.path.exists(convert_path):
    os.mkdir(convert_path)

dir_dataset = os.path.join(DATA_PATH, DATASET_FILE)
dir2save_query = os.path.join(convert_path, QUERY_FILE)
dir2save_label = os.path.join(convert_path, LABEL_FILE)

logging.info('Data set dir: {}'.format(dir_dataset))
logging.info('Query dir: {}'.format(dir2save_query))
logging.info('Label dir: {}'.format(dir2save_label))
logging.info('Glove dir: {} / Embedding Dim: {}'.format(GLOVE_DIR, EMBEDDING_DIM))

# Convert format
if not os.path.exists(dir2save_query):
    convert(dir_dataset, dir2save_query, dir2save_label)

logging.info('Loading query and labels files')

# Load files
with open(dir2save_query, "rb") as fp:
    sentences = pickle.load(fp)

with open(dir2save_label, "rb") as fp:
    labels = pickle.load(fp)
    labels_original = labels

logging.info('Data loaded')
logging.info('Sentence: {}'.format(sentences[10]))
logging.info('Label: {}'.format(labels[10]))

# Join sentences
sentences = [' '.join(sent) for sent in sentences]

# Set of all entities
entities = [y for x in labels for y in x]
tags = list(set(entities))

# Create dictionary for labels
idx = np.arange(0, len(tags))
labels2idx = dict(zip(tags, idx))

logging.info('\tLabels2index: {}'.format(labels2idx))

# Convert list of labels into index_labels
logging.info('Convert list of labels into index_labels:')

labels_idx = []
for label in labels:
    tag = []
    for tags in label:
        index = labels2idx.get(tags)
        tag.append(index)
    labels_idx.append(tag)

logging.info('\tlabels_idx: {}'.format(labels_idx[0]))
logging.info('\tlabels: {}'.format(labels[0]))

# Tokenizer
tokenizer = Tokenizer(num_words=20000, split=' ', oov_token='UNK')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Add UNK key in words2index for unknown words
logging.info('Creating words2index')

words2idx = tokenizer.word_index

n_classes = len(labels2idx)
n_vocab = len(words2idx)

logging.info('Number of labels: {}'.format(n_classes))
logging.info('Number of words: {}'.format(n_vocab))

# Load and prepare embedding
logging.info('Loading Glove...')

# Open embedding file
f = open(GLOVE_DIR, encoding='utf-8')

embeddings_index = {}
words_glove = []
for line in f:
    values = line.split()
    word = values[0]
    words_glove.append(word)
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

for tok in words_glove:
    if tok not in words2idx:
        words2idx.update({tok.lower(): list(words2idx.values())[-1] + 1})

logging.info('len: {}'.format(len(words2idx)))
logging.info('Creating Embedding Matrix...')

i = 0
empty = []
embedding_matrix = np.random.random((len(words2idx) + 1, EMBEDDING_DIM))
for word, i in words2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        empty.append(i)
        i += 1

logging.info('Embedding Matrix created')

# Pad Sequences
data_train = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)

# Create matrix with labels one-hot
labels_train = []
for items in labels_idx:
    label = items
    label = np.eye(n_classes)[items]
    labels_train.append(label)

# Apply pad sequences to each labels
labels_train = pad_sequences(labels_train, maxlen=MAX_SEQ_LEN)

x_train = data_train
y_train = labels_train

logging.info('Shape of x_train: {}'.format(x_train.shape))
logging.info('Shape of y_train: {}'.format(y_train.shape))


# Define our model
def model():
    model = Sequential()
    model.add(Embedding(len(words2idx) + 1, EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(300, return_sequences=True)))
    model.add(Dense(n_classes, activation='softmax'))

    return model


logging.info('Compiling model')
model = model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

logging.info('Training model')
model.summary()
hist = model.fit(x_train, y_train,
                 validation_split=0.1,
                 nb_epoch=20,
                 batch_size=64)

# Save model
logging.info('Save model in: {}'.format(MODEL_DIR))

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

model.save(os.path.join(MODEL_DIR, 'model.h5'))

with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(MODEL_DIR, 'labels2idx.pkl'), 'wb') as f:
    pickle.dump(labels2idx, f, pickle.HIGHEST_PROTOCOL)

logging.info('Model saved')
