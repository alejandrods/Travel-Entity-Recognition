import os
import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk import word_tokenize

from dotenv import load_dotenv
load_dotenv()

# Run prediction on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Flask creates a new threads which will generate their own Tensorflow session
graph = tf.get_default_graph()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')


MODEL_DIR = os.environ.get('MODEL_DIR')

logging.info('Loading model and tokenizer from: {}'.format(MODEL_DIR))

model = load_model(os.path.join(MODEL_DIR, 'model.h5'))
tokenizer = pickle.load(open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), "rb"))
labels2idx = pickle.load(open(os.path.join(MODEL_DIR, 'labels2idx.pkl'), "rb"))

words2idx = tokenizer.word_index

idx2words = {words2idx[k]: k for k in words2idx}
idx2labels = {labels2idx[k]: k for k in labels2idx}

logging.info('Loaded correctly')


def prediction(query):
    """
    Function to extract the entities from a query
    :param query: User sentence
    :return: Sentence with the entities
    """
    global graph
    with graph.as_default():
        # Tokenizer sentence
        tok_ls = word_tokenize(query.lower())

        # Convert to idx
        tok_idx = []
        for el in tok_ls:
            if el in words2idx:
                tok_idx.append(words2idx[el])
            else:
                tok_idx.append(words2idx['UNK'])

        # Reshape this array as same before
        reshape_tok_ls = np.array(tok_idx)[np.newaxis, :]

        # Prediction
        pred = model.predict(reshape_tok_ls)

        # Take the best result
        pred_max = np.argmax(pred, -1)[0]

        # Show the decoding prediction
        pred_decode = []
        for el in pred_max:
            pred_decode.append(idx2labels.get(el))

        logging.info('Prediction decode: {}'.format(pred_decode))

        labels_decode = []
        tokens_decode = []
        for el1, el2 in zip(pred_decode, tok_ls):
            if el1 != 'O':
                labels_decode.append(el1)
                tokens_decode.append(el2)

        ext = ['O', 'O']
        labels_decode = labels_decode + ext
        tokens_decode = tokens_decode + ext

        for i in range(len(labels_decode)):
            if labels_decode[i] != 'O':
                # print(labels_decode[i])
                item = labels_decode[i]
                next_item = labels_decode[i + 1]
                next_next_item = labels_decode[i + 2]

                if item[:2] != 'I-' and next_item[:2] != 'I-':
                    logging.info('{} : {}'.format(labels_decode[i], tokens_decode[i]))

                if item[:2] != 'I-' and next_item[:2] == 'I-' and next_next_item[:2] != 'I-':
                    logging.info('B+I {}: {} {}'.format(labels_decode[i], tokens_decode[i], tokens_decode[i + 1]))

                if item[:2] == 'I-' and next_item[:2] == 'I-' and next_next_item[:2] != 'I-':
                    logging.info('B+I {}: {} {} {}'.format(labels_decode[i - 1], tokens_decode[i - 1],
                                                           tokens_decode[i], tokens_decode[i + 1]))

        # Combine and return result
        result_sent = []
        tokens = tok_ls + ext
        labels = pred_decode + ext

        for items in range(len(labels)):
            if labels[items] != 'O':
                before_item = labels[items - 1]
                item = labels[items]
                next_item = labels[items + 1]
                next_next_item = labels[items + 2]
                if item[:2] != 'I-' and next_item[:2] != 'I-':
                    a = ' '.join(['[', tokens[items], item, ']'])
                    result_sent.append(a)

                if item[:2] == 'I-' and next_item[:2] == 'I-' and next_next_item[:2] != 'I-':
                    a = ' '.join(['[', tokens[items - 1], tokens[items], tokens[items + 1], before_item, ']'])
                    result_sent.append(a)

                if item[:2] != 'I-' and next_item[:2] == 'I-' and next_next_item[:2] != 'I-':
                    a = ' '.join(['[', tokens[items], tokens[items + 1], item, ']'])
                    result_sent.append(a)
            else:
                result_sent.append(tokens[items])

        return ' '.join(result_sent[:-2])
