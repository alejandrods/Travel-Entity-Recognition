# -*- coding: utf-8 -*-

import csv
import logging
import numpy as np
import string
import itertools
import pickle

logging.basicConfig(level=logging.INFO,
                    format='TRAIN - %(asctime)s :: %(levelname)s :: %(message)s')


def convert(directory_of_dataset, directory_to_save_sentences, directory_to_save_labels):
    """
    Function to split the original data set into queries and labels.
    The converted result for the next sentence is:
    >> What's the airport at [orlando CITY_NAME]
    >> Query: ['what', "'s", 'the', 'airport', 'at', 'orlando']
    >> Label: ['O', 'O', 'O', 'O', 'O', 'B-CITY_NAME']

    :param directory_of_dataset: Directory of our dataset
    :param directory_to_save_sentences: Converted sentences directory
    :param directory_to_save_labels: Converted labels directory
    :return:
    """

    logging.info('Converting format...This can take a while')

    # Load data set to convert
    lines = []
    with open(directory_of_dataset, encoding='latin1') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # Get sentences
    sentences = []
    for line in lines:
        sentences.append(line[0])

    # Get tokens between brackets
    spaces_sentes = []
    for sent in sentences:
        
        new_sent = " ".join(sent.split())
    
        new_sent = new_sent.split(' ')
        original_sent = new_sent
        # new_sentences = []
        for tok in range(len(new_sent)):
            item = new_sent[tok]
            # original = new_sent[tok]
            
            if '[' == item:
                next_item = new_sent[tok+1]
                item_join = item + next_item
                original_sent[tok] = item_join
                original_sent[tok+1] = ' '
    
            if ']' == item:
                last_item = new_sent[tok-1]
                item_join = last_item + item
                original_sent[tok] = ' '
                original_sent[tok-1] = item_join
    
            if '][' in item:
                for i in range(len(item)):
                    if item[i] == ']':
                        word = item[:i+1]
                        original_sent[tok] = word

                    if item[i] == '[':
                        word = item[i:]
                        original_sent.insert(tok+1, word)

        original_sent = " ".join(original_sent)
        spaces_sentes.append(original_sent.split())

        # Special cases
        cleared_sentences = []
        for sent in spaces_sentes:
            delete = []
            for items in sent:
                if '[' in items:
                    if items[0] != '[':
                        while items[0] != '[':
                            items = items.replace(items[0], "")
                if ']' in items:
                    if items[-1] != ']':
                        while items[-1] != ']':
                            items = items.replace(items[-1], "")
               
                delete.append(items)
        
            cleared_sentences.append(' '.join(delete))
            
    sentences = cleared_sentences

    # Take the tags
    tokens_sent = []
    tags_sent = []

    for sentence in sentences:
        result = [p.split(']')[0] for p in sentence.split('[') if ']' in p]
        tokens = []
        tag = []

        for items in result:
            items = items.split(' ')

            if len(items) == 2:
                tokens.append(items[0])
                tag.append(items[1])
            if len(items) > 2:
                token = items[:len(items)-1]
                tokens.append(' '.join(token))
                tag.append(items[-1])
        
        tokens_sent.append(tokens)
        tags_sent.append(tag)

    # Create tags and tokens list
    start_words_sentences = []
    end_words_sentences = []
    
    for items in sentences:
        items = items.split(' ')
        empty_string = items.count('')

        for i in range(empty_string):
            new_item = items
            if "" in new_item:
                new_item.remove('')
            items = new_item

        # new_array = np.empty(len(items))
        j = 0
        start_words = []
        end_words = []

        for words in items: 
            start = []

            if words[:1] == '[':
                start.append(j)
                end = []
                k = j

                for words2 in items[j:]:
                    if words2[-1] == ']':
                        end.append(k)
                        break
                    k += 1
                    
                end_words.append(end)
                start_words.append(start)

            j += 1
        start_words_sentences.append(start_words)
        end_words_sentences.append(end_words)

    # Order the sentences
    new_sentences = []
    i = 0
    
    for items in sentences:
        # label_array = np.zeros(len(items))
    
        items = items.split(' ')
        right_tokens = tokens_sent[i]
        # right_tags = tags_sent[i]

        index_start = []
        for index in start_words_sentences[i]:
            index_start.append(index[0])
            
        # Take the end index
        index_end = []
        for index in end_words_sentences[i]:
            index_end.append(index[0])
        
        # Replace tokens
        k = 0
        for start1 in index_start:
            items[start1] = right_tokens[k]
                
            k += 1

        # Eliminate tags
        items = [i for j, i in enumerate(items) if j not in index_end]
        
        new_start_index = []
        new_end_index = []
        
        w = 0
        for numbers_start, numbers_end in zip(index_start, index_end):
            if w == 0:
                new_start_index.append(numbers_start)
                new_end_index.append(numbers_end)
                
                w += 1
            else:
                new_start_index.append(numbers_start-w)
                new_end_index.append(numbers_end-w)
                
                w += 1
    
        i += 1
        new_sentence = items

        z = 0
        for start, end, tokens in zip(new_start_index, new_end_index, right_tokens):
            tokens = tokens.split(' ')
            len_tokens = len(tokens)

            dif = end - start

            if dif > 1:
                # Number of tokens to delete
                delete_words_index = np.arange(0, len_tokens-1)

                last_sentences = new_sentence

                for items in delete_words_index:
                    last_sentences[items+start+1] = ''
                
                new_sentence = last_sentences
    
            z += 1
            
        join = ' '.join(new_sentence)
        new_sentences.append(join.split(' '))

    # Clean final sentences
    sentences_cleaned = []
    for items in new_sentences:
        new_sentence = []
    
        for tokens in items:
            if tokens not in list(string.punctuation + '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
                new_sentence.append(tokens)        
                
        sentences_cleaned.append([x for x in new_sentence if x])

    # Order Labels
    labels = []
    i = 0
    
    for sentence in sentences_cleaned:
        search_token = tokens_sent[i]
        target_label = tags_sent[i]

        # Join each tag
        w = 0
        new_target = []
        for items in search_token:
            items = items.split(' ')

            if len(items) > 1:
                a = 'B-'
                b = 'I-'
                tags = [target_label[w]]*len(items)

                for items2 in range(len(tags)):
                    if items2 == 0:
                        tag = a+tags[items2]
                        tags[items2] = tag
                    else:
                        tag = b+tags[items2]
                        tags[items2] = tag
                
                new_target.append(tags)      
       
            else:
                tags = ['B-'+target_label[w]]
                new_target.append(tags)
    
            w += 1
                    
        # New target label
        target_label = list(itertools.chain.from_iterable(new_target))

        lab = ['O'] * len(sentence)
        
        # Join each token
        new_search_token = []
        for items in search_token:
            items = items.split(' ')
            for items2 in items:
                new_search_token.append(items2)

        # Search the tokens in sentences
        search_token = new_search_token
        for items in range(len(search_token)):
            k = 0
            for tokens in sentence:
                if search_token[items] == tokens:
                    lab[k] = target_label[items]
                k += 1
        labels.append(lab)
        i += 1

    # Save data
    logging.info('Format Converted')
    
    with open(directory_to_save_sentences, "wb") as fp:
        pickle.dump(sentences_cleaned, fp)
    
    with open(directory_to_save_labels, "wb") as fp:
        pickle.dump(labels, fp)
    
    return
