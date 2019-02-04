import os
import pickle
import re
from collections import OrderedDict

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

from ModelFactory import get_models_dir

SPACY_MAX_LENGTH = 3000000

class ExtractorLinguistic:
    @staticmethod
    def extract(lang, tweet, params) :
        nlp = spacy.load(lang)
        nlp.max_length = SPACY_MAX_LENGTH

        tweet_dict = OrderedDict()
        if lang == 'en':
            pos_dict = {
                'SPACE': 0,
                'NOUN': 0,
                'PART': 0,
                'PRON': 0,
                'INTJ': 0,
                'SYM': 0,
                'ADJ': 0,
                'CCONJ': 0,
                'PUNCT': 0,
                'X': 0,
                'VERB': 0,
                'ADP': 0,
                'ADV': 0,
                'PROPN': 0,
                'NUM': 0,
                'DET': 0
            }

            tense_dict = {
                'Past': 0,
                'Pres': 0
            }
            n_stopwords = 0
            
            for word in nlp(tweet['text']):
                # n_pos features
                pos = word.pos_
                pos_dict[pos] = pos_dict[pos] + 1
                # n_stopwords
                if word.is_stop:
                    n_stopwords += 1
                # n_tense features
                tag = word.tag_
                if tag in params['tense']['present']:
                    tense_dict['Pres'] = tense_dict['Pres'] + 1
                elif tag in params['tense']['past']:
                    tense_dict['Past'] = tense_dict['Past'] + 1

        for feature, params in params.items():
            if feature == 'tense':
                tweet_dict['n_tense_past'] = tense_dict['Past']
                tweet_dict['n_tense_pres'] = tense_dict['Pres']
            elif feature == 'pos_counts':
                tweet_dict['n_pos_space'] = pos_dict['SPACE']
                tweet_dict['n_pos_noun'] = pos_dict['NOUN']
                tweet_dict['n_pos_par'] = pos_dict['PART']
                tweet_dict['n_pos_pron'] = pos_dict['PRON']
                tweet_dict['n_pos_intj'] = pos_dict['INTJ']
                tweet_dict['n_pos_sym'] = pos_dict['SYM']
                tweet_dict['n_pos_adj'] = pos_dict['ADJ']
                tweet_dict['n_pos_conj'] = pos_dict['CCONJ']
                tweet_dict['n_pos_punct'] = pos_dict['PUNCT']
                tweet_dict['n_pos_x'] = pos_dict['X']
                tweet_dict['n_pos_verb'] = pos_dict['VERB']
                tweet_dict['n_pos_adp'] = pos_dict['ADP']
                tweet_dict['n_pos_adv'] = pos_dict['ADV']
                tweet_dict['n_pos_propn'] = pos_dict['PROPN']
                tweet_dict['n_pos_num'] = pos_dict['NUM']
                tweet_dict['n_pos_det'] = pos_dict['DET']
            elif feature == 'word_counts':
                tweet_dict['n_words'] = len(tweet['text'].split())
            elif feature == 'stopword_counts':
                tweet_dict['n_stopwords'] = n_stopwords

        return tweet_dict
