import os
import pickle
import re
from collections import OrderedDict

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

from model_factory import get_models_dir


class ExtractorBow:
    @staticmethod
    def extract(lang, tweet, params) :
        vocab_path = os.path.join(get_models_dir(lang), params['vocabulary'])
        vocab = pickle.load(open(vocab_path, 'rb'))
        bow_vectorizer = CountVectorizer(
            min_df=params['min_df'],
            max_df=params['max_df']
        )
        bow_vectorizer.fit_transform(vocab)
        
        bow = bow_vectorizer.transform([tweet['processed_tweet']])
        
        tweet_dict = OrderedDict(zip(bow_vectorizer.get_feature_names(), bow.toarray()[0]))

        return tweet_dict
