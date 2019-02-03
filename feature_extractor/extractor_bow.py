import os
import pickle
import re

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

from ModelFactory import get_models_dir


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
        df = pd.DataFrame(data=bow.toarray(), columns=bow_vectorizer.get_feature_names())
        df = df.add_prefix('bow_')

        return df
