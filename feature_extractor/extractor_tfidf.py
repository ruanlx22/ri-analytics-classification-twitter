import os
import pickle
import re

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from ModelFactory import get_models_dir


class ExtractorTfidf:
    @staticmethod
    def extract(lang, tweet, params) :
        vocab_path = os.path.join(get_models_dir(lang), params['vocabulary'])
        vocab = pickle.load(open(vocab_path, 'rb'))
        tfidf_vectorizer = TfidfVectorizer(
            min_df=params['min_df'],
            max_df=params['max_df']
        )
        tfidf_vectorizer.fit_transform(vocab)
        
        tfidf = tfidf_vectorizer.transform([tweet['processed_tweet']])
        df = pd.DataFrame(data=tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names())
        df = df.add_prefix('tfidf_')

        return df
