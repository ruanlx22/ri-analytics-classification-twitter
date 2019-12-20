import os
import re
import shlex
import subprocess
from collections import OrderedDict

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from feature_extractor.extractor_bow import ExtractorBow
from feature_extractor.extractor_keywords import ExtractorKeywords
from feature_extractor.extractor_linguistic import ExtractorLinguistic
from feature_extractor.extractor_sentiment import ExtractorSentiment
from feature_extractor.extractor_tfidf import ExtractorTfidf
from feature_extractor.extractor_length import ExtractorLength
from model_factory import CFG_FEATURES, CFG_LANG

# Load models globally to avoid crashes when paralell call are coming
SPACY_MAX_LENGTH = 3000000
SPACY_NLP_IT = spacy.load('it')
SPACY_NLP_EN = spacy.load('en')
SPACY_NLP_IT.max_length = SPACY_MAX_LENGTH
SPACY_NLP_EN.max_length = SPACY_MAX_LENGTH

class FeatureExtractor:
    def __init__(self, cfg, lang, tweet):
        self.cfg = cfg
        self.lang =lang
        self.tweet = tweet
        self.data_vector = None

        self.run_pipeline()
    
        self.tweet.pop('processed_tweet', None)

    def run_pipeline(self):
        # extract features in the order given by the config
        df = pd.DataFrame()
        feature_vector = OrderedDict()

        self.preprocess()
        for feature, params in self.cfg[CFG_FEATURES].items():
            if feature == 'bow':
                tmp_feature_vector = ExtractorBow.extract(self.lang, self.tweet, params)
            elif feature == 'tfidf':
                tmp_feature_vector = ExtractorTfidf.extract(self.lang, self.tweet, params)
            elif feature == 'has_keyword':
                tmp_feature_vector = ExtractorKeywords.extract_has_keywords(self.lang, self.tweet, params)
            elif feature == 'n_keyword':
                tmp_feature_vector = ExtractorKeywords.extract_n_keywords(self.lang, self.tweet, params)
            elif feature == 'sentiment':
                tmp_feature_vector, tweet_update = ExtractorSentiment.extract(self.lang, self.tweet, params)
                self.tweet.update(tweet_update)
            elif feature == 'linguistic':
                tmp_feature_vector = ExtractorLinguistic.extract(self.lang, self.tweet, params)
            elif feature == 'length':
                tmp_feature_vector = ExtractorLength.extract(self.lang, self.tweet, params)
            feature_vector.update(tmp_feature_vector)

        self.data_vector = pd.DataFrame([feature_vector], columns=feature_vector.keys())

    def preprocess(self) :
        if self.lang == 'en':
            text = self.tweet['text'].strip().lower()
            text = ' '.join([w for w in text.split() if w not in SPACY_NLP_EN.Defaults.stop_words])
            text = ' '.join([w.lemma_ for w in SPACY_NLP_EN(text) if not w.is_punct])  # lemmatize
            text = ' '.join([w for w in text.split() if not '@' in w])  # remove @ mentions
            text = ' '.join([w for w in text.split() if not 'PRON' in w])  # remove pronouns

            self.tweet['processed_tweet'] = text

        elif self.lang == 'it':
            text = self.tweet['text'] if self.tweet['text'] and isinstance(self.tweet['text'], str) else ''
            if not text or text == '' or len(text) >= SPACY_MAX_LENGTH:
                return ''

            text = text.strip().lower()
            text = re.sub(r'(http|https):\s*/\s*/[\w\\-]+(\.[\w\\-]+)+\s*\\S*', '', text)
            text = re.sub(r'xx*', '', text)
            text = re.sub(r'[^\w\s\?\\@]', '', text)
            text = text.replace('\\n', '').replace('\\t', '')
            # since spacy splits important information such as 4g into to two tokens, we transform 
            # that kind of information by adding an underscore betweet the number 
            for token in re.findall(r'\\d\\w+', text):
                match = re.match(r'([0-9]+)([a-z]+)', token, re.I)
                if match:
                    items = match.groups()
                    correction = items[0]+'_'+items[1]
                    text = text.replace(token, correction)
            text = ' '.join([w for w in text.split() if '@' not in w])
            # text = ' '.join([w for w in text.split() if w not in SPACY_NLP_IT.Defaults.stop_words])
            text = ' '.join([w.lemma_ for w in SPACY_NLP_IT(text) if w.lemma_ != '-PRON-'])

            self.tweet['processed_tweet'] = text
