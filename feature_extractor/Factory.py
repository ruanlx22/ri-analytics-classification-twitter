import os
import re
import shlex
import subprocess

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
from ModelFactory import CFG_FEATURES, CFG_LANG

SPACY_MAX_LENGTH = 3000000

class FeatureExtractor:
    def __init__(self, cfg, lang, tweet):
        self.cfg = cfg
        self.lang =lang
        self.tweet = tweet
        self.data_vector = None

        self.run_pipeline()

    def run_pipeline(self):
        # extract features in the order given by the config
        df = pd.DataFrame()
        self.preprocess()
        for feature, params in self.cfg[CFG_FEATURES].items():
            if feature == 'bow':
                df_tmp = ExtractorBow.extract(self.lang, self.tweet, params)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)
            elif feature == 'tfidf':
                df_tmp = ExtractorTfidf.extract(self.lang, self.tweet, params)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)
            elif feature == 'has_keyword':
                df_tmp = ExtractorKeywords.extract_has_keywords(self.lang, self.tweet, params)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)
            elif feature == 'n_keyword':
                df_tmp = ExtractorKeywords.extract_n_keywords(self.lang, self.tweet, params)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)
            elif feature == 'sentiment':
                df_tmp, tweet_update = ExtractorSentiment.extract(self.lang, self.tweet, params)
                self.tweet.update(tweet_update)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)
            elif feature == 'linguistic':
                df_tmp = ExtractorLinguistic.extract(self.lang, self.tweet, params)
                df_tmp.reset_index(drop=True, inplace=True)
                df = pd.concat([df, df_tmp], axis=1)

        self.data_vector = df

    def preprocess(self) :
        if self.lang == 'en':
            nlp = spacy.load(self.lang)
            nlp.max_length = SPACY_MAX_LENGTH

            text = self.tweet['text'].strip().lower()
            text = ' '.join([w for w in text.split() if w not in nlp.Defaults.stop_words])
            text = ' '.join([w.lemma_ for w in nlp(text) if not w.is_punct])  # lemmatize
            text = ' '.join([w for w in text.split() if not '@' in w])  # remove @ mentions
            text = ' '.join([w for w in text.split() if not 'PRON' in w])  # remove pronouns

            self.tweet['processed_tweet'] = text

        elif self.lang == 'it':
            nlp = spacy.load(self.lang)
            nlp.max_length = SPACY_MAX_LENGTH
            
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
            # text = ' '.join([w for w in text.split() if w not in self.nlp.Defaults.stop_words])
            text = ' '.join([w.lemma_ for w in nlp(text) if w.lemma_ != '-PRON-'])

            self.tweet['processed_tweet'] = text
