import os
import pickle
import re
import shlex
import subprocess

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from ModelFactory import CFG_FEATURES, CFG_LANG, CFG_MODELS, CFG_PARAMS

SPACY_MAX_LENGTH = 3000000

class FeatureExtractionFactory:
    def __init__(self):
        self.extractor_it = None
        self.extractor_en = None

    def create(self, model=None):
        if model.config[CFG_LANG] == 'it':
            if self.extractor_it is None:
                self.extractor_it = Extractor(model)
            return self.extractor_it
        elif model.config[CFG_LANG] == 'en':
            if self.extractor_en is None:
                self.extractor_en = Extractor(model)
            return self.extractor_en


class Extractor:
    def __init__(self, model):
        self.model = model
        self.conf = model.config
        self.tweet = None

        print('load vectorizers')
        self.bow_vectorizer = CountVectorizer(
            min_df=self.conf[CFG_PARAMS]['bow']['min_df'],
            max_df=self.conf[CFG_PARAMS]['bow']['max_df']
        )
        self.bow_vectorizer.fit_transform(model.vocab)

        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=self.conf[CFG_PARAMS]['tfidf']['min_df'],
            max_df=self.conf[CFG_PARAMS]['tfidf']['max_df']
        )
        self.tfidf_vectorizer.fit_transform(model.vocab)

        print('load spacy')
        self.nlp = spacy.load(self.conf[CFG_PARAMS]['spacy']['lang'])
        self.nlp.max_length = SPACY_MAX_LENGTH

        self.df_root = None
        self.df_linguistic = None
        self.df_sentiments = None
        self.df_bow = None
        self.df_tfidf = None
        self.df_n_keywords = None
        self.df_has_keyword = None
        self.df_similarity = None
        self.sentiment = dict()

    def reset(self):
        self.df_root = None
        self.df_linguistic = None
        self.df_sentiments = None
        self.df_bow = None
        self.df_tfidf = None
        self.df_n_keywords = None
        self.df_has_keyword = None
        self.df_similarity = None
        self.sentiment = dict()

    def extract_features(self, tweet):
        self.tweet = self.prepare_tweet(tweet)
        self.extract_linguistic_features(self.tweet)
        self.extract_sentiment(self.tweet)
        self.extract_bow(self.tweet)
        self.extract_tfidf(self.tweet)
        self.extract_keyword_counts(self.tweet)
        self.extract_keyword_onehot_encoding(self.tweet)
        self.extract_similarity(self.tweet)

    def prepare_tweet(self, tweet):
        text = tweet['text'] if tweet['text'] and isinstance(tweet['text'], str) else ''
    
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
        text = ' '.join([w for w in text.split() if w not in self.nlp.Defaults.stop_words])
        text = ' '.join([w.lemma_ for w in self.nlp(text) if w.lemma_ != '-PRON-'])

        tweet_dict = dict()
        tweet_dict['text'] = tweet['text']
        tweet_dict['processed_tweet'] = text

        self.df_root = pd.DataFrame([tweet_dict])

        return tweet_dict

    def extract_linguistic_features(self, tweet):
        if self.conf['lang'] == 'it':
            pos_dict = {
                'DET': 0,
                'SYM': 0,
                'VERB': 0,
                'NUM': 0,
                'NOUN': 0,
                'AUX': 0,
                'ADJ': 0,
                'PART': 0,
                'PUNCT': 0,
                'ADP': 0,
                'X': 0,
                'INTJ': 0,
                'SCONJ': 0,
                'PRON': 0,
                'ADV': 0,
                'SPACE': 0,
                'PROPN': 0,
                'CONJ': 0
            }
            tense_dict = {
                'Past': 0,
                'Pres': 0,
                'Fut': 0,
                'Imp': 0
            }
            n_stopwords = 0

            for word in self.nlp(tweet['text']):
                # n_pos features
                pos = word.pos_
                pos_dict[pos] = pos_dict[pos] + 1
                # n_stopwords
                if word.is_stop:
                    n_stopwords += 1
                # n_tense features
                if 'Tense=' in word.tag_:
                    tense = word.tag_.split('Tense=')[1].split('|')[0]
                    tense_dict[tense] = tense_dict[tense] + 1

            tweet_dict = dict()
            tweet_dict['n_tense_past'] = tense_dict['Past']
            tweet_dict['n_tense_pres'] = tense_dict['Pres']
            tweet_dict['n_tense_fut'] = tense_dict['Fut']
            tweet_dict['n_tense_imp'] = tense_dict['Imp']
            tweet_dict['n_words'] = len(tweet['text'].split())
            tweet_dict['n_stopwords'] = n_stopwords
            tweet_dict['n_pos_det'] = pos_dict['DET']
            tweet_dict['n_pos_sym'] = pos_dict['SYM']
            tweet_dict['n_pos_verb'] = pos_dict['VERB']
            tweet_dict['n_pos_num'] = pos_dict['NUM']
            tweet_dict['n_pos_noun'] = pos_dict['NOUN']
            tweet_dict['n_pos_aux'] = pos_dict['AUX']
            tweet_dict['n_pos_adj'] = pos_dict['ADJ']
            tweet_dict['n_pos_par'] = pos_dict['PART']
            tweet_dict['n_pos_punct'] = pos_dict['PUNCT']
            tweet_dict['n_pos_adp'] = pos_dict['ADP']
            tweet_dict['n_pos_x'] = pos_dict['X']
            tweet_dict['n_pos_intj'] = pos_dict['INTJ']
            tweet_dict['n_pos_sconj'] = pos_dict['SCONJ']
            tweet_dict['n_pos_pron'] = pos_dict['PRON']
            tweet_dict['n_pos_adv'] = pos_dict['ADV']
            tweet_dict['n_pos_space'] = pos_dict['SPACE']
            tweet_dict['n_pos_propn'] = pos_dict['PROPN']
            tweet_dict['n_pos_conj'] = pos_dict['CONJ']

        elif self.conf['lang'] == 'en':
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

            for word in self.nlp(tweet['text']):
                # n_pos features
                pos = word.pos_
                pos_dict[pos] = pos_dict[pos] + 1
                # n_stopwords
                if word.is_stop:
                    n_stopwords += 1
                # n_tense features
                tag = word.tag_
                if tag in self.conf['params']['tense']['present']:
                    tense_dict['Pres'] = tense_dict['Pres'] + 1
                elif tag in self.conf['params']['tense']['past']:
                    tense_dict['Past'] = tense_dict['Past'] + 1

            tweet_dict = dict()
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
            tweet_dict['n_tense_past'] = tense_dict['Past']
            tweet_dict['n_tense_pres'] = tense_dict['Pres']
            tweet_dict['n_words'] = len(tweet['text'].split())
            tweet_dict['n_stopwords'] = n_stopwords

        self.df_linguistic = pd.DataFrame([tweet_dict])
        if 'linguistic' in self.conf[CFG_MODELS]['scaler']:
            self.df_linguistic[self.df_linguistic.columns] = self.model.scaler_linguistic.transform(self.df_linguistic[self.df_linguistic.columns])

    def extract_sentiment(self, tweet):
        text = tweet['text']
        # open a subprocess using shlex to get the command line string into the correct args list format
        DIR_ROOT = os.path.dirname(__file__)
        senti_jar = os.path.join(DIR_ROOT, 'sentistrength/SentiStrength.jar')
        senti_folder = ''
        if self.conf['lang'] == 'it':
            senti_folder = os.path.join(DIR_ROOT, 'sentistrength/SentStrength_Data_IT/')
        elif self.conf['lang'] == 'en':
            senti_folder = os.path.join(DIR_ROOT, 'sentistrength/SentStrength_Data/')
        p = subprocess.Popen(
            shlex.split('java -jar ' + senti_jar + ' stdin sentidata ' + senti_folder),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # communicate via stdin the string to be rated. Note that all spaces are replaced with +
        stdout_text, stderr_text = p.communicate(bytearray(text.replace(' ', '+'), 'utf8'))
        p.kill()

        raw_sentiment = str(stdout_text, 'utf-8').split()
        score_pos = int(raw_sentiment[0])
        score_neg = int(raw_sentiment[1])
        score_single = score_pos - -score_neg

        tweet_dict = dict()
        if 'sentiment_simplified' in self.conf[CFG_FEATURES]['sentiment']:
            tweet_dict['sentiment_is_negative'] = 1 if score_single < -1 else 0
            tweet_dict['sentiment_is_neutral'] = 1 if score_single >= -1 and score_single <= 1 else 0
            tweet_dict['sentiment_is_positive'] = 1 if score_single > 1 else 0
        tweet_dict['sentiment_pos'] = score_pos
        tweet_dict['sentiment_neg'] = score_neg
        tweet_dict['sentiment_single'] = score_single


        self.sentiment['sentiment_pos'] = score_pos
        self.sentiment['sentiment_neg'] = score_neg
        self.sentiment['sentiment_single'] = score_single

        self.df_sentiments = pd.DataFrame([tweet_dict])
        if 'sentiment' in self.conf[CFG_MODELS]['scaler']:
            self.df_sentiments[self.df_sentiments.columns] = self.model.scaler_sentiment.transform(self.df_sentiments[self.df_sentiments.columns])

    def extract_sentiment_onehot_encoding(self):
        tweet = dict()
        tweet['sentiment_score'] = int(self.sentiment['sentiment_single'])
        sentiment = ''
        if (self.sentiment['sentiment_single'] < -1):
            sentiment = 'NEGATIVE'
        elif (self.sentiment['sentiment_single'] >= -1 and self.sentiment['sentiment_single'] <= 1):
            sentiment = 'NEUTRAL'
        elif (self.sentiment['sentiment_single'] > 1):
            sentiment = 'POSTIVE'
        tweet['sentiment'] = sentiment

        return tweet

    def extract_bow(self, tweet):
        text = tweet['processed_tweet']

        # bow extraction
        bow_arr = self.bow_vectorizer.transform([text])

        tweet_dict = dict()
        for i in range(len(self.bow_vectorizer.get_feature_names())):
            key = str(self.bow_vectorizer.get_feature_names()[i])
            val = bow_arr.toarray()[0][i]
            tweet_dict[key] = val

        self.df_bow = pd.DataFrame([tweet_dict])
        self.df_bow = self.df_bow.add_prefix('bow_')

    def extract_tfidf(self, tweet):
        text = tweet['processed_tweet']

        # bow extraction
        tfidf_arr = self.tfidf_vectorizer.transform([text])

        tweet_dict = dict()
        for i in range(len(self.tfidf_vectorizer.get_feature_names())):
            key = str(self.tfidf_vectorizer.get_feature_names()[i])
            val = tfidf_arr.toarray()[0][i]
            tweet_dict[key] = val

        self.df_tfidf = pd.DataFrame([tweet_dict])
        self.df_tfidf = self.df_tfidf.add_prefix('tfidf_')

    def extract_keyword_counts(self, tweet):
        text = tweet['text']

        tweet_dict = dict()
        tweet_dict['n_keywords_problem'] = len(re.findall(self.conf[CFG_PARAMS]['keywords']['problem'], text, re.IGNORECASE))
        tweet_dict['n_keywords_support'] = len(re.findall(self.conf[CFG_PARAMS]['keywords']['support'], text, re.IGNORECASE))

        self.df_n_keywords = pd.DataFrame([tweet_dict])
        if 'n_keywords' in self.conf[CFG_MODELS]['scaler']:
            self.df_n_keywords[self.df_n_keywords.columns] = self.model.scaler_keywords.transform(self.df_n_keywords[self.df_n_keywords.columns])

    def extract_keyword_onehot_encoding(self, tweet):
        text = tweet['text']
        
        tweet_dict = dict()
        tweet_dict['has_keyword_bug'] = 1 if len(re.findall('bug', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_crash'] = 1 if len(re.findall('crash', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_malfunzionamento'] = 1 if len(re.findall('malfunzionamento', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_blocca'] = 1 if len(re.findall('blocca', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_non_funziona'] = 1 if len(re.findall('non funziona', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_morto'] = 1 if len(re.findall('morto', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_chiuso'] = 1 if len(re.findall('chiuso', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_err'] = 1 if len(re.findall('err', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_andato'] = 1 if len(re.findall('andato', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_problem'] = 1 if len(re.findall('problem', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_dovrebbe'] = 1 if len(re.findall('dovrebbe', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_vorrei'] = 1 if len(re.findall('vorrei', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_aggiungere'] = 1 if len(re.findall('aggiungere', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_manca'] = 1 if len(re.findall('manca', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_bisogno'] = 1 if len(re.findall('bisogno', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_aiuto'] = 1 if len(re.findall('aiuto', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_supporto'] = 1 if len(re.findall('supporto', text, re.IGNORECASE)) > 0 else 0,
        tweet_dict['has_keyword_help'] = 1 if len(re.findall('help', text, re.IGNORECASE)) > 0 else 0

        self.df_has_keyword = pd.DataFrame([tweet_dict])

    def extract_similarity(self, tweet):
        if self.model.sim_vectorizer:
            doc_matrix = self.model.sim_vectorizer.transform([tweet['processed_tweet']])
            tweet_dict = dict()
            tweet_dict['mean_similarity_to_problems'] = cosine_similarity(doc_matrix, self.model.sim_corpus_problem_report)[0].mean()
            tweet_dict['mean_similarity_to_inquiries'] = cosine_similarity(doc_matrix, self.model.sim_corpus_inquiry)[0].mean()
            tweet_dict['mean_similarity_to_irrelevants'] = cosine_similarity(doc_matrix, self.model.sim_corpus_irrelevant)[0].mean()

            self.df_similarity = pd.DataFrame([tweet_dict])

    def get_features(self, features):
        df_features = pd.DataFrame()
        if 'linguistic' in features:
            df_linguistic = self.df_linguistic
            df_features = pd.concat([df_features, df_linguistic], axis=1)

        if 'linguistic_tense' in features:
            filtered_cols = [col for col in self.df_linguistic if col.startswith('n_tense_')]
            df_linguistic_tense = self.df_linguistic[filtered_cols]
            df_features = pd.concat([df_features, df_linguistic_tense], axis=1)

        if 'linguistic_pos_counts' in features:
            filtered_cols = [col for col in self.df_linguistic if col.startswith('n_pos_')]
            df_linguistic_pos = self.df_linguistic[filtered_cols]
            df_features = pd.concat([df_features, df_linguistic_pos], axis=1)

        if 'linguistic_word_counts' in features:
            df_linguistic_n_words = self.df_linguistic['n_words']
            df_features = pd.concat([df_features, df_linguistic_n_words], axis=1)

        if 'linguistic_stopword_counts' in features:
            df_linguistic_n_stopwords = self.df_linguistic['n_stopwords']
            df_features = pd.concat([df_features, df_linguistic_n_stopwords], axis=1)

        if 'sentistrength_plus' in features:
            df_sentiment = self.df_sentiments[['sentiment_pos', 'sentiment_neg', 'sentiment_single']]
            df_features = pd.concat([df_features, df_sentiment], axis=1)

        if 'sentiment_single_val' in features:
            df_sentiment_single = self.df_sentiments['sentiment_single']
            df_features = pd.concat([df_features, df_sentiment_single], axis=1)

        if 'sentiment_pos_neg' in features:
            df_sentiment_pos_neg = self.df_sentiments[['sentiment_pos', 'sentiment_neg']]
            df_features = pd.concat([df_features, df_sentiment_pos_neg], axis=1)

        if 'sentiment_simplified' in features:
            df_sentiment_simple = self.df_sentiments[['sentiment_is_negative', 'sentiment_is_neutral', 'sentiment_is_positive']]
            df_features = pd.concat([df_features, df_sentiment_simple], axis=1)

        if 'n_keywords' in features:
            df_features = pd.concat([df_features, self.df_n_keywords], axis=1)

        if 'has_keyword' in features:
            df_features = pd.concat([df_features, self.df_has_keyword], axis=1)

        if 'bow' in features:
            df_features = pd.concat([df_features, self.df_bow], axis=1)

        if 'tfidf' in features:
            df_features = pd.concat([df_features, self.df_tfidf], axis=1)

        if 'similarity' in features:
            df_features = pd.concat([df_features, self.df_similarity], axis=1)

        return df_features
