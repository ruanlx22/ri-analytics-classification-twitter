import os
import re
import shlex
import subprocess
from collections import OrderedDict

import pandas as pd


class ExtractorSentiment:
    @staticmethod
    def extract(lang, tweet, params):
        text = tweet['text']
        DIR_ROOT = os.getcwd()
        senti_jar = os.path.join(DIR_ROOT, 'sentistrength/SentiStrength.jar')
        senti_folder = ''
        if lang == 'it':
            senti_folder = os.path.join(DIR_ROOT, 'sentistrength/SentStrength_Data_IT/')
        elif lang == 'en':
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

        tweet_dict = OrderedDict()
        for feature in params['forms']:
            if feature == 'sentiment_is_negative':
                tweet_dict['sentiment_is_negative'] = 1 if score_single < -1 else 0
            elif feature == 'sentiment_is_neutral':
                tweet_dict['sentiment_is_neutral'] = 1 if score_single >= -1 and score_single <= 1 else 0
            elif feature == 'sentiment_is_positive':
                tweet_dict['sentiment_is_positive'] = 1 if score_single > 1 else 0
            elif feature == 'sentiment_pos':
                tweet_dict['sentiment_pos'] = score_pos
            elif feature == 'sentiment_neg':
                tweet_dict['sentiment_neg'] = score_neg
            elif feature == 'sentiment_single':
                tweet_dict['sentiment_single'] = score_single

        tweet_update = dict()
        sentiment = ''
        if (score_single < -1):
            sentiment = 'NEGATIVE'
        elif (score_single >= -1 and score_single <= 1):
            sentiment = 'NEUTRAL'
        elif (score_single > 1):
            sentiment = 'POSITIVE'
        tweet_update['sentiment'] = sentiment
        tweet_update['sentiment_score'] = score_single

        return tweet_dict, tweet_update
