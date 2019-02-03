import re
from collections import OrderedDict

import pandas as pd


class ExtractorKeywords:
    @staticmethod
    def extract_has_keywords(lang, tweet, params):
        tweet_dict = OrderedDict()
        for keyword in params['keywords']:
            tweet_dict['has_keyword_'+keyword] = 1 if len(re.findall(keyword, tweet['text'], re.IGNORECASE)) > 0 else 0
        
        return pd.DataFrame([tweet_dict], columns=tweet_dict.keys())
    
    @staticmethod
    def extract_n_keywords(lang, tweet, params):
        tweet_dict = OrderedDict()
        for feature, values in params.items():
            tweet_dict['n_keywords_'+feature] = len(re.findall(values, tweet['text'], re.IGNORECASE))
        
        return pd.DataFrame([tweet_dict], columns=tweet_dict.keys())
