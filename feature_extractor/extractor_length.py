import os
from collections import OrderedDict

class ExtractorLength():
    @staticmethod
    def extract(lang, tweet, params):
        text = tweet['processed_tweet']
        length = len(text.split())
        tweet_dict = OrderedDict()
        tweet_dict['processed_tweet_length'] = length
        return tweet_dict
        