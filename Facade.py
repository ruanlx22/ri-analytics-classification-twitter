import uuid

import numpy as np

from FeatureExtractionFactory import FeatureExtractionFactory
from ModelFactory import ModelFactory

FEATURE_EXTRACTION_FACTORY = FeatureExtractionFactory()


def process_tweets(tweets, lang):
    model = ModelFactory.create(lang)
    classified_tweets = classify(model, tweets)
    return classified_tweets


def classify(model, tweets):
    unique_id = uuid.uuid4()
    print('{}, start classification of {} tweets'.format(unique_id, len(tweets)))

    ###################
    # prepare tweets
    ###################
    encoded_tweets = []
    for t in tweets:
        tweet = dict(t)
        tweet['text'] = t['text'].encode('ascii', errors='ignore').decode("utf-8")
        encoded_tweets.append(tweet)

    ###################
    # extract ml features and classify tweets
    ###################
    classified_tweets = []
    for tweet in encoded_tweets:
        extractor = FEATURE_EXTRACTION_FACTORY.create(model)
        extractor.extract_features(tweet)
        df_problem_report_model_data = extractor.get_features(model.features_problem_report)
        df_inquiry_model_data = extractor.get_features(model.features_inquiry)
        df_irrelevant_model_data = extractor.get_features(model.features_irelevant)

        # add sentiment
        tweet.update(extractor.extract_sentiment_onehot_encoding())
        extractor.reset()

        # add classified class
        tweet_classes = ['irrelevant', 'inquiry', 'problem_report']
        tweet_class = tweet_classes[0]  # default is irrelevant
        tweet_class_certainty = np.array([])
        if int(model.clf_irrelevant.predict(df_irrelevant_model_data)[0]) == 0:   # if the class != irrelevant, check for others
            if int(model.clf_inquiry.predict(df_inquiry_model_data)[0]) == 1:
                tweet_class = tweet_classes[1]  # class == inquiry
                try:
                    tweet_class_certainty = model.clf_inquiry.predict_proba(df_inquiry_model_data)[0]
                except:
                    pass
            elif int(model.clf_problem_report.predict(df_problem_report_model_data)[0]) == 1:
                tweet_class = tweet_classes[2]  # class == problem_report
                try:
                    tweet_class_certainty = model.clf_problem_report.predict_proba(df_problem_report_model_data)[0]
                except:
                    pass
        else:
            try:
                tweet_class_certainty = model.clf_irrelevant.predict_proba(df_irrelevant_model_data)[0]
            except:
                pass

        tweet['tweet_class'] = tweet_class
        if tweet_class_certainty.any():
            tweet['classifier_certainty'] = int(tweet_class_certainty[1]*100)
        else:
            tweet['classifier_certainty'] = -1

        classified_tweets.append(tweet)

    return classified_tweets
