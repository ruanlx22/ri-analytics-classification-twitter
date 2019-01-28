import codecs
import json
import os
import pickle

DIR_ROOT = os.path.dirname(__file__)
DIR_MODELS = os.path.join(DIR_ROOT, 'models/')
DIR_MODELS_IT = os.path.join(DIR_MODELS, 'italian/')
CFG_ITALIAN = os.path.join(DIR_ROOT, 'configs/italian.json')

DIR_MODELS_EN = os.path.join(DIR_MODELS, 'english/')
CFG_ENGLISH = os.path.join(DIR_ROOT, 'configs/english.json')

CFG_LANG = 'lang'
CFG_FEATURES = 'ml_features'
CFG_FEATURE_SENTIMENT = 'sentiment'
CFG_FEATURE_PROBLEM_REPORT = 'problem_report'
CFG_FEATURE_INQUIRY = 'inquiry'
CFG_FEATURE_IRRELEVANT = 'irrelevant'

CFG_MODELS = 'models'
CFG_MODEL_VOCABULARY = 'vocabulary'
CFG_MODEL_CLASSIFIERS = 'classifiers'
CFG_MODEL_PROBLEM_REPORT = 'problem_report'
CFG_MODEL_INQUIRY = 'inquiry'
CFG_MODEL_IRRELEVANT = 'irrelevant'

CFG_SCALERS = 'scaler'
CFG_SCALER_LINGUISTIC = 'linguistic'
CFG_SCALER_KEYWORDS = 'n_keywords'
CFG_SCALER_SENTIMENT = 'sentiment'

CFG_SIMILARITY = 'similarity'
CFG_SIM_VECTORIZER = 'vectorizer'
CFG_SIM_CORPUS_PROBLEM = 'corpus_problem_report'
CFG_SIM_CORPUS_INQUIRY = 'corpus_inquiry'
CFG_SIM_CORPUS_IRRELEVANT = 'corpus_irrelevant'

CFG_PARAMS = 'params'

LANGUAGE_MODEL_IT = None
LANGUAGE_MODEL_EN = None


class ModelFactory:
    @staticmethod
    def create(lang=None):
        reader = codecs.getreader("utf-8")
        if lang is None:
            return None

        elif lang == 'en':
            if LANGUAGE_MODEL_EN is None:
                print(DIR_ROOT)
                config_it = json.load(reader(open(CFG_ENGLISH, 'rb')))
                return LanguageModel(DIR_MODELS_EN, config_it)
            else:
                return LANGUAGE_MODEL_EN

        elif lang == 'it':
            if LANGUAGE_MODEL_IT is None:
                print(DIR_ROOT)
                config_it = json.load(reader(open(CFG_ITALIAN, 'rb')))
                return LanguageModel(DIR_MODELS_IT, config_it)
            else:
                return LANGUAGE_MODEL_IT


class LanguageModel:
    def __init__(self, model_folder, config):
        if config is None:
            return None
        else:
            self.config = config
            self.features_sentiment = config[CFG_FEATURES][CFG_FEATURE_SENTIMENT]
            self.features_problem_report = config[CFG_FEATURES][CFG_FEATURE_PROBLEM_REPORT]
            self.features_inquiry = config[CFG_FEATURES][CFG_FEATURE_INQUIRY]
            self.features_irelevant = config[CFG_FEATURES][CFG_FEATURE_IRRELEVANT]
            self.clf_problem_report = pickle.load(open(os.path.join(model_folder, config[CFG_MODELS][CFG_MODEL_CLASSIFIERS][CFG_MODEL_PROBLEM_REPORT]), 'rb'))
            self.clf_inquiry = pickle.load(open(os.path.join(model_folder, config[CFG_MODELS][CFG_MODEL_CLASSIFIERS][CFG_MODEL_INQUIRY]), 'rb'))
            self.clf_irrelevant = pickle.load(open(os.path.join(model_folder, config[CFG_MODELS][CFG_MODEL_CLASSIFIERS][CFG_MODEL_IRRELEVANT]), 'rb'))
            self.vocab = pickle.load(open(os.path.join(model_folder, config[CFG_MODELS][CFG_MODEL_VOCABULARY]), 'rb'))
            self.scaler_linguistic = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SCALERS][CFG_SCALER_LINGUISTIC]), 'rb')
            ) if CFG_SCALER_LINGUISTIC in config[CFG_MODELS][CFG_SCALERS] else None
            self.scaler_keywords = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SCALERS][CFG_SCALER_KEYWORDS]), 'rb')
            ) if CFG_SCALER_KEYWORDS in config[CFG_MODELS][CFG_SCALERS] else None
            self.scaler_sentiment = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SCALERS][CFG_SCALER_SENTIMENT]), 'rb')
            ) if CFG_SCALER_SENTIMENT in config[CFG_MODELS][CFG_SCALERS] else None
            self.sim_vectorizer = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SIMILARITY][CFG_SIM_VECTORIZER]), 'rb')
            ) if CFG_SIM_VECTORIZER in config[CFG_MODELS][CFG_SIMILARITY] else None
            self.sim_corpus_problem_report = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SIMILARITY][CFG_SIM_CORPUS_PROBLEM]), 'rb')
            ) if CFG_SIM_CORPUS_PROBLEM in config[CFG_MODELS][CFG_SIMILARITY] else None
            self.sim_corpus_inquiry = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SIMILARITY][CFG_SIM_CORPUS_INQUIRY]), 'rb')
            ) if CFG_SIM_CORPUS_INQUIRY in config[CFG_MODELS][CFG_SIMILARITY] else None
            self.sim_corpus_irrelevant = pickle.load(
                open(os.path.join(model_folder, config[CFG_MODELS][CFG_SIMILARITY][CFG_SIM_CORPUS_IRRELEVANT]), 'rb')
            ) if CFG_SIM_CORPUS_IRRELEVANT in config[CFG_MODELS][CFG_SIMILARITY] else None


if __name__ == "__main__":
    model = ModelFactory.create('it')
    print(model.vocab)
