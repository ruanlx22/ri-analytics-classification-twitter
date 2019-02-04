import codecs
import json
import os
import pickle
from collections import OrderedDict

# folders
DIR_ROOT = os.path.dirname(__file__)
DIR_MODELS = os.path.join(DIR_ROOT, 'models/')

# configs per language
DIR_MODELS_IT = os.path.join(DIR_MODELS, 'italian/')
CFG_ITALIAN = os.path.join(DIR_ROOT, 'configs/italian.json')

DIR_MODELS_EN = os.path.join(DIR_MODELS, 'english/')
CFG_ENGLISH = os.path.join(DIR_ROOT, 'configs/english.json')

# config keys
CFG_LANG = 'lang'
CFG_PBR = 'problem_report'
CFG_INQ = 'inquiry'
CFG_IRR = 'irrelevant'
CFG_FEATURES = 'features'
CFG_MODEL = 'model'

# language models
LANGUAGE_MODEL_IT = None
LANGUAGE_MODEL_EN = None

def get_models_dir(lang):
    if lang == 'en':
        return DIR_MODELS_EN
    if lang == 'it':
        return DIR_MODELS_IT
    return None

class ModelFactory:
    @staticmethod
    def create(lang=None):
        reader = codecs.getreader("utf-8")
        if lang is None:
            return None

        elif lang == 'en':
            if LANGUAGE_MODEL_EN is None:
                cfg = json.load(open(CFG_ENGLISH), object_pairs_hook=OrderedDict)
                return LanguageModel(DIR_MODELS_EN, cfg)
            else:
                return LANGUAGE_MODEL_EN

        elif lang == 'it':
            if LANGUAGE_MODEL_IT is None:
                cfg = json.load(open(CFG_ITALIAN), object_pairs_hook=OrderedDict)
                return LanguageModel(DIR_MODELS_IT, cfg)
            else:
                return LANGUAGE_MODEL_IT


class LanguageModel:
    def __init__(self, model_folder, cfg):
        if cfg is None:
            return None
        else:
            self.lang = cfg[CFG_LANG]
            
            self.cfg_pbr = OrderedDict(cfg[CFG_PBR])
            self.cfg_inq = OrderedDict(cfg[CFG_INQ])
            self.cfg_irr = OrderedDict(cfg[CFG_IRR])

            self.clf_pbr = pickle.load(open(os.path.join(model_folder, cfg[CFG_PBR][CFG_MODEL]), 'rb'))
            self.clf_inq = pickle.load(open(os.path.join(model_folder, cfg[CFG_INQ][CFG_MODEL]), 'rb'))
            self.clf_irr = pickle.load(open(os.path.join(model_folder, cfg[CFG_IRR][CFG_MODEL]), 'rb'))
