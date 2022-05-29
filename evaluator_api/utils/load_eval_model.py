from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


IMG_EVAL_MODEL = None
IMG_TOKENIZER = None
TEXT_EVAL_MODEL = None
TEXT_TOKENIZER = None

MODEL_NAME = f"cardiffnlp/twitter-roberta-base-offensive"
LABEL_MAPPING = {
    0: 'Not Offensive',
    1: 'Offensive'
}


def load_text_eval_model():
    global TEXT_EVAL_MODEL
    global TEXT_TOKENIZER

    TEXT_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    TEXT_EVAL_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def load_img_eval_model():
    global IMG_EVAL_MODEL
    global IMG_TOKENIZER

    IMG_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

    IMG_EVAL_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def get_text_eval_model():
    if TEXT_EVAL_MODEL is None and TEXT_TOKENIZER is None:
        raise Exception("Model or Tokenizer - Not loaded")
    return {
        "tokenizer": TEXT_TOKENIZER,
        "model": TEXT_EVAL_MODEL
    }


def get_img_eval_model():
    if IMG_EVAL_MODEL is None and IMG_TOKENIZER is None:
        raise Exception("Model or Tokenizer - Not loaded")
    return {
        "tokenizer": IMG_TOKENIZER,
        "model": IMG_EVAL_MODEL
    }

