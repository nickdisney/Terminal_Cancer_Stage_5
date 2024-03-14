# utils/tokenizer.py

from transformers import BertTokenizer

def get_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer