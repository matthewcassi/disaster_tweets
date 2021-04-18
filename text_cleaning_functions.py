import pandas as pd
import numpy as np
import re
import spacy
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def count_words(text):
    """ Counts the number of words using Spacy"""
    doc = nlp(text)
    count_words = 0
    # Iterate over the tokens in the doc
    for token in doc:
        # Check if the token resembles a word
        if token.is_alpha:
            count_words += 1
    return count_words

def count_numbers(text):
    """ Counts the number of numbers using Spacy"""
    doc = nlp(text)
    count_nums = 0
    # Iterate over the tokens in the doc
    for token in doc:
        # Check if the token resembles a word
        if token.like_num:
            count_nums += 1
    return count_nums

def count_punc(text):
    """ Counts the number of punctuation symbols using Spacy"""
    doc = nlp(text)
    count_punct = 0
    # Iterate over the tokens in the doc
    for token in doc:
        # Check if the token resembles a word
        if token.is_punct:
            count_punct += 1
    return count_punct

def get_ner(text):
    """ Get the named entities for each tweet"""
    doc = nlp(text)
    return [X.label_ for X in doc.ents]

def remove_URL(text):
    """Remove URLs from text"""
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    """Remove emojis from text"""
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    """Remove HTML from text"""
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def count_vectors(df, col, ngrams):
    """Counts number of words or ngrams"""
    vec = CountVectorizer(stop_words = 'english', max_df = 0.9, ngram_range = (ngrams, ngrams))
    bow = vec.fit_transform(df[col])
    sum_bow = bow.sum(axis=0)
    # Count ngrams
    words_freq = [(word, sum_bow[0, idx])
                       for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq
