import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import numpy as np
from typing import List, Dict, Tuple


def process_tweet(tweet):
    """
    Preprocess a tweet into a list of stemmed tokens.

    Operations (in order):
      1) Remove stock tickers like "$GE".
      2) Remove leading "RT" retweet markers.
      3) Remove URLs (http/https).
      4) Strip the "#" character but keep hashtag text (e.g., "#topic" → "topic").
      5) Tokenize using NLTK's TweetTokenizer with:
         - preserve_case=False (lowercasing),
         - strip_handles=True (remove @handles),
         - reduce_len=True (normalize character repetitions, e.g., "soooo" → "soo").
      6) Remove English stopwords and punctuation.
      7) Apply Porter stemming to remaining tokens.

    Parameters
    ----------
    tweet : str
        Raw tweet text.

    Returns
    -------
    List[str]
        List of cleaned, lowercased, stemmed tokens.

    Notes
    -----
    - Requires NLTK resources: `stopwords`. Run once in your environment:
        >>> import nltk
        >>> nltk.download('stopwords')
    - This function is adapted from the DeepLearning.AI NLP specialization.
    """
        
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    #tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.

    From: DeepLearning.ai NLP Specialization course
    
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n


def _normalize_punctuation(s: str) -> str:
    """
    Normalize curly quotes, apostrophes, and dashes to ASCII equivalents.

    Replacements
    ------------
    ’, ‘  → '
    “, ”  → "
    —, –  → -

    Parameters
    ----------
    s : str
        Input text.

    Returns
    -------
    str
        Text with normalized punctuation.
    """
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return s


def clean_text(
    s: str,
    *,
    lower: bool = True,
    keep_hashtag_text: bool = True,
    keep_mention_text: bool = True,
) -> str:
    """
    Lightweight tweet cleaner for simple, reproducible preprocessing.

    Steps
    -----
    1) Normalize punctuation (curly quotes/dashes → ASCII).
    2) Remove URLs (http/https and www.*).
    3) Mentions:
       - If keep_mention_text=True: "@user" → " user " (drops '@', keeps text)
       - Else: remove entire mention.
    4) Hashtags:
       - If keep_hashtag_text=True: "#topic" → " topic " (drops '#', keeps text)
       - Else: remove the entire hashtag token.
    5) Replace any non [A-Za-z0-9' -] characters with spaces.
    6) Collapse repeated whitespace and strip.
    7) Lowercase if `lower=True`.

    Parameters
    ----------
    s : str
        Raw input text.
    lower : bool, optional
        If True, lowercase the output, by default True.
    keep_hashtag_text : bool, optional
        If True, keep the hashtag text while dropping '#', by default True.
    keep_mention_text : bool, optional
        If True, keep the mention text while dropping '@', by default True.

    Returns
    -------
    str
        Cleaned text.

    
    """
    
    # Apply normalize punctuation
    s = _normalize_punctuation(s)

    # Remove URLs
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)

    # Mentions: @user -> "user" (if keep_mention_text) else removed
    if keep_mention_text:
        s = re.sub(r"@([A-Za-z0-9_]+)", r" \1 ", s)
    else:
        s = re.sub(r"@[A-Za-z0-9_]+", " ", s)

    # Hashtags: #topic -> "topic" (if keep_hashtag_text) else removed
    if keep_hashtag_text:
        s = re.sub(r"#([A-Za-z0-9_]+)", r" \1 ", s)
    else:
        s = re.sub(r"#[A-Za-z0-9_]+", " ", s)

    # Keep letters, digits, apostrophes, hyphens. Turn other punctuation into spaces
    # (allow apostrophes/hyphens inside words like: it's, state-of-the-art)
    s = re.sub(r"[^A-Za-z0-9'\-]+", " ", s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    if lower:
        s = s.lower()

    return s


def tokenize(s: str) -> List[str]:
    """
    Minimal whitespace tokenizer.

    Parameters
    ----------
    s : str
        Pre-cleaned string (e.g., output of `clean_text`).

    Returns
    -------
    List[str]
        Tokens obtained by splitting on spaces.

    Notes
    -----
    - This function intentionally does not perform any normalization or filtering;
      it assumes cleaning decisions have already been applied upstream.
    """
    return s.split()


