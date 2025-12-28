import emojis
import requests
import re
import string

from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import Counter
from hazm import POSTagger , stopwords_list , word_tokenize
from dataclasses import dataclass
from transformers import pipeline

class FeatureExtractor:
    def __init__(self,tokenizer,min_count):
        self.tokenizer = tokenizer
        self.min_count = min_count # for counting repeative words
        self.pipe = pipeline("token-classification", model="wietsedv/xlm-roberta-base-ft-udpos28-fa")


    def __call__(self,text):
        hashtags = self.extract_hashtags(text)
        emojis = self.extract_emojies(text)
        pos = self.extract_POS_tags(text)
        # repeated_words = self.extract_repeated_persian_words(text)
        # intense_puncs = self.extract_punctuation(text)

        features = ""
        features += hashtags
        features += emojis
        # features += pos
        # features += intense_puncs

        return features

    # extract hashtags
    def extract_hashtags(self,text):
        hashtag_list = []
        for word in text.split():
            if word[0] == '#':
                hashtag_list.append(word[1:])
        return f"</s>{','.join(hashtag_list)}</s>" if hashtag_list != [] else ''

    # get emojis from text
    def extract_emojies(self,text):
        emojies = list(emojis.get(text))
        if ":)" in text or "(:" in text:
            emojies.append(":)")

        if ":|" in text or "|:" in text:
            emojies.append(":|")

        if ":(" in text or "):" in text:
            emojies.append(":(")

        if "<3" in text:
            emojies.append("<3")
        
        if ":-)" in text or "(-:" in text:
            emojies.append(":-)")

        if ":-(" in text or ")-:" in text:
            emojies.append(":-(")

        return f"</s>{','.join(emojies)}</s>" if emojies != [] else ''

    # part of speech
    def extract_POS_tags(self,text):
        output = self.pipe(text)
        POS_tags = [entity['entity'] for entity in output]

        return f"</s>{','.join(POS_tags)}</s>"
    

    def extract_repeated_persian_words(self,text):
        tokens = word_tokenize(text)
        word_counts = Counter(tokens)
        stop_words = set(stopwords_list())
        punctuation_marks = set(string.punctuation) 
        repeated_words = [word for word, count in word_counts.items() if count > self.min_count and word.replace('▁','') not in stop_words and word not in punctuation_marks]
        
        return f"</s>{','.join(repeated_words)}</s>" if repeated_words else ''


    def extract_punctuation(self,text):
        punctuation_marks = re.findall(r'[?!-]', text)
        return f"</s>{','.join(punctuation_marks)}</s>" if punctuation_marks != [] else ''





