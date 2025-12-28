import re
import hazm
import config
from transformers import AutoTokenizer
from cleantext import clean
from dadmatools.models.normalizer import Normalizer
from feature_extraction import FeatureExtractor


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
min_count = 3
feature_extractor = FeatureExtractor(tokenizer=tokenizer,min_count=min_count)
hanorm = hazm.Normalizer(
    seperate_mi=False,decrease_repeated_chars=True,persian_numbers=False
)
normalizer = Normalizer(
    full_cleaning=False,
    unify_chars=True,
    refine_punc_spacing=True,
    remove_extra_space=True,
    remove_puncs=True,
    remove_html=True,
    remove_stop_word=False,
    replace_email_with=None,
    replace_number_with=None,
    replace_url_with=None,
    replace_mobile_number_with=None,
    replace_emoji_with=None,
    replace_home_number_with=None
)

def preprocess(df):
    # Normalize emotion labels
    emotion_columns = ['Anger', 'Fear', 'Happiness', 'Hatred', 'Sadness', 'Wonder']

    for col in emotion_columns:
        df[col] = df[col] / df[col].max()  # Normalize to the range [0, 1]

    # Apply threshold for binary labels
    threshold = config.EMOPARS_THRESHOLD
    for col in emotion_columns:
        df[col] = df[col].apply(lambda x: 1 if x >= threshold else 0)

    return df


def preprocess_pipeline(text):
    features = feature_extractor(text)
    text = cleaning(text)
    text = correct_repeating_characters(text)
    text = remove_arabic_diacritics(text)
    text = remove_english_characters(text)
    # text = remove_non_persian_chars(text)
    final_data = text + features
    return final_data

def remove_arabic_diacritics(text):
    """
        Some common Arabic diacritical marks include:
            Fatha (ً): Represents the short vowel "a" or "u" when placed above a letter.
            Kasra (ٍ): Represents the short vowel "i" when placed below a letter.
            Damma (ٌ): Represents the short vowel "u" when placed above a letter.
            Sukun (ـْ): Indicates the absence of any vowel sound.
            Shadda (ّ): Represents consonant doubling or gemination.
            Tanween (ًٌٍ): Represents the nunation or the "n" sound at the end of a word.
    """
    """
        The regular expression [\u064B-\u065F] represents a character range that covers the Unicode code points for Arabic diacritics.
    """
    # مرحبا بكم <== "مَرْحَبًا بِكُمْ"
    arabic_diacritics_pattern = re.compile(r'[\u064B-\u065F]')
    cleaned_text = re.sub(arabic_diacritics_pattern, '', text)
    return cleaned_text


def remove_non_persian_chars(text):
    persian_chars_pattern = re.compile(r'[^\u0600-\u06FF\uFB8A\u067E\u0686\u06AF\u200C\u200F\U0001F000-\U0001F9FF]+')
    cleaned_text = re.sub(persian_chars_pattern, ' ', text)
    return cleaned_text


def normalize(text):
    normalized_text = normalizer.normalize(text)
    normalized_text = hanorm.normalize(normalized_text)
    return normalized_text

def correct_repeating_characters(text):
    corrected_text = re.sub(r'(.)\1+', r'\1', text)
    return corrected_text



def remove_english_characters(text):
    # Define a regular expression pattern to match English characters
    pattern = re.compile("[a-zA-Z]")
    
    # Replace English characters with an empty string
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleaning(text):
    text = text.strip()   
    text = clean(text)
    text = cleanhtml(text)
    text = normalize(text)  
    text = re.sub("\s+", " ", text)
    return text



def data_balancer(df,num_samples):
    import pandas as pd
    import random

    # Load your dataset

    # Calculate the number of samples needed for each class to reach approximately 10,000 samples
    desired_samples_per_class = num_samples // len(df['emotion'].unique())

    # Create a list to store oversampled data for each class
    oversampled_data = []

    # Iterate over each class
    for emotion in df['emotion'].unique():
        # Calculate the number of samples needed for the current class
        samples_needed = desired_samples_per_class - len(df[df['emotion'] == emotion])
        
        # If there are more samples needed for the current class
        if samples_needed > 0:
            # Sample with replacement to oversample from the current class
            oversampled_samples = df[df['emotion'] == emotion].sample(n=samples_needed, replace=True, random_state=42)
            oversampled_data.append(oversampled_samples)
        # If there are more samples than needed for the current class
        elif samples_needed < 0:
            # Drop random samples from the current class
            random_drop_indices = df[df['emotion'] == emotion].sample(n=-samples_needed, random_state=42).index
            df = df.drop(random_drop_indices)

    # Concatenate the original dataset with the oversampled data for each class
    balanced_df = pd.concat([df] + oversampled_data)

    # Shuffle the dataframe to mix the samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure that the final dataset contains exactly 10,000 samples
    if len(balanced_df) < num_samples:
        # If there are less than 10,000 samples, add random samples from the original dataset
        remaining_samples_needed = num_samples - len(balanced_df)
        random_samples = df.sample(n=remaining_samples_needed, random_state=42)
        balanced_df = pd.concat([balanced_df, random_samples])
    elif len(balanced_df) > num_samples:
        # If there are more than 10,000 samples, randomly drop excess samples
        balanced_df = balanced_df.sample(n=num_samples, random_state=42)

    # Shuffle the dataframe again after adding or removing samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df
