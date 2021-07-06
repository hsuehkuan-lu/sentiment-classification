import re
import string
import yaml
from collections import Counter, OrderedDict
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer

# Removing all punctuations from Text
mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def word_replace(text):
    return text.replace('<br />', '')


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def preprocess_text(text, remove_punc=True):
    text = clean_contractions(text, mapping)
    text = text.lower()
    text = word_replace(text)
    text = remove_urls(text)
    text = remove_html(text)
    if remove_punc:
        text = remove_punctuation(text)
    return text


with open('params.yaml', 'r') as f:
    PARAMS = yaml.safe_load(f)


tokenizer = get_tokenizer('basic_english')


def generate_vocabulary():
    counter = Counter()
    df = pd.read_csv('data/all.csv')
    for line in df[PARAMS['feature']]:
        counter.update([''.join(list(filter(lambda x: x.isalpha(), [ch for ch in word])))
                        for word in tokenizer(preprocess_text(line))])
    del counter['']
    num_classes = len(set([label for label in df[PARAMS['label']]]))
    sorted_by_freq_tuples = counter.most_common(PARAMS['basic']['vocab_size'])
    specials = (PARAMS['unk_token'], PARAMS['pad_token'], PARAMS['sos_token'], PARAMS['eos_token'])
    vocab = torchtext.vocab.vocab(OrderedDict(
        [(tok, 1) for tok in specials] + sorted_by_freq_tuples
    ))
    vocab.set_default_index(0)
    config = {
        'vocab_size': len(vocab),
        'num_classes': num_classes,
        'padding_idx': vocab[PARAMS['pad_token']],
        'sos_idx': vocab[PARAMS['sos_token']],
        'eos_idx': vocab[PARAMS['eos_token']]
    }
    return vocab, config


class Preprocessor(object):
    def __init__(self, vocab):
        super(Preprocessor, self).__init__()
        self._vocab = vocab
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._vocab)

    def text_pipeline(self, text):
        if isinstance(text, list):
            return [[self._vocab[''.join(list(filter(lambda x: x.isalpha(), i)))] for i in tokenizer(preprocess_text(t))] for t in text]
        return [self._vocab[''.join(list(filter(lambda x: x.isalpha(), i)))] for i in tokenizer(preprocess_text(text))]

    def label_pipeline(self, label):
        return label

    @property
    def vocab(self):
        return self._vocab
