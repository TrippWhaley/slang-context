import string

from gensim.models import Word2Vec
from data.load_datasets import load_reddit_teenagers_dataset, load_slang_dataset
from nltk import word_tokenize
import re
import os
import itertools


MODEL_NAME = 'test_word2vec_model'
MODEL_FILE = os.path.join(os.path.dirname(__file__), MODEL_NAME)

SLANG = load_slang_dataset()
REDDIT = load_reddit_teenagers_dataset()

def remove_non_utf8(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def train():
    # list of sentences
    reddit_dataset = [sent for sent in itertools.chain.from_iterable(REDDIT.values) if sent]
    slang_examples = [sent for sent in SLANG['Example'].values if sent]
    combined = reddit_dataset + slang_examples
    tokenized_sentences = [word_tokenize(remove_non_utf8(sent.lower())) for sent in combined]

    model = Word2Vec(sentences=tokenized_sentences)
    model.save(MODEL_NAME)


def evaluate():
    slang_words = SLANG['Slang'].to_list()
    # tokenize
    preprocessed = ["".join([c for c in remove_non_utf8(word.lower()) if c not in string.punctuation]) for word in slang_words]

    model = Word2Vec.load(MODEL_FILE)

    for word in preprocessed:
        try:
            print(word, ": ", model.wv.most_similar(word, topn=3))
        except Exception as e:
            print(e)

def main():
    if not os.path.exists(MODEL_FILE):
        train()
    evaluate()

if __name__ == "__main__":
    main()