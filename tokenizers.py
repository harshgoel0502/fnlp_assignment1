from collections import defaultdict
import re
import jsonlines
from typing import List
from tqdm import tqdm
import pickle
import os
import random

# compiled regex for splitting text to words and punctuation
identify_words_regex = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def convert_text_to_words(text: str) -> List[str]:
    """
    Utility function to split text to words and punctuation.
    """
    return identify_words_regex.findall(text)


class Tokenizer(object):
    """
    A generic class that tokenizes text using a given vocabulary.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size # the maximum number of tokens in the vocabulary
        self.token_to_id = {} # map from a token (e.g. 'the') to a unique integer id (e.g. 1)
        self.id_to_token = {} # map from a unique integer id to a token (e.g. 1 -> 'the')

    def tokenize(self, text: str, return_token_ids: bool = False):
        raise Exception("tokenize not implemented")

    def train(self, corpus: List[str]):
        raise Exception("train not implemented")

    def __len__(self):
        raise Exception("__len__ not implemented")


class ReturnWordsTokenizer(Tokenizer):
    def tokenize(self, text: str, return_token_ids: bool = False) -> List[str]:
        return convert_text_to_words(text)
    def train(self, corpus: List[str]):
        pass

class NgramTokenizer(Tokenizer):
    def __init__(self, n: int = 2, vocab_size: int = -1, *args, **kwargs):
        super().__init__(vocab_size, *args, **kwargs)
        self.n = n # n-gram size

    def tokenize(self, text: str, return_token_ids: bool = False) -> List[List[str]] | List[int]:
        """
        TODO: Tokenize a text using the NgramTokenizer.
        If return_token_ids is True, return a list of token ids (from 0 to len(vocab)).
        Otherwise, return a list of list of token strings (remember each 'token' is a tuple, e.g. ("hi",) or ("movie", "was")).
        Detailed instructions:
        1. Split the text into words and punctuation using convert_text_to_words utility function.
        2. Iterate over the words, get ngrams of size self.n, and convert them into token ids if return_token_ids is True.
        
        Remember, we ignore ngrams that are not in the vocabulary, and are only looking at ngrams of size self.n.
        
        Example input/outputs:
        self.n = 1
        Input `return_token_ids` = False
        Input `text`: "This movie was really bad, but bad in a fun way, so I loved it."
        Output: [
                ("This",),
                ("movie",),
                ("was",),
                ("really",),
                ("bad",),
                (",",),
                ("but",),
                ("bad",),
                ("in",),
                ("a",),
                ("fun",),
                ("way",),
                (",",),
                ("so",),
                ("I",),
                ("loved",),
                ("it",),
                (".",),
            ],
        return_token_ids = True
        self.n = 3
        Input: "This movie was really bad, but bad in a fun way, so I loved it."
        Output: [16999, 51610, 39000, 44191, 89954, 14539, 50931]
        """
        return_vals = []
        text_in_words = convert_text_to_words(text)
        for index in range(len(text_in_words) - (self.n - 1)):

            n_gram = tuple(text_in_words[index:(index + self.n)])
            # n_gram = tuple(n_gram)
            if n_gram in self.token_to_id:
                if return_token_ids:
                    return_vals.append(self.token_to_id[n_gram])  # Map to token ID
                else:
                    return_vals.append(n_gram)

        
        return return_vals


    def train(self, corpus: List[str]):
        n_gram_frequency = {}
        count = 0
        
        for text in corpus:

            text_in_words = convert_text_to_words(text)

            for index in range(len(text_in_words) - (self.n - 1)):

                n_gram = []
                for i in range(self.n):
                    n_gram.append(text_in_words[index + i])
                n_gram = tuple(n_gram)
                if n_gram in n_gram_frequency.keys():
                    n_gram_frequency[n_gram] += 1
                else:
                    n_gram_frequency[n_gram] = 1
                    self.token_to_id[n_gram] = count
                    self.id_to_token[count] = n_gram
                    count += 1
        
        count = 0

        if self.vocab_size != -1:
            n_gram_frequency = list(dict(sorted(n_gram_frequency.items(), key=lambda item: item[1], reverse=True)).keys())
            n_gram_frequency = n_gram_frequency[:self.vocab_size]
            for n_gram in list(self.token_to_id.keys())[:]:
                if n_gram in n_gram_frequency:
                    self.token_to_id[n_gram] = count
                    self.id_to_token[count] = n_gram
                    count += 1
                else:
                    temp = self.token_to_id[n_gram]
                    del(self.token_to_id[n_gram])
                    del(self.id_to_token[temp])
        


    def __len__(self):
        return len(self.token_to_id.keys())

if __name__ == "__main__":
    with jsonlines.open("data/imdb_train.txt", "r") as reader:
        dataset = list(reader)

    dataset = dataset[:500]

    corpus = [datapoint["text"] for datapoint in dataset]

    unigram = NgramTokenizer(n=1)
    unigram.train(corpus)

    ngram = NgramTokenizer(n=2)
    ngram.train(corpus)

    sample_text = "I love scifi and am willing to put up with a lot. Scifi movies and TV are usually underfunded, under-appreciated and misunderstood."

    # ngram = NgramTokenizer(n=3)
    # ngram.train(corpus)

    # sample_text = "This movie was really bad, but bad in a fun way, so I loved it."

    # print(ngram.tokenize(sample_text, True))

    print(unigram.tokenize(sample_text))
    print("-" * 100)
    print(ngram.tokenize(sample_text))
    print("-" * 100)
