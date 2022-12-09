

# n gram model 

from collections import Counter
import numpy as np

class Ngram:
    def __init__(self, N) -> None:
        
        self.N = N # TODO
        self.counts = {i: Counter() for i in range(1, N+1)}
        self.word = {i: [] for i in range(1, N+1)}

    def _log_prob(self, n_gram):
        # the number of n-gram `w_n, ..., w_1` divided by the number of (n-1)-gram `w_{n-1}, ..., w_1`.
        
        n = len(n_gram)
        return np.log( self.count[n](n_gram)) -np.log(self.count[n-1](n_gram[:-1]))

    def _log_prob(self, words, N): # correction
        # N: N gram model
        # words: input sequence

        assert N in self.counts
        if N > len(words):
            err = "Not enough words for a gram-size of {}:{}",format(N, len(words))
            raise ValueError(err)
        
        total_prob = 0
        for ngram in ngrams(words,N):
            total_prob += self._log_ngram_prob(ngram)
        
        return total_prob


    def cross_entropy(words, N):
        y_hat = self.predict(words)
        return y * np.log(y_hat)

    def perplexity(self, seq):
        np.exp(np.cross_entropy)

        prod  = 1
        count = Counter(seq)
        n = len(seq)
        for s in seq:
            prod *= self.count[s]/n

        return np.



# tokenize:

words = tokenize_words(line, filter_stopwords=filter_stop)

words_padded = [eok*]