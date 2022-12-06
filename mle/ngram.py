import numpy as np
from collections import Counter

def ngrams(sequence, N):
    """
    compute all the N grams for input sequence (list of strings)
    Parameters:
    -----------
    sequence: list of strings
    N: the N in 'ngram'
    Returns:
    --------
    return list of tuples of ngrams. e.g. "[('Split', 'a', 'string', 'into'), ('a', 'string', 'into', 'individual'), ('string', 'into', 'individual', 'words,'),... ]"
    """

    return list(zip(*[sequence[i:] for i in range(N)])) # return list of tuple of n grams, the tuple can be used in the keys of counters


class NGramBase:
    def __init__(self, N) -> None:
        self.N = N
    
    def _train(self, words, vocab=None, encoding=None):
        """Actual N-gram training logic"""
        # H = self.hyperparameters
        grams = {N: [] for N in range(1, self.N + 1)}
        counts = {N: Counter() for N in range(1, self.N + 1)}
        # filter_stop, filter_punc = H["filter_stopwords"], H["filter_punctuation"]

        _n_words = 0
        tokens = {"<unk>"}
        bol, eol = ["<bol>"], ["<eol>"]

        # calculate n, n-1, ... 1-grams
        for N in range(1, self.N + 1):
            words_padded = bol * max(1, N - 1) + words + eol * max(1, N - 1)
            grams[N].extend(ngrams(words_padded, N))

        for N in counts.keys():
            counts[N].update(grams[N])

        n_words = {N: np.sum(list(counts[N].values())) for N in range(1, self.N + 1)}
        n_words[1] = _n_words

        n_tokens = {N: len(counts[N]) for N in range(2, self.N + 1)} # counts of unique ngrams
        n_tokens[1] = len(vocab) if vocab is not None else len(tokens) # 

        self.counts = counts
        self.n_words = n_words
        self.n_tokens = n_tokens

    def _n_completions(self, words, N):
        """
        Return the number of unique word tokens that could follow the sequence
        `words` under the *unsmoothed* `N`-gram language model.
        """
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        assert len(words) <= N - 1, "Need > {} words to use {}-grams".format(N - 2, N) # words here refers to a sequence

        if isinstance(words, list):
            words = tuple(words)

        base = words[-N + 1 :]
        return len([k[-1] for k in self.counts[N].keys() if k[:-1] == base])

    def completions(self, words, N):
        """
        Return the distribution over proposed next words under the `N`-gram
        language model.
        Parameters
        ----------
        words : list or tuple of strings
            The initial sequence of words
        N : int
            The gram-size of the language model to use to generate completions
        Returns
        -------
        probs : list of (word, log_prob) tuples
            The list of possible next words and their log probabilities under
            the `N`-gram language model (unsorted)
        """
        N = min(N, len(words) + 1)
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        assert len(words) >= N - 1, "`words` must have at least {} words".format(N - 1)

        probs = []
        base = tuple(w.lower() for w in words[-N + 1 :])
        for k in self.counts[N].keys():
            if k[:-1] == base:
                c_prob = self._log_ngram_prob(base + k[-1:])
                probs.append((k[-1], c_prob))
        return probs

    def _log_prob(self, words, N):
        """
        Calculate the log probability of a sequence of words under the
        `N`-gram model
        """
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        total_prob = 0
        for ngram in ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob

    def _log_ngram_prob(self, ngram):
        """Return the unsmoothed log probability of the ngram"""
        N = len(ngram)
        num = self.counts[N][ngram]
        den = self.counts[N - 1][ngram[:-1]] if N > 1 else self.n_words[1]
        return np.log(num) - np.log(den) if (den > 0 and num > 0) else -np.inf



words = "Split a string into individual words, optionally removing punctuation and stop-words in the process."

N = 4
seq_words = words.split()
Ngram = NGramBase(N)
Ngram._train(seq_words)

arr = ngrams(seq_words, N) # [('Split', 'a', 'string', 'into'), ('a', 'string', 'into', 'individual'), ('string', 'into', 'individual', 'words,'), ('into', 'individual', 'words,', 'optionally'), ('individual', 'words,', 'optionally', 'removing'), ('words,', 'optionally', 'removing', 'punctuation'), ('optionally', 'removing', 'punctuation', 'and'), ('removing', 'punctuation', 'and', 'stop-words'), ('punctuation', 'and', 'stop-words', 'in'), ('and', 'stop-words', 'in', 'the'), ('stop-words', 'in', 'the', 'process.')]
print("Done")