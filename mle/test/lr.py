

import numpy as np
import pandas as pd

def euclid_dist(X1, X2):
    """
    
    Parameters:
    -----------
    X1, X2 : (N, M)

    Returns:
    --------
    
    """

    return np.linalg.norm(X1-X2, ord=2, axis = 1)

class LinerRegression:

    def __init__(self, fit_intercept):

        self.fit_intercept = fit_intercept
        self.beta = None
        self.sigma_inv = None
        self._is_fit = False

    def fit(self, X, Y):

        N, M = X.shape[0], X.shape[1]
        # X, Y = np.atleast_2d(X), np.atleast_2d(Y)

        if self.fit_intercept:
            X = np.c_(np.ones(N), X) # -> X = np.c_[np.ones(N), X]
        
        self.sigma_inv = np.linalg.pinv(X.T @ X)
        self.beta = self.sigma_inv @ X.T @ Y
        self._is_fit = True


    def predict(self, X):

        return X @ self.beta


    def update(self, X, Y):
        # X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        N, M = X.shape[0], X.shape[1]
        if self.fit_intercept:
            X = np.c_(np.ones(N), X)

        beta = self.beta
        S_inv = self.sigma_inv
        I = np.eyes(N) #  I = np.eye(X.shape[0])

        S_inv -= S_inv @ X.T @ np.linalg.pinv(I + X @ X.T) @ X @ S_inv   # np.linalg.pinv(I + X @ S_inv @ X.T)
        Y_pred = X @ beta
        beta += S_inv @ X.T @ (Y-Y_pred)


def _sigmoid(X):
    return 1/(1 + np.exp(-X))

class LogisticRegression:

    def __init__(self, fit_intercept, penalty, gamma) -> None:
        
        self.fit_intercept = fit_intercept
        self.beta = None
        self.gamma = None # gamma is the penalty factor
        self.penalty = penalty

    
    def predict(self, X): # correction

        #return _sigmoid(X @ self.beta / self.gamma)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return _sigmoid( X @ self.beta)

    def _NLL(self, X, Y, Y_pred):
        #Y_pred = self.predict(X)
        #Y_pred[Y == 1] + (1 - Y_pred)[ Y == 0]

        N, M = X.shape[0], X.shape[1]
        beta, gamma = self.beta, self.gamma

        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, order=order)
        nll = np.log(Y_pred[Y==1]).sum() + np.log(1 - Y_pred[Y == 0]).sum()
        penalty = (gamma / 2) * norm_beta **2 if order == 2 else gamma * norm_beta

        return (nll + penalty) / N

    def _NLL_grad(self, X, Y, Y_pred):
        N, M = X.shape[0], X.shape[1]
        beta, gamma = self.beta, self.gamma

        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, order=order)

        d_penalty = gamma * norm_beta if order == 2 else gamma * np.sign(beta)

        # d_sigmoid = (1 - Y_pred) * Y_pred @ X ???
        d_sigmoid = (Y - Y_pred) @ X

        return -(d_sigmoid + d_penalty) /N

    def fit(self, X, Y):

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        i = 0
        """while i < max_iter:
            Y_pred = self.predict(X)
            self.beta += lr * self._NLL_grad(X, Y, Y_pred)"""

        while i< max_iter:
            Y_pred = _sigmoid(X @ self.beta)
            loss  = self._NLL(X, Y, Y_pred)
            if np.abs(loss - l_prev) < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, Y, Y_pred)

class KNN:

    def __init__(self, K, X, Y) -> None:

        self.K = K
        self.X = X
        self.Y = Y

    def fit(self, X, Y):
        pass

    def predict(self, x):

        dists = np.linalg.norm(self.X - x, order = 2)
        idx_ranked = np.argsort(dists)
        counts = np.bincount(self.Y[idx_ranked[:self.K]])
        return np.argmax(counts)


class Kmeans:

    def __init__(self):
        pass


def Ngrams(sequence, N):

    return list(zip(*[sequence[i:] for i in range(N)]))

from collections import Counter
class Ngram:

    def __init__(self, N) -> None:
        self.N = N

    
    def fit(self, text):

        bok, eok = '<bok>', '<eok>'

        N = self.N
        self.counts = {i:Counter() for i in range(1, N+1)}
        self.ngrams = {i:[] for i in range(1, N+1)}
        self.vocab = set([bok, eok])

        
        for line in text:

            self.vocab.add(line.split())

            for n in range(1, N+1):
                line = [bok] * max(n-1, 1) + line + [eok] * max(n-1, 1)

                ngrams = Ngrams(line, n)
                self.ngrams[n].extend(ngrams)

            for n in range(1, N+1):
                self.counts[n].update(ngrams)

        self.n_words = {i: len(self.ngrams[i]) for i in range(1, N+1)}
        self.n_tokens = {i: np.sum(list(self.counts[i].values())) for i in range(1, N+1)}
        
        return self

    
    def _log_prob_ngram(self, ngram):

        n = len(ngram)

        den = self.counts[n-1][ngram[:-1]] if n >1 else len(self.words[1]) # -> self.n_words[1]
        num = self.counts[n][ngram]

        return np.log(num) - np.log(den) if num >0 and den>0 else -np.inf

    def _log_prob_seq(self, seq, N):

        if len(seq) < N:
            raise ValueError

        else:
            prob = 0
            for ngram in Ngrams(seq, N):
                prob += self._log_prob_ngram(ngram)

        return prob

    def completion(self, seq):
        pass

class DecisionTree:

    def __init__(self, max_depth, n_feats, criterion = 'gini', classifier=True) -> None:

        self.root = None
        self.max_depth = max_depth
        self.criterion = criterion
        self.classifier = classifier

        # current depth missing
        self.depth = 0

        # missing n_feat
        self.n_feats = n_feats

    def _traverse(self, x, node):

        if isinstance(node, Leaf):
            return node.value if self.classifier else np.argmax(node.value)

        if x[node.feature] <= node.threshold:
            self._traverse(x, node.left)
        else:
            self._traverse(x, node.right)
    
    def _grow(self, X, Y, depth=0):

        # case 1: exceeds max_depth
        if self.depth >= self.max_depth:
            v = np.mean(Y)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes)/len(Y)
            
            return Leaf(v)

        # current set leaves 1 element / threshold elements
        if len(set(Y)) == 1:
            v = np.zeros(self.n_classes)
            v[Y[0]] = 1.0
            return Leaf(v) if self.classifier else Leaf(Y[0])
            
        
        """
        # if all labels are the same, return a leaf
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            
            return Leaf(prob) if self.classifier else Leaf(Y[0])
            

        # if we have reached max depth, return a leaf
        if cur_depth >= self.max_depth:
            v = np.mean(Y)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v) # In the leaf node, there is a probability that the input classified into any class
        """
        N, M = X.shape[0], X.shape[1]
        cur_depth += 1
        self.depth = max(cur_depth, self.depth)

        feats = np.random.choice(M, self.n_feats, replace=False)

        thresh, feat = self._segment(X[:, feats], Y, feats)
        left = np.argwhere(X[:, feat]<=thresh).flatten()
        right = np.argwhere(X[:, feat]>thresh).flatten()

        node = Node(thresh, feat)
        node.left = self._grow(X[left], Y[left], depth+1)
        node.right = self._grow(X[right], Y[right], depth+1)

        return node


    def _segment(self, X, Y, feats):

        """
        best_gain, split_idx = -np.inf, feat_idxs[0]
        for i in  feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[1:] + levels[:-1])/2 if len(levels>1) else levels #[1,2,3,4] -> ([1,2,3] + [2,3,4]) /2 = [1.5, 2.5, 3.5]
            gains = np.array([self._impurity_gain(Y, thresh, X[:, i]) for thresh in thresholds])
        
            best_cur = gains.max()
            if best_cur > best_gain:
                best_gain = best_cur
                split_thresh = thresholds[ np.argmax(gains)]
                split_idx = i
        
        return split_idx, split_thresh
        """

        best_feat = None
        best_gain = 0
        best_thresh = None

        for idx in feats:

            values = np.unique(X[idx]).sort()
            #levels = (values[:-1] + values[1:]) /2
            threshes = (values[:-1] + values[1:]) /2 if len(values) > 1 else values

            gains = np.array([self._impunity_gain(thresh, idx, X, Y) for thresh in threshes])
            max_gain = np.max(gains)
            if max_gain > best_gain:
                best_gain = max_gain
                best_feat = idx
                best_thresh = threshes[np.argmax(gains)]

        return best_thresh, best_feat


    def _impunity_gain(self, thresh, feat, X, Y):

        if loss == "mse":
            loss = dt_mse
        elif loss == "gini":
            loss = dt_gini
        elif loss == "entropy":
            loss = dt_entropy
        else:
            raise NotImplementedError

        parent_loss = loss(Y)
        l = np.argwhere(X[:, feat]<=thresh).flatten()
        r = np.argwhere(X[:, feat]>thresh).flatten()
        e_l, e_r = loss(Y[l]), loss(Y[r])
        n_l, n_r = len(l), len(r)
        n = len(Y)
        child_loss = (n_l/n) * e_l + (n_r/n) * e_r
        ig = parent_loss - child_loss

        return ig

        """
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l/n) * e_l + (n_r/n) * e_r

        ig = parent_loss - child_loss

        return ig
        """

        return loss_child - loss_parent

    def dt_gini(self, Y):
        num = np.bincount(Y)/len(Y)
        return 1 - np.sum(num**2)

    def dt_entropy(self, Y):
        ps = np.bincount(Y)/len(Y)
        return -np.sum(ps*np.log2(ps)) # -> -np.sum([p*np.log2(p) for p in ps if p>0])
    
    def dt_mse(self,Y):
        return np.sqrt(np.sum((Y-np.mean(Y))**2)) # -> np.mean((Y-np.mean(Y))**2)


    """
    def _traverse(self, X, node, prob = False):

        if isinstance(node, Leaf):
            if self.classifier:
                #idx = np.argmax(np.random.choice(len(node.value), 1, node.value))
                return node.value if prob else node.value.argmax()
            return node.value


        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        else:
            return self._traverse(X, node.right, prob)
    
    """

            
    """
    def fit(self, X, Y):

        self.n_classes = max(Y) + 1
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self._grow(X, Y)
    """


    def _grow(self, X, Y, depth=0):

        n_classes = X.shape[1] # n_classes = max(Y) + 1, 
        # self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        if self.depth >= self.max_depth:
            v = np.mean(Y)
            if self.classifier:
                v = np.bincount(Y, minlength=n_classes)
            return Leaf(v)

        
        if len(set(X)) == 1:
            v = Y[0]
            if self.classifier:
                return Leaf(v)
            else:
                pass


class CBOW:

    def __init__(self, window, text, vocab, word_dim) -> None:

        self.window = window
        self.text = text
        self.vocab = vocab
        self.word_index = { k:v for v, k in enumerate(vocab)}
        self.word_dim = word_dim
        self.U = np.random.uniform(-0.001, 0.001, (len(self.vocab), self.word_dim))
        self.V = np.random.uniform(-0.001, 0.001, (self.word_dim, len(self.vocab)))
    
    def fit(self):

        for i, word in enumerate(self.text):
            center_word = word
            if i < self.window:
                left_window, right_window = self.text[:i], self.text[i+1 : i+self.window+1]
            else:
                left_window, right_window = self.text[i-self.window: i]
            
            aver_vec = self.get_aver_context_vec(left_window, right_window)

            score = self.get_score(self.U, aver_vec)
            y_hat = self.softmax(score)
            idx = self.word_index(center_word)
            loss = self.entropy(self.get_rep_from_onehot(idx), y_hat)
            
            self.updateU(loss, aver_vec = aver_vec)
            self.updateV(loss, left_context=left_window, right_context=right_window)

    def computEH(self, error):
        EH = np.zeros(self.word_dim)
        for i in range(self.word_dim):
            EH[i] = np.sum(self.U[j, i] * error[j] for j in range(len(self.vocab)))

        return EH

    def updateU(self, error, aver_vec):
        for i in range(len(self.vocab)):
            self.U[i, :] -= self.learning_rate * error * aver_vec
    
    def updateV(self, error, left_context, right_context):
        EH = self.computEH(error)
        window_size = len(left_context) + len(right_context)
        for word in left_context+right_context:
            j = self.word_index(word)
            self.V[:, j] -= window_size*self.learning_rate* EH

    def softmax(self, Z):

        Z -= Z.max(axis = 1) # .reshape(-1, 1)
        return np.exp(Z)/np.sum(np.exp(Z), axis = 1) # .reshape(-1, 1)

    def entropy(self, p, q):
        return np.sum(-p * np.log2(q), axis=1)

    
    def get_score(self, aver_vec):

        return np.dot(self.U, aver_vec)

    def get_rep_from_onehot(self, one_hot):

        return np.dot(self.V, one_hot)
    
    def get_onehot(self, i):
        v = np.zeros(len(self.vocab))
        v[i] = 1.0
        return v

    def get_aver_context_vec(self, left_window, right_window):

        aver_vec = 0
        for word in left_window:
            onehot = self.get_onehot(self.word_index[word])
            v = self.get_rep_from_onehot(onehot)
            aver_vec += v
        
        for word in right_window:
            onehot = self.get_onehot(self.word_index[word])
            v = self.get_rep_from_onehot(onehot)
            aver_vec += v
        
        return aver_vec / (len(left_window)+len(right_window))
    

from torch import nn
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v) -> None:
        
        self.d_k, self.d_v, self.n_head = d_k, d_v, n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.noraml_(self.w_qs.weight, mean=0, std=np.sqrt(2.0/(d_k+d_model)))
        nn.init.noraml_(self.w_ks.weight, mean=0, std=np.sqrt(2.0/(d_k+d_model)))
        nn.init.noraml_(self.w_vs.weight, mean=0, std=np.sqrt(2.0/(d_k+d_model)))

        self.attention = ScaledDotProductAttention(temperature = np.power(d_k, 0.5))

        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head*d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight) ###
        self.dropout = nn.Dropout(dropout)

        # outpu -> fc -> dropout -> add and norm
    
    def forward(self, q, k, v):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residuel = q
        q = self.w_qs(q).view(bsz, len_q, n_head, d_k)
        k = self.w_ks(k).view(bsz, len_q, n_head, d_k)
        v = self.w_vs(v).view(bsz, len_q, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous.view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous.view(-1, len_q, d_k)
        v = v.permute(2, 0, 1, 3).contiguous.view(-1, len_q, d_k)

        mask = mask.repeat(n_head, 1, 1) # mask.shape: (n_head, lenq, d_k)
        output, atten = self.attention(q, k , v, mask=mask)

        output = output.view(n_head, bsz, len_q, d_v)
        output = output.permute(1,2,0,3).contiguous().view(bsz, len_q, -1)

        output = self.dropout(self.fc(output))

        output += residuel

        output = self.layernorm(output)

        return output, atten

    class ScaledDotProductAttention(nn.Module):

        def __init__(self) -> None:
            super(ScaledDotProductAttention, self).__init__()

            self.temperature = temperature # np.power(d_k, 0.5)
            self.dropout = nn.Dropout(attn_dropout)
            self.softmax = nn.Softmax(dim=2)

        def forward(self, q, k, v, mask=None):

            attn = torch.bmm(q, k.transpose(1,2)) # q, k: bsz, lenq, d_q (or d_k) -> bsz, lenq, lenk
            attn = attn / self.temperature

            if mask is not None:
                attn = attn.masked_fill(mask, -np.inf)

            attn = self.softmax(dim=2)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)

            return output, attn

import re

# https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/preprocessing/nlp.py

_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")  # sklearn default
_WORD_REGEX_W_PUNC = re.compile(r"(?u)\w+|[^a-zA-Z0-9\s]") 
""" []
Used to indicate a set of characters. In a set:

Characters can be listed individually, e.g. [amk] will match 'a', 'm', or 'k'."""
_WORD_REGEX_W_PUNC_AND_WHITESPACE = re.compile(r"(?u)s?\w+\s?|\s?[^a-zA-Z0-9\s]\s?")

_PUNC_BYTE_REGEX = re.compile(
    r"(33|34|35|36|37|38|39|40|41|42|43|44|45|"
    r"46|47|58|59|60|61|62|63|64|91|92|93|94|"
    r"95|96|123|124|125|126)",
)
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_PUNC_TABLE = str.maketrans("", "", _PUNCTUATION) 
# str1: replacement str2: to be replaced str3: to be deleted, return: mapping table of Unicode characters



_WORD_REGEX = re.compile(r"(?u)\b\w+\b")  # sklearn default
_WORD_REGEX_W_PUNC = re.compile(r"(?u)\w+|[^a-zA-Z0-9\s]") 
""" []
Used to indicate a set of characters. In a set:
Characters can be listed individually, e.g. [amk] will match 'a', 'm', or 'k'."""
_WORD_REGEX_W_PUNC_AND_WHITESPACE = re.compile(r"(?u)s?\w+\s?|\s?[^a-zA-Z0-9\s]\s?")
_STOP_WORDS = ""


# how to remove punctuation from a string:https://datagy.io/python-remove-punctuation-from-string/ 

def remove_stop_words(words):
    """Remove stop words from a list of word strings"""
    return [w for w in words if w.lower() not in _STOP_WORDS]


def strip_punctuation(line):
    """Remove punctuation from a string"""
    return line.translate(_PUNC_TABLE).strip()

def tokenize_chars(line, lowercase=True, filter_punctuation=True, **kwargs):
    """
    Split a string into individual characters, optionally removing punctuation
    and stop-words in the process.
    """
    line = line.lower() if lowercase else line
    line = strip_punctuation(line) if filter_punctuation else line
    chars = list(re.sub(" {2,}", " ", line).strip()) # reduce multiple spaces to one space
    return chars

def tokenize_words(
    line, lowercase=True, filter_stopwords=True, filter_punctuation=False, **kwargs,
):
    """
    Split a string into individual words, optionally removing punctuation and
    stop-words in the process.
    """
    REGEX = _WORD_REGEX_W_PUNC_AND_WHITESPACE
    words = REGEX.findall(line.lower() if lowercase else line)
    return remove_stop_words(words) if filter_stopwords else words

tokenize_words("Today is a good day !@#$$! !#$")

import logging
# error and exception
def error_and_exception():
    try:
        bar('0')
    except Exception as e:
        logging.exception(e)

    
    try:
        print('try...')
        r = 10 / int('2')
        print('result:', r)
    except ValueError as e:
        print('ValueError:', e)
    except ZeroDivisionError as e:
        print('ZeroDivisionError:', e)
    else:
        print('no error!')
    finally:
        print('finally...')
    print('END')

class Building(object):
     def __init__(self, floors):
         self._floors = [None]*floors
     def __setitem__(self, floor_number, data):
          self._floors[floor_number] = data
     def __getitem__(self, floor_number):
          return self._floors[floor_number]

"""
once you have a getitem you don't have to explicitly call that function. When he calls building1[2] that call itself internally calls the getitem. So the point @tony-suffolk-66 is making is that, any property/variable of the class can be retrieved during run time by simply calling objectname[variablename]


The __lt__ "dunder" method is what allows you to use the < less-than sign for an object. It might make more sense written as follows:
"""
    
class distance:
  def __init__(self, x=5,y=5):
    self.ft=x
    self.inch=y

  def __eq__(self, other):
    if self.ft==other.ft and self.inch==other.inch:
      return "both objects are equal"
    else:
      return "both objects are not equal"

  def __lt__(self, other):
    in1=self.ft*12+self.inch
    in2=other.ft*12+other.inch
    if in1<in2:
      return "first object smaller than other"
    else:
      return "first object not smaller than other"

  def __gt__(self, other):
    in1=self.ft*12+self.inch
    in2=other.ft*12+other.inch
    if in1<in2:
      return "first object greater than other"
    else:
      return "first object not greater than other"


class SparseMatrix:
    # 没有的值key都是0 0的位置不记录 在multiplication后有清零的操作
    def __init__(self, nrow, ncol) -> None:
        self.sm = {}
        self.nrow = nrow
        self.ncol = ncol

    def __getitem__(self, key):
        assert isinstance(key, tuple) # isinstance(object, classinfo)
        if len(key) != 2 or not (0 <= key[0] < self.nrow and 0 <= key[1] < self.ncol):
            raise IndexError("please check index")
        
        return self.sm[key] if key in self.sm.keys() else 0 ## TODO： 不是None或者False

    def __setitem__(self, key: list, val):
        
        # val == 0, key exist
        # val != 0, key exist
        # val == 0, key doesnt exist
        # val != 0, key doesnt exist
        # key out of range
        assert isinstance(key,tuple) # isinstance(object, classinfo)
        if len(key) != 2 or not (0 <= key[0] < self.nrow and 0 <= key[1] < self.ncol):
            raise IndexError("please check index")

        self.sm[key] = val
        if val == 0:
            del self.sm[key]

    def dot_product(self, sm2):
        sm1 = self

        assert sm1.nrow == sm2.nrow and sm1.ncol == sm2.ncol
        sm3 = SparseMatrix(sm1.nrow, sm1.ncol)
        
        if len(sm1.sm) > len(sm2.sm):
            return sm2.multiply(sm1)

        for key in sm1.sm:
            if key in sm2.sm:
                sm3[key] = sm1[key] * sm2[key]
        
        return sm3

    def matrix_product(self, sm2):
        sm1 = self
        assert sm1.ncol == sm2.nrow
        
        sm3 = SparseMatrix(sm1.nrow, sm2.ncol)

        for key, val in sm1.sm.items():
            i, j = key
            for k in range(sm2.ncol):
                if (j, k) in sm2.sm:
                    sm3[(i, k)] += sm2[(j, k)] * val
        
        for key, val in sm3.sm.items():
            if val == 0:
                del sm3[key]

        return sm3
        
sm1 = SparseMatrix(3, 4)
sm2 = SparseMatrix(3, 4)
from itertools import product
for i, j in product(range(sm1.nrow), range(sm1.ncol)): # not zip !!!
    sm1[(i, j)] = i + j * 0.1
for i, j in product(range(sm2.nrow), range(sm2.ncol)):
    sm2[(i, j)] = i + 1 + j * 0.2

print(sm1.matrix_product(sm2))