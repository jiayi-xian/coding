1 3.1 明确标注任务

前文讲过，NER可以根据业务需求标注各种不同类型的实体，因此首先要明确需要抽取的实体类型。一般通用场景下，最常提取的是时间、人物、地点及组织机构名，因此本任务提取TIME、PERSON、LOCATION、ORGANIZATION四种实体。

3.2 数据及工具准备

明确任务后就需要训练数据和模型工具。对于训练数据，我们使用经典的人民日报1998中文标注语料库，其中包括了分词和词性标注结果，下载地址为：http://icl.pku.edu.cn/icl_groups/corpus/dwldform1.asp。对于CRF，有很多开源的工具包可供选择，在此使用CRF++进行训练。CRF++官方主页为CRF++: Yet Another CRF toolkit，包括下载及使用等说明。

3.3 数据预处理

人民日报1998语料库下载完毕后，解压打开“199801.txt”这个文件（注意编码转换成UTF-8），可以看到内容是由word/pos组成，中间以两个空格隔开。我们需要的提取的实体是时间、人名、地名、组织机构名，根据1998语料库的词性标记说明，对应的词性依次为t、nr、ns、nt。通过观察语料库数据，需要注意四点：1，1998语料库标注人名时，将姓和名分开标注，因此需要合并姓名；2，中括号括起来的几个词表示大粒度分词，表意能力更强，需要将括号内内容合并；3，时间合并，例如将”1997年/t 3月/t” 合并成”1997年3月/t”；4，全角字符统一转为半角字符，尤其是数字的表示。


Preprocessing is where text data is cleaned and processed via NLP tasks and is a preparatory task for feature processing. First, the text data is cleansed by removing non informative characters and replacing special characters with corresponding spellings. The text is then tokenized with a tokenization tool. We evaluated two different tokenization strategies: a simple white space tokenizer and the BANNER simple tokenizer. The white space tokenizer splits the text simply, based on blanks within it, whereas the BANNER tokenizer breaks tokens into either a contiguous block of letters and/or digits or a single punctuation mark. Finally, the lemma and the part-of-speech (POS) information were obtained for a further usage in the feature extraction phase. In BANNER-CHEMDNER, BioLemmatizer [27] was used for lemma extraction, which resulted in a sig nificant improvement in overall system performance. In addition to these preprocessing steps, special care is taken to parse the PMC XML documents to get the full text for the unlabeled data collection. 


We extract features from the preprocessed text to repre-
sent each token as a feature vector, and then an ML
algorithm is employed to build a model for NER.

The proposed method includes extraction of the base-
line and the word representation feature sets. The base-
line feature set is essential in NER, but is poor at
representing the domain background because it only
carries some morphological and shallow-syntax informa-
tion of words. On the other hand, the word representa-
tion features can be extracted by learning on a large
amount of text and may be capable of introducing
domain background to the NER model.
The entire feature set for a token is expanded to
include features for the surroundings with a two-length
sliding window. The word, the word n-gram, the charac-
ter -gram, lemma and the traditional orthographic
information are extracted as the baseline feature set.
The regular expressions that reveal orthographic infor-
mation are matched to the tokens to give orthographic
information. These baseline features are summarized in
Table 1.
For word representation features, we train Brown clus-
tering models [28] and Word Vector (WV) models [17]
on a large PubMed and PMC document collection.
Brown clustering is a hierarchical word clustering
method, grouping words in an input corpus to maximize
the mutual information of bigrams. Therefore, the qual-
ity of a partition can be computed as a sum of mutual
information weights between clusters. It runs in time /
(V × K3), where V is the size of the vocabulary and K is
the number of clusters.


The VW model is induced via a Recurrent Neural
Network (RNN) and can be seen as a language model
that consists of n-dimensional continuous valued vec-
tors, each of which represents a word in the training
corpus. The RNN instance is trained to predict either
the middle word of a token sequence captured in a win-
dow (CBOW) or surrounding words given the middle word 
of the sequence (skip-gram) depending on the
model architecture [17]. The RNN becomes a log-linear
classifier, once its non-linear hidden layer is removed, so
the training process speeds up allowing millions of
documents to process within an hour. We used a tool
implemented by Mikolov et al. [17] to build our WV
model from the PubMed collection.
Further, the word vectors are clustered using
K-means algorithm to drive a Word Vector Class (WVC)
model. Since Brown clustering is a bigram model, this
model may not be able to carry wide context information
of a word, whereas the WVC model is an n-gram model
(usually n
= 5) and learns broad context information
from the domain corpus. We drive the cluster label pre-
fixes with 4, 6, 10 and 20 lengths in the Brown model by
following the experiment of Turian et al. [16], and the
WVC models induced from 250-dimension WVs as word
representation features.
For feature extraction, we do not rely on any lexicon
nor any dictionary other than the free text in the domain
in order to keep the system applicable to other NER
tasks in bio-text data, even though the usage of such
resources is reported to considerably boost system per-
formance. Most of the top performing systems partici-
pated in CHEMDNER task use the domain lexicon and
observed a considerable performance boost [29].


def nosiy_allreduce(send_funcs, receive_funcs, value):
    N = len(send_funcs)
    v = value
 

    for s_f in send_funcs:
        s_f(v)

    for r_f in receive_funcs:
        sv += r_f() # E(error) = 0

    for s_f in send_funcs:
        s_f(sv)
    
    sv2 = sv

    for r_f in receive_funcs:
        sv2 += r_f()

    return sv2/N

