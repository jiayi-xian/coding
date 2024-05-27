import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras_preprocessing.sequence import pad_sequences

from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM


def data_process(data_path, samp_rows=10000):
    data = pd.read_csv(data_path, nrows=samp_rows)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train = data.iloc[:int(len(data)*0.8)].copy()
    test = data.iloc[int(len(data)*0.8):].copy()
    return train, test, data


def get_company_feature(data):
    """
    sparse:
    #"cohort_id",
    "company_hq_state", "Unknown"
    "deal_type", "Unknown"
    "series"
    dense: "size"
    deal_features_text: 
        "industry_code_topk", 
        "industry_group_topk",
        "industry_sector_topk",
        # "description"
    """
    # data_group = data[data['rating'] == 1] # TODO ! 我们这里id全是string
    data_group = data_group[['company_id', 'investor_id']].groupby('company_id').agg(list).reset_index()
    data_group['company_hist'] = data_group['investor_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    # ? concatenate investor id as history
    data = pd.merge(data_group.drop('investor_id', axis=1), data, on='company_id')
    # ? get deal size median or weighted mean or ....
    data_group = data[['company_id', 'deal_size']].groupby('company_id').agg('mean').reset_index()
    data_group.rename(columns={'deal_size': 'deal_mean_size'}, inplace=True)
    data = pd.merge(data_group, data, on='company_id')
    return data

def get_investor_feature(data):
    """
    "series_deal_count",
    "industry_code",
    "industry_sector", 
    "industry_group", 
    """
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'investor_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
    return data


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("company_hist_list: \n")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


if __name__ == '__main__':
    ## %%
    data_path = './data/movielens.txt'
    train, test, data = data_process(data_path, samp_rows=10000)
    train = get_company_feature(train)
    train = get_investor_feature(train)

    sparse_features = ['company_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features = ['company_mean_rating', 'investor_mean_rating']
    target = ['rating']

    company_sparse_features, company_dense_features = ['company_id', 'gender', 'age', 'occupation'], ['company_mean_rating']
    investor_sparse_features, investor_dense_features = ['movie_id', ], ['investor_mean_rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    train[dense_features] = mms.transform(train[dense_features])

    # 2.preprocess the sequence feature
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    company_key2index, train_company_hist, company_maxlen = get_var_feature(train, 'company_hist')

    company_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(company_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               company_dense_features]
    investor_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(investor_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               investor_dense_features]

    investor_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=4),
                                                    maxlen=genres_maxlen, combiner='mean', length_name=None)]

    company_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('company_hist', vocabulary_size=3470, embedding_dim=4),
                                                    maxlen=company_maxlen, combiner='mean', length_name=None)]

    # 3.generate input data for model
    company_feature_columns += company_varlen_feature_columns
    investor_feature_columns += investor_varlen_feature_columns

    # add company history as company_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    train_model_input["genres"] = train_genres_list
    train_model_input["company_hist"] = train_company_hist

    ## %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DSSM(company_feature_columns, investor_feature_columns, task='binary', device=device)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])

    # %%
    model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    # model.save

    # %%
    # 5.preprocess the test data
    test = pd.merge(test, train[['movie_id', 'investor_mean_rating']].drop_duplicates(), on='movie_id', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['company_id', 'company_mean_rating']].drop_duplicates(), on='company_id', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['company_id', 'company_hist']].drop_duplicates(), on='company_id', how='left').fillna('1')
    test[dense_features] = mms.transform(test[dense_features])

    test_genres_list = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
    test_company_hist = get_test_var_feature(test, 'company_hist', company_key2index, company_maxlen)

    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    test_model_input["genres"] = test_genres_list
    test_model_input["company_hist"] = test_company_hist

    # %%
    # 6.Evaluate
    eval_tr = model.evaluate(train_model_input, train[target].values)
    print(eval_tr)

    # %%
    pred_ts = model.predict(test_model_input, batch_size=2000)
    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))

    # %%
    # 7.Embedding
    print("company embedding shape: ", model.company_dnn_embedding[:2])
    print("investor embedding shape: ", model.investor_dnn_embedding[:2])

    # %%
    # 8. get single tower
    dict_trained = model.state_dict()    # trained model
    trained_lst = list(dict_trained.keys())

    # company tower
    model_company = DSSM(company_feature_columns, [], task='binary', device=device)
    dict_company = model_company.state_dict()
    for key in dict_company:
        dict_company[key] = dict_trained[key]
    model_company.load_state_dict(dict_company)    # load trained model parameters of company tower
    company_feature_name = company_sparse_features + company_dense_features
    company_model_input = {name: test[name] for name in company_feature_name}
    company_model_input["company_hist"] = test_company_hist
    company_embedding = model_company.predict(company_model_input, batch_size=2000)
    print("single company embedding shape: ", company_embedding[:2])

    # investor tower
    model_investor = DSSM([], investor_feature_columns, task='binary', device=device)
    dict_investor = model_investor.state_dict()
    for key in dict_investor:
        dict_investor[key] = dict_trained[key]
    model_investor.load_state_dict(dict_investor)  # load trained model parameters of investor tower
    investor_feature_name = investor_sparse_features + investor_dense_features
    investor_model_input = {name: test[name] for name in investor_feature_name}
    investor_model_input["genres"] = test_genres_list
    investor_embedding = model_investor.predict(investor_model_input, batch_size=2000)
    print("single investor embedding shape: ", investor_embedding[:2])
