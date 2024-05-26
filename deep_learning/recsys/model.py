def DSSM(user_feature_columns, item_feature_columns, dnn_units=[64, 32], 
        temp=10, task='binary'):
    # 构建所有特征的Input层和Embedding层
    feature_encode = FeatureEncoder(user_feature_columns + item_feature_columns)
    feature_input_layers_list = list(feature_encode.feature_input_layer_dict.values())

    # 特征处理
    user_dnn_input, item_dnn_input = process_feature(user_feature_columns, item_feature_columns, feature_encode)

    # 构建模型的核心层
    if len(user_dnn_input) >= 2:
        user_dnn_input = Concatenate(axis=1)(user_dnn_input)
    else:
        user_dnn_input = user_dnn_input[0]
    if len(item_dnn_input) >= 2:
        item_dnn_input = Concatenate(axis=1)(item_dnn_input)
    else:
        item_dnn_input = item_dnn_input[0]
    user_dnn_input = Flatten()(user_dnn_input) 
    item_dnn_input = Flatten()(item_dnn_input)
    user_dnn_out = DNN(dnn_units)(user_dnn_input)
    item_dnn_out = DNN(dnn_units)(item_dnn_input)


    # 计算相似度
    scores = CosinSimilarity(temp)([user_dnn_out, item_dnn_out]) # (B,1)
    # 确定拟合目标
    output = PredictLayer()(scores)
    # 根据输入输出构建模型
    model = Model(feature_input_layers_list, output)
    return model 