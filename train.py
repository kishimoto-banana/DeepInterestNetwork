import pickle
import numpy as np
import tensorflow as tf
from model import deep_interest_network

FILE_PATH = 'dataset_small.pkl'
LOG_PATH = './logs'
SEQ_MAX_LEN = 10
EPOCHS = 100


def build_data(file_path, seq_max_len):
    """
    学習データと特徴量の情報を定義した辞書を作成する
    :param file_path: データセットのパス
    :param seq_max_len: 行動データの最大系列長
    :return: 学習データ、正解ラベル、特徴量情報の辞書
    """

    # データの読み込み
    with open(file_path, 'rb') as f:
        # 各要素が各サンプルのリスト
        # train_data[i] -> (ユーザーID、購入商品のリスト、商品ID、正解ラベル)
        train_data = pickle.load(f)
        _ = pickle.load(f)
        item_to_category = pickle.load(f)
        user_count, item_count, category_count = pickle.load(f)

    # 学習データをモデルに入力できるように整形
    X = [
        np.array([sample[0] for sample in train_data]),
        np.array([sample[2] for sample in train_data]),
        np.array([item_to_category[sample[2]] for sample in train_data])
    ]
    behavior_item_feature = tf.keras.preprocessing.sequence.pad_sequences(
        [sample[1] for sample in train_data],
        padding='post',
        truncating='post',
        maxlen=seq_max_len)
    behavior_category_feature = tf.keras.preprocessing.sequence.pad_sequences(
        [item_to_category[sample[1]] for sample in train_data],
        padding='post',
        truncating='post',
        maxlen=seq_max_len)
    X.append(behavior_item_feature)
    X.append(behavior_category_feature)
    y = np.array([sample[3] for sample in train_data])

    # 特徴量情報の辞書定義
    features_info = [{
        'name': 'user_id',
        'dim': user_count,
        'is_behavior': False
    }, {
        'name': 'item_id',
        'dim': item_count,
        'is_behavior': True
    }, {
        'name': 'category_id',
        'dim': category_count,
        'is_behavior': True
    }]

    return X, y, features_info


X, y, features_info = build_data(FILE_PATH, SEQ_MAX_LEN)

model = deep_interest_network(features_info, SEQ_MAX_LEN)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.9)
model.compile(
    optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)
model.fit(
    X,
    y,
    epochs=EPOCHS,
    validation_split=0.2,
    batch_size=512,
    callbacks=[tb_callback])
