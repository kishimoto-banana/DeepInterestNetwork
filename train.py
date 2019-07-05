import pickle
import numpy as np
import tensorflow as tf
from model import deep_interest_network

FILE_PATH = 'dataset_small.pkl'
SEQ_MAX_LEN = 100
EPOCHS = 2


def build_data(filepath, seq_max_len):

    with open(filepath, 'rb') as f:
        train_set = pickle.load(f)
        _ = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    X = [
        np.array([sample[0] for sample in train_set]),
        np.array([sample[2] for sample in train_set]),
        np.array([cate_list[sample[2]] for sample in train_set])
    ]
    behavior_item_feature = tf.keras.preprocessing.sequence.pad_sequences(
        [sample[1] for sample in train_set],
        padding='post',
        truncating='post',
        maxlen=seq_max_len)
    behavior_category_feature = tf.keras.preprocessing.sequence.pad_sequences(
        [cate_list[sample[1]] for sample in train_set],
        padding='post',
        truncating='post',
        maxlen=seq_max_len)
    X.append(behavior_item_feature)
    X.append(behavior_category_feature)
    y = np.array([sample[3] for sample in train_set])

    # 情報定義
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
        'dim': cate_count,
        'is_behavior': True
    }]

    return X, y, features_info


X, y, features_info = build_data(FILE_PATH, SEQ_MAX_LEN)

model = deep_interest_network(features_info)
model.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=EPOCHS, validation_split=0.2)
