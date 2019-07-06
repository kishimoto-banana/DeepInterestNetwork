from collections import OrderedDict
import tensorflow as tf


def build_input(features_info, seq_max_len):
    """
    各カテゴリ変数と行動データのInputレイヤを構築する
    :param features_info: 特徴量情報の辞書
    :param seq_max_len: 行動データの最大系列長
    :return: カテゴリ変数のInputレイヤの辞書、行動データのInputレイヤの辞書
    """
    categorical_input_dict = OrderedDict()
    behavior_input_dict = OrderedDict()
    for feature in features_info:
        categorical_input_dict[feature['name']] = tf.keras.layers.Input(
            shape=(1, ), name=feature['name'])
        if feature['is_behavior']:
            behavior_input_dict[feature['name']] = tf.keras.layers.Input(
                shape=(seq_max_len, ), name=f"behavior_{feature['name']}")

    return categorical_input_dict, behavior_input_dict


def build_embedding_layers(features_info, embedding_dim=8):
    """
    各カテゴリ変数のEmbeddingレイヤを構築する
    :param features_info: 特徴量情報の辞書
    :param embedding_dim: Embeddingの次元数
    :return: Embeddingレイヤの辞書
    """
    embedding_dict = OrderedDict()
    for feature in features_info:
        if feature['is_behavior']:
            embedding_dict[feature['name']] = tf.keras.layers.Embedding(
                feature['dim'],
                embedding_dim,
                name=f"embedding_{feature['name']}",
                mask_zero=True)
        else:
            embedding_dict[feature['name']] = tf.keras.layers.Embedding(
                feature['dim'],
                embedding_dim,
                name=f"embedding_{feature['name']}")

    return embedding_dict


def embedding(categorical_input_dict, behavior_input_dict, embedding_dict,
              features_info):
    """
    Embeggingを行う
    :param categorical_input_dict: カテゴリ変数のInputレイヤの辞書
    :param behavior_input_dict: 行動データのInputレイヤの辞書
    :param embedding_dict: Embeddingレイヤの辞書
    :param features_info: 特徴量情報の辞書
    :return: カテゴリ変数のEmbedding、候補データのEmbedding、行動データのEmbedding
    """
    categorical_embeddings = []
    candidate_embeddings = []
    behavior_embeddings = []

    # 特徴量ごとにEmbedding
    for feature in features_info:
        embedding = embedding_dict[feature['name']](
            categorical_input_dict[feature['name']])
        categorical_embeddings.append(embedding)

        if feature['is_behavior']:
            embedding = embedding_dict[feature['name']](
                categorical_input_dict[feature['name']])
            candidate_embeddings.append(embedding)

            embedding = embedding_dict[feature['name']](
                behavior_input_dict[feature['name']])
            behavior_embeddings.append(embedding)

    # 特徴量を結合
    categorical_embeddings = tf.keras.layers.Concatenate()(
        categorical_embeddings)
    candidate_embeddings = tf.keras.layers.Concatenate()(candidate_embeddings)
    behavior_embeddings = tf.keras.layers.Concatenate()(behavior_embeddings)

    return categorical_embeddings, candidate_embeddings, behavior_embeddings


def behavior_pooling(behavior_embeddings, behavior_input):
    """
    行動データのSum Pooling（論文におけるBase Model用）
    :param behavior_embeddings: 行動データのEmbedding
    :param behavior_input: 行動データのInputレイヤ
    :return: pooling後の出力
    """

    # パディング用マスキング
    mask = tf.equal(behavior_input, 0)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, tf.shape(behavior_embeddings)[2]])
    padding = tf.zeros_like(behavior_embeddings)
    masked_behavior_embeddings = tf.where(mask, behavior_embeddings, padding)

    # sum pooling
    pooled_output = tf.reduce_sum(masked_behavior_embeddings, 1)
    pooled_output = tf.expand_dims(pooled_output, 1)

    return pooled_output


def behavior_attention(query, keys, behavior_input, padding_value=-2**32 + 1):
    """
    # 候補データ(query)と行動データ(Keys)のAttention
    :param query:
    :param keys:
    :param behavior_input: 行動データのInputレイヤ
    :param padding_value: padding部分のsoftmax関数に入力する値
    :return: Attentionの出力
    """

    # 時系列分repeat
    # Shape: (B, T, H)
    # B: バッチサイズ、T: 系列長、H: 次元数
    query = tf.keras.backend.repeat_elements(query, keys.get_shape()[1], 1)

    # フィードフォワードネットワークへの入力
    attention_input = tf.concat([query, keys, query - keys, query * keys],
                                axis=-1)

    # フィードフォワードネットワークでAttention weightを求める
    attention_weight = tf.keras.layers.Dense(
        80, activation='relu')(attention_input)
    attention_weight = tf.keras.layers.Dense(
        40, activation='relu')(attention_weight)
    attention_weight = tf.keras.layers.Dense(
        1, activation=None)(attention_weight)

    # Shape: (B, T, 1) => (B, 1, T)
    attention_weight = tf.transpose(attention_weight, (0, 2, 1))

    # パディング用マスキング
    mask = tf.equal(behavior_input, 0)  # Shape: (B, T)
    mask = tf.expand_dims(mask, 1)  # Shape: (B, 1, T)
    padding = tf.ones_like(attention_weight) * padding_value
    attention_weight = tf.where(mask, attention_weight, padding)

    # caled Dot-Product Attention的なやつ
    attention_weight = attention_weight / (keys.get_shape()[-1]**0.5)

    # softmax関数で正規化
    # 論文ではsoftmax関数に通していなかったが、実装では通していたのでそれに合わせる
    attention_weight = tf.nn.softmax(attention_weight)

    # Attention weightに従って過去の行動のEmbeddingのsum pooling
    attention_output = tf.matmul(attention_weight, keys)  # Shape: (B, 1, H)

    return attention_output


def deep_interest_network(features_info, seq_max_len=100, dropout_rate=0.5):
    """
    DeepInterestNetworkモデルを構築する
    :param features_info: 特徴量情報の辞書
    :param seq_max_len: 行動データの最大系列長
    :param dropout_rate: ドロップアウト率
    :return: DeepInterestNetworkモデル
    """

    # Inputレイヤ作成
    categorical_input_dict, behavior_input_dict = build_input(
        features_info, seq_max_len)

    # Embeddingレイヤ作成
    embedding_dict = build_embedding_layers(features_info)

    # Embedding
    categorical_embeddings, candidate_embeddings, behavior_embeddings = embedding(
        categorical_input_dict, behavior_input_dict, embedding_dict,
        features_info)

    # 行動データと候補データのAttention
    behavior_attention_embeddings = behavior_attention(
        candidate_embeddings, behavior_embeddings,
        list(behavior_input_dict.values())[0])

    # 全Embeddingの結合
    embeddings = tf.keras.layers.Concatenate()(
        [categorical_embeddings, behavior_attention_embeddings])
    embeddings = tf.keras.layers.Flatten()(embeddings)
    embeddings = tf.keras.layers.Dropout(dropout_rate)(embeddings)

    # 全結合層
    output = tf.keras.layers.Dense(80, activation='relu')(embeddings)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(40, activation='relu')(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    inputs = [input for input in categorical_input_dict.values()
              ] + [input for input in behavior_input_dict.values()]
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
