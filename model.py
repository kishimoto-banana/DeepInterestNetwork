from collections import OrderedDict
import tensorflow as tf


def build_input(features_info, seq_max_len):
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
    categorical_embeddings = []
    candidate_embeddings = []
    behavior_embeddings = []
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

    categorical_embeddings = tf.keras.layers.Concatenate()(
        categorical_embeddings)
    candidate_embeddings = tf.keras.layers.Concatenate()(candidate_embeddings)
    behavior_embeddings = tf.keras.layers.Concatenate()(behavior_embeddings)

    return categorical_embeddings, candidate_embeddings, behavior_embeddings


def behavior_pooling(behavior_embeddings, behavior_input):

    # パディング用マスキング
    mask = tf.equal(behavior_input, 0)
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    mask = tf.tile(mask, [1, 1, tf.shape(behavior_embeddings)[2]])  # [B, T, H]
    padding = tf.zeros_like(behavior_embeddings)
    masked_behavior_embeddings = tf.where(mask, behavior_embeddings, padding)

    # sum pooling
    pooled_output = tf.reduce_sum(masked_behavior_embeddings, 1)

    return pooled_output


def behavior_attention(query, keys, behavior_input, padding_value=-2**32 + 1):

    # 時系列分repeat
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

    # (B, T, 1) => (B, 1, T)
    attention_weight = tf.transpose(attention_weight, (0, 2, 1))

    # パディング用マスキング
    mask = tf.equal(behavior_input, 0)
    mask = tf.expand_dims(mask, 1)
    padding = tf.ones_like(attention_weight) * padding_value
    attention_weight = tf.where(mask, attention_weight, padding)

    # softmax関数で正規化
    # 論文ではsoftmax関数に通していなかったが、実装では通していたのでそれに合わせる
    attention_weight = tf.nn.softmax(attention_weight)

    # Attention weightに従って過去の行動のEmbeddingのsum pooling
    attention_output = tf.matmul(attention_weight, keys)

    return attention_output


def deep_interest_network(features_info, seq_max_len=100):
    # --- こっからモデル --- #
    categorical_input_dict, behavior_input_dict = build_input(
        features_info, seq_max_len)

    # Embeddingレイヤ
    embedding_dict = build_embedding_layers(features_info)

    categorical_embeddings, target_embeddings, source_embeddings = embedding(
        categorical_input_dict, behavior_input_dict, embedding_dict,
        features_info)

    behavior_attention_embeddings = behavior_attention(
        target_embeddings, source_embeddings,
        list(behavior_input_dict.values())[0])

    embeddings = tf.keras.layers.Concatenate()(
        [categorical_embeddings, behavior_attention_embeddings])
    embeddings = tf.keras.layers.Flatten()(embeddings)

    output = tf.keras.layers.Dense(80, activation='relu')(embeddings)
    output = tf.keras.layers.Dense(40, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    inputs = [input for input in categorical_input_dict.values()
              ] + [input for input in behavior_input_dict.values()]
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
