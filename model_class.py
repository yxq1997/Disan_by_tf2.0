import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, concatenate, Flatten
from tensorflow.keras import optimizers, losses, regularizers, Model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):  # 设置epoch从0到4000时lr的变化函数, warmup_steps越大，最大的lr越小
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class position_attention(Model):
    def __init__(self, vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None, attn_mask=False):
        super(position_attention, self).__init__()
        self.attn_mask = 0
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_states = 64
        self.l2 = 0.1
        self.pad_data = 0
        self.word_embedding = 0

        if embedding_matrix is None:
            self.word_embed = Embedding(vocab_size + 1, embedding_dim, input_length=max_len)
        else:
            self.word_embed = Embedding(vocab_size + 1, embedding_dim, input_length=max_len,
                                        weights=[embedding_matrix])
        self.position_embed = Embedding(input_dim=max_len, output_dim=embedding_dim, input_length=max_len)
        self.drop_embedding = Dropout(rate=0.1)
        self.d1 = Dense(embedding_dim)
        self.drop = Dropout(rate=0.3)

        self.d2 = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3 = Dense(embedding_dim, use_bias=False)
        self.bias = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        # 混合attention和输入,后接dropout
        self.d6 = Dense(embedding_dim)
        self.d7 = Dense(embedding_dim)

        self.d1_ex = Dense(embedding_dim)
        self.drop_ex = Dropout(rate=0.3)
        self.d2_ex = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3_ex = Dense(embedding_dim, use_bias=False)
        self.bias_ex = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias_ex = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.d6_ex = Dense(embedding_dim)
        self.d7_ex = Dense(embedding_dim)

        self.d1_no = Dense(embedding_dim)
        self.drop_no = Dropout(rate=0.3)
        self.d2_no = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3_no = Dense(embedding_dim, use_bias=False)
        self.bias_no = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias_no = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.d6_no = Dense(embedding_dim)
        self.d7_no = Dense(embedding_dim)

        # 不同direction下模型输出的out融合
        self.d8 = Dense(self.hidden_states)
        self.d9 = Dense(self.hidden_states)
        self.d10 = Dense(self.hidden_states)
        self.d11 = Dense(embedding_dim)

        # 模型输出之后的全连接层
        self.d4 = Dense(embedding_dim)  # 后接dropout和tf.nn.relu
        self.flat = Flatten()
        self.d5 = Dense(output_dim, use_bias=False, kernel_regularizer=regularizers.l2(self.l2))  # 返回的是没有经过sigmoid的数

        # HAPN的模块
        self.dw1 = Dense(embedding_dim)
        self.dw2 = Dense(embedding_dim)
        self.du1 = Dense(embedding_dim)
        self.du2 = Dense(embedding_dim)

        self.dw1_ex = Dense(embedding_dim)
        self.dw2_ex = Dense(embedding_dim)
        self.du1_ex = Dense(embedding_dim)
        self.du2_ex = Dense(embedding_dim)

    def call(self, inputs):
        word_embedding = self.ProduceEmbedding(inputs)

        f_result, f_score = self.direction_position_embedding(inputs, word_embedding, direction='forward')
        b_result, b_score = self.direction_position_embedding(inputs, word_embedding, direction='backward')
        no_direction_result = None
        attn_result = self.output_gate(f_result, b_result, no_direction_result)

        output = self.FCLayer(attn_result)
        return output

    def ProduceEmbedding(self, inputs):
        word_data, position_data, sentence_length = inputs  # 处理以list形式输入的训练样本

        word_embedding = self.word_embed(word_data)
        position_embedding = self.position_embed(position_data)
        word_embedding = word_embedding + position_embedding  # 加入position信息
        return word_embedding

    def direction_position_embedding(self, inputs, word_embedding, direction=None):
        word_data, position_data, sentence_length = inputs  # 处理以list形式输入的训练样本
        x_train, position_data, sentence_length = inputs  # 加入sentence_length以修正填充式的错误 ----> （改善了mask函数的缺点）
        bs = tf.shape(word_data)[0]
        sll = tf.shape(word_data)[1]

        sl = tf.range(self.max_len)
        word_col, word_row = tf.meshgrid(sl, sl)

        if direction == "forward":
            direct_mask = tf.greater(word_row, word_col)
            dense_d1 = self.d1
            dense_d2 = self.d2
            dense_d3 = self.d3
            dense_drop = self.drop
            dense_bias = self.bias

        elif direction == 'backward':
            direct_mask = tf.greater(word_col, word_row)
            dense_d1 = self.d1_ex
            dense_d2 = self.d2_ex
            dense_d3 = self.d3_ex
            dense_drop = self.drop_ex
            dense_bias = self.bias_ex

        else:
            direct_mask = tf.cast(tf.linalg.tensor_diag(- tf.ones([sll], tf.int32)) + 1, tf.bool)
            dense_d1 = self.d1_no
            dense_d2 = self.d2_no
            dense_d3 = self.d3_no
            dense_drop = self.drop_no
            dense_bias = self.bias_no

        pad_data = tf.cast(tf.cast(x_train, bool), tf.float32)
        self.pad_data = pad_data
        direct_mask_1_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])
        pad_data_tile = sentence_length

        direct_mask_1_tile = tf.cast(direct_mask_1_tile, bool)
        pad_data_tile = tf.cast(pad_data_tile, bool)
        attn_mask = tf.logical_and(direct_mask_1_tile, pad_data_tile)
        self.attn_mask = attn_mask

        word_embedding1 = dense_d1(word_embedding)
        word_embedding1 = self.selu(word_embedding1)
        word_embedding1 = dense_drop(word_embedding1)
        word_embedding_tile = tf.tile(tf.expand_dims(word_embedding1, 1), [1, sll, 1, 1])
        d2 = dense_d2(word_embedding1)
        d2 = dense_drop(d2)
        d2 = tf.expand_dims(d2, axis=1)
        d3 = dense_d3(word_embedding1)
        d3 = dense_drop(d3)
        d3 = tf.expand_dims(d3, axis=2)
        self_attention_data = d2 + d3 + dense_bias
        logits = self.fx_q(self_attention_data)

        logits_masked = self.attention_mask(logits, attn_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)
        attn_score = self.position_mask(attn_score, attn_mask)
        attn_result_no_reduce = attn_score * word_embedding_tile
        attn_result = tf.reduce_sum(attn_result_no_reduce, 2)

        attn_result = self.fusion_gate(word_embedding1, attn_result, pad_data, direction)
        self.word_embedding = word_embedding

        return attn_result, attn_score

    def FCLayer(self, attn_result):
        attn_result = self.d4(attn_result)
        attn_result = self.drop(attn_result)
        attn_result = tf.nn.relu(attn_result)
        attn_result = self.flat(attn_result)
        out = self.d5(attn_result)
        return out

    def fx_q(self, val, scale=5.):
        return scale * tf.nn.tanh(1. / scale * val)

    def fusion_gate(self, word_embedding, attn_result, pad_data, direction=None):
        if direction == "forward":
            dense_fusion_bias = self.fusion_bias
            dense_d6 = self.d6
            dense_d7 = self.d7
            dense_drop = self.drop
        elif direction == 'backward':
            dense_fusion_bias = self.fusion_bias_ex
            dense_d6 = self.d6_ex
            dense_d7 = self.d7_ex
            dense_drop = self.drop_ex
        else:
            dense_fusion_bias = self.fusion_bias_no
            dense_d6 = self.d6_no
            dense_d7 = self.d7_no
            dense_drop = self.drop_no
        d6 = dense_d6(word_embedding)
        d6 = dense_drop(d6)
        d7 = dense_d7(attn_result)
        d7 = dense_drop(d7)
        fusion_weight = tf.nn.sigmoid(d6 + d7 + dense_fusion_bias)
        out = fusion_weight * word_embedding + (1 - fusion_weight) * attn_result
        output = self.position_mask(out, pad_data)  # 进行填充遮挡
        return output

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

    def position_mask(self, embedding, mask):
        mask = tf.expand_dims(mask, -1)
        # mask = mask[tf.newaxis, ...]
        mask = tf.multiply(embedding, tf.cast(mask, tf.float32))
        return mask

    def attention_mask(self, val, attn_mask):
        attn_mask = tf.expand_dims(attn_mask, -1)
        attn_mask = (1 - tf.cast(attn_mask, tf.float32)) * (-1e30)
        val = tf.add(val, attn_mask)
        return val

    def output_gate(self, out1, out2, out3):
        pad_data = self.pad_data
        out1 = tf.expand_dims(self.attention_mask(out1, pad_data), axis=1)
        out2 = tf.expand_dims(self.attention_mask(out2, pad_data), axis=1)
        word_embedding = tf.expand_dims(self.attention_mask(self.word_embedding, pad_data), axis=1)
        if out3 is None:
            fixed_embedding = concatenate([out1, out2, word_embedding], axis=1)
        else:
            out3 = tf.expand_dims(self.attention_mask(out3, pad_data), axis=1)
            fixed_embedding = concatenate([out1, out2, out3, word_embedding], axis=1)
        weights = tf.nn.softmax(fixed_embedding, axis=1)
        pad_data = tf.expand_dims(pad_data, axis=1)
        weights = self.position_mask(weights, pad_data)
        return tf.reduce_sum(weights * fixed_embedding, axis=1)



