import math
from function import *
from model_class import *

print(tf.__version__)
print(tf.test.is_gpu_available())


vocab = from_txt_get_vocab(dir1='vocab.txt')

id2word = {}
id2word[0] = ''
for word, id in vocab.items():
    id2word[id] = word

sentences_1, labels_1 = get_data(dir1='plot_1.txt', word_to_index=vocab, encode='ISO-8859-1')
sentences_0, labels_0 = get_data(dir1='quote_0.txt', word_to_index=vocab, encode='ISO-8859-1')

lens = len(sentences_1)
k = 10
batch_num = math.floor(lens / k)
dataset = {}
labels = {}
for i in range(k):
    dataset[i] = sentences_1[i * batch_num: (i + 1) * batch_num] + sentences_0[i * batch_num: (i + 1) * batch_num]
    labels[i] = labels_1[i * batch_num: (i + 1) * batch_num] + labels_0[i * batch_num: (i + 1) * batch_num]

vocab_size = len(vocab)
embedding_dim = 300
max_len = 64
batchsz = 128
epoch = 30
learning_rate = CustomSchedule(embedding_dim, warmup_steps=4000)

dataset_train = {}
dataset_test = {}
for _iter in range(k):
    x_train = []
    y_train = []
    for i in range(k):
        if i == _iter:
            x_test = dataset[i]
            y_test = labels[i]
        else:
            x_train += dataset[i]
            y_train += labels[i]
    dataset_train[_iter] = (x_train, y_train)
    dataset_test[_iter] = (x_test, y_test)


accuracy = {}
print(batch_num)
print(learning_rate)

for i in range(k):
    (x_train, y_train) = dataset_train[i]
    (x_test, y_test) = dataset_test[i]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)

    x_train, x_test, x_test = change_train_data(x_train, x_test, x_test, max_len)

    weight_path = 'saved_weights/weights_' + str(i)

    """早停和动态衰减学习率有些冲突 -- 别人的实验证明没有冲突，即是patience相同"""
    checkpoint_filepath = 'checkpoint/checkpoint'
    filepath = checkpoint_filepath + '_' + str(i)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + '_' + str(i),
                                                                   save_best_only=True, save_weights_only=True,
                                                                   monitor='val_accuracy', mode='max', verbose=1)

    model = position_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),  # 静态学习率没有动态学习率更适应模型
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=15, verbose=1, validation_data=(x_test, y_test), batch_size=128,
                        callbacks=[model_checkpoint_callback])
    # , callbacks=[model_checkpoint_callback, model_early_stop, model_tensorboard])

    loss, accuracy[i] = model.evaluate(x_test, y_test, verbose=2)


for i in range(k):
    (x_train, y_train) = dataset_train[i]
    (x_test, y_test) = dataset_test[i]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)

    x_train, x_test, x_test = change_train_data(x_train, x_test, x_test, max_len)

    filepath = checkpoint_filepath + '_' + str(i)
    model.load_weights(filepath)

    y_predict = model.predict(x_test)
    y_predict = tf.nn.sigmoid(y_predict).numpy()
    y_predict = np.squeeze(y_predict)
    y_predict = tf.stack([1 if value > 0.5 else 0 for value in y_predict]).numpy()

    samples = (x_test, y_test, y_predict)
    false_sample = []
    counter = 0
    for _iter in range(y_test.shape[0]):
        p = y_predict[_iter]
        t = y_test[_iter]
        if p != t:
            counter += 1
            false_sample.append((x_test[0][_iter], t, p))

    with open('error_sample_' + str(i) + '.txt', 'w', encoding='ISO-8859-1') as f:
        for sent_tuple in false_sample:
            sent_id, t_label, p_label = sent_tuple
            sent_id = to_word(sent_id, id2word)
            sent_id = ' '.join(word for word in sent_id)
            f.writelines(sent_id + '\t' + str(t_label) + '\t' + str(p_label) + '\n')
        f.close()

    loss, accuracy[i] = model.evaluate(x_test, y_test, verbose=2)

len_accuracy = len(accuracy)
sum = 0
for i in range(len_accuracy):
    sum += accuracy[i] / len_accuracy

print(sum)
