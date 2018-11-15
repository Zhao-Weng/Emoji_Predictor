import numpy as np
np.random.seed(0)
from emo_utils import *
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

np.random.seed(1)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    emb_dim = word_to_vec_map["apple"].shape[0] 
    vocab_len = len(word_to_index) + 1  # fit keras embedding

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


# build lstm neural network
def Emojify_V2(input_shape, word_to_vec_map, word_to_index, dropoutRate):

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = embedding_layer(sentence_indices)


    X = LSTM(64, return_sequences=True)(embeddings)
    X = Dropout(0.4)(X)    # 0.4 probability to drop off
    X = LSTM(64, return_sequences=False)(X)
    # X = Dropout(0.7)(X)
    X = Dense(5, activation=None)(X)    # 5 output choices
    X = Activation('softmax')(X)
    model = Model(inputs=[sentence_indices], outputs=X)
    return model

# convert sentences to indices
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] 
    X_index = np.zeros((m, max_len))
    for i in range(m):  
        sentence_words = [word.lower().replace('\t', '') for word in X[i].split(' ') if word.replace('\t', '') != '']
        j = 0

        for w in sentence_words:
            X_index[i, j] = word_to_index[w]
            j += 1


    return X_index

X_test, Y_test = read_csv('data/tesss.csv')
X_training, Y_train = read_csv('data/train_emoji.csv')

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
maxLen = len(max(X_training, key=len).split())
Y_oneHot_test = convert_to_one_hot(Y_test, C = 5)
Y_oneHotTrain = convert_to_one_hot(Y_train, C = 5)


dropoutRates = [0.5]
epochs = [200]
for dropoutRate in dropoutRates:
    for epoch in epochs:
        model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index, dropoutRate)
        
        # use adam algorithm, categorical cross entropy as loss function
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        X_trainingIndices = sentences_to_indices(X_training, word_to_index, maxLen)
        Y_trainingOneHot = convert_to_one_hot(Y_train, C = 5)
        # model.summary()
        model.fit(X_trainingIndices, Y_trainingOneHot, epochs = epoch, batch_size = 32, shuffle=True)
        X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
        Y_test_oh = convert_to_one_hot(Y_test, C = 5)
        loss, acc = model.evaluate(X_test_indices, Y_test_oh)
        print("\n\n testing accuracy = ", acc)


X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
numOfOutputs = 5
y_test_oh = np.eye(numOfOutputs)[Y_test.reshape(-1)]
pred = model.predict(X_test_indices)
# mislablled emojies
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])       # find the predicted label
    if(num != Y_test[i]):          # find mislabelled sentences
        print('correct emoji:'+ label_to_emoji(Y_test[i]) + ' predicted emoji: '+ X_test[i] + label_to_emoji(num).strip())
# my own sentence for testing
x_test = np.array(['please give me another chance'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +'  '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))