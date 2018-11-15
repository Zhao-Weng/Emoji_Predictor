import numpy as np
import matplotlib.pyplot as plt
from emo_utils import *
import emoji

def model(X, Y, word_to_vector_map, learningRate=0.005, numIterations=500):

    np.random.seed(1)
    n_h = 50  
    numOfExamples = Y.shape[0]  # number of examples
    numOfOutputs = 5  # 5 output classes


    W = np.random.randn(numOfOutputs, n_h) / np.sqrt(n_h)
    b = np.zeros((numOfOutputs,))

    output_oneHot = convert_to_one_hot(Y, C=numOfOutputs)

    for iteration in range(numIterations): 
        for i in range(numOfExamples):  # for all traning samples

            avg = sentence_to_avg(X[i], word_to_vector_map)
            z = np.dot(W, avg) + b
            a = softmax(z)

            cost = -np.squeeze(np.sum(output_oneHot[i] * np.log(a)))
            dz = a - output_oneHot[i]
            dW = np.dot(dz.reshape(numOfOutputs, 1), avg.reshape(1, n_h))
            db = dz

            b = b - learningRate * db
            W = W - learningRate * dW
            
            # update W, b parameters using gradient descent

        if iteration % 80 == 0:   # show results for every 80 steps
            # print("epoch number: " + str(iteration) + "   cost: " + str(cost))
            pred = predict(X, Y, W, b, word_to_vector_map)
    return pred, W, b

def sentence_to_avg(sentence, word_to_vector_map):
    words = [word.lower() for word in sentence.split(' ') if word != '']
    avg = np.zeros(word_to_vector_map[words[0]].shape)

    for w in words: # loop over words to calculate average of word embeddings
        avg += word_to_vector_map[w]
    avg = avg /float(len(words))
    return avg


X_training, Y_training = read_csv('data/train_emoji.csv')
maxLen = len(max(X_training, key=len).split())
X_testing, Y_testing = read_csv('data/tesss.csv')



Y_oh_test = convert_to_one_hot(Y_testing, C = 5)
Y_oh_train = convert_to_one_hot(Y_training, C = 5)
word_to_index, index_to_word, word_to_vector_map = read_glove_vecs('data/glove.6B.50d.txt')


iteartions = [500, 1000]
learningRates = [0.001, 0.005, 0.01]
for iteration in iteartions:
    for learningRate in learningRates:
        pred, W, b = model(X_training, Y_training, word_to_vector_map, learningRate, iteration)
        print("number of iteration ={0}, learning rate = {1}".format(iteration, learningRate))
        print('training :')
        pred_test = predict(X_testing, Y_testing, W, b, word_to_vector_map)
        print("testing :")
        pred_train = predict(X_training, Y_training, W, b, word_to_vector_map)
