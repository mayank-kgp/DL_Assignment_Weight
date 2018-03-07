# Since windows doesn't support tensorflow for python 2, this code
# is written for python 3

from sys import argv
import numpy as np
import tensorflow as tf
from data_loader import *
# from __future__ import print_function # Uncomment it for python 2.7

# Necessary Constants
split_ratio = 0.8
batch_size = 100
epochs = 20
learning_rate = 0.01
no_of_nodes = 100

# Necessary Functions
def softmax(x):
    e_x = tf.exp(x)
    return e_x / tf.reduce_sum(e_x, 1, keep_dims = True)
def cross_entropy(ypred,y):
    return tf.reduce_mean(tf.reduce_sum(-y*tf.log(ypred),1))
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

class RNN_LSTM():
    def __init__(self, input_entry, hidden_size):
        self.unfolded_length = input_entry.shape[-1]
        self.hidden_size = hidden_size
        self.batch_size = input_entry.shape[1]
        self.input_size = input_entry.shape[-2]
        self.w_c = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_c = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.w_f = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_f = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.w_u = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_u = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.w_o = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_o = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.h_t_1 = None
        self.c_t_1 = None
        
    def forward(self, input_entry):
        if self.h_t_1 is not None:
            concated = tf.concat([self.h_t_1,input_entry], 1)
            c_t_dash = tf.tanh(tf.matmul(concated,self.w_c)+self.b_c)
            f = sigmoid(tf.matmul(concated,self.w_f)+self.b_f)
            u = sigmoid(tf.matmul(concated,self.w_u)+self.b_u)
            o = sigmoid(tf.matmul(concated,self.w_o)+self.b_o)
            c_t = f*self.c_t_1 + u*c_t_dash
        else:
            concated = input_entry
            c_t_dash = tf.tanh(tf.matmul(concated,self.w_c[-int(input_entry.shape[1]):])+self.b_c)
            f = sigmoid(tf.matmul(concated,self.w_f[-int(input_entry.shape[1]):])+self.b_f)
            u = sigmoid(tf.matmul(concated,self.w_u[-int(input_entry.shape[1]):])+self.b_u)
            o = sigmoid(tf.matmul(concated,self.w_o[-int(input_entry.shape[1]):])+self.b_o)
            c_t = u*c_t_dash
        
        h_t = o*tf.tanh(c_t)
        self.h_t_1 = h_t
        self.c_t_1 = c_t
        
    def train(self, input_entry, output_entry):
        self.h_t_1 = None
        self.c_t_1 = None
        w_final = tf.Variable(np.random.randn(self.hidden_size,output_entry.shape[1])*np.sqrt(2.0/int(self.hidden_size)))
        b_final = tf.Variable(np.random.randn(1,output_entry.shape[1])*np.sqrt(2.0/int(self.hidden_size)))
        for i in range(self.unfolded_length):
            self.forward(input_entry[:,:,i])
        softmax_layer = softmax(tf.matmul(self.h_t_1,w_final)+b_final)
        return softmax_layer

class RNN_GRU():
    def __init__(self, input_entry, hidden_size):
        self.unfolded_length = input_entry.shape[-1]
        self.hidden_size = hidden_size
        self.batch_size = input_entry.shape[1]
        self.input_size = input_entry.shape[-2]
        self.w_c = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_c = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.w_u = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_u = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.w_r = tf.Variable(np.random.randn(input_entry.shape[-2]+hidden_size,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2]+hidden_size)))
        self.b_r = tf.Variable(np.random.randn(1,hidden_size)*np.sqrt(2.0/int(input_entry.shape[-2])))
        self.h_t_1 = None
        
    def forward(self, input_entry):
        h_t = None
        if self.h_t_1 is not None:
            concated = tf.concat([self.h_t_1,input_entry], 1)
            u = sigmoid(tf.matmul(concated,self.w_u)+self.b_u)
            r = sigmoid(tf.matmul(concated,self.w_r)+self.b_r)
            concated2 = tf.concat([r*self.h_t_1,input_entry], 1)
            c_t_dash = tf.tanh(tf.matmul(concated2,self.w_c)+self.b_c)
            h_t = (1-u)*self.h_t_1 + u*c_t_dash
        else:
            concated = input_entry
            u = sigmoid(tf.matmul(concated,self.w_u[-int(input_entry.shape[1]):])+self.b_u)
            r = sigmoid(tf.matmul(concated,self.w_r[-int(input_entry.shape[1]):])+self.b_r)
            c_t_dash = tf.tanh(tf.matmul(concated,self.w_c[-int(input_entry.shape[1]):])+self.b_c)
            h_t = u*c_t_dash
        self.h_t_1 = h_t
        
    def train(self, input_entry, output_entry):
        self.h_t_1 = None
        w_final = tf.Variable(np.random.randn(self.hidden_size,output_entry.shape[1])*np.sqrt(2.0/int(self.hidden_size)))
        b_final = tf.Variable(np.random.randn(1,output_entry.shape[1])*np.sqrt(2.0/int(self.hidden_size)))
        for i in range(self.unfolded_length):
            self.forward(input_entry[:,:,i])
        softmax_layer = softmax(tf.matmul(self.h_t_1,w_final)+b_final)
        return softmax_layer

if len(argv)<4:
    print("Very less arguments Passed")
else:
    model_name = argv[1].split('=')[1].strip('\'')
    hidden_unit = argv[3].split('=')[1].strip('\'')
    if argv[2] == "--train" :
        # Data loading
        trainDataLoader = DataLoader()
        trainDataLoader.load_data()
        Xtrain, Ytrain, Xval, Yval = trainDataLoader.create_batches(split_ratio, batch_size) # train-Validation split

        hidden_size = int(hidden_unit)

        # NN Model Architecture
        tf.reset_default_graph()
        X = tf.placeholder(tf.float64, shape = [None, Xtrain.shape[2], Xtrain.shape[3]], name = 'X') #Input
        Y = tf.placeholder(tf.float64, shape = [None, Ytrain.shape[2]], name = 'Y') #Ouput

        if model_name == 'lstm':
            rnn_lstm = RNN_LSTM(Xtrain, hidden_size)
        else:
            rnn_lstm = RNN_GRU(Xtrain, hidden_size)

        softmax_layer = rnn_lstm.train(X,Y)
        cost = cross_entropy(softmax_layer, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #Optimizer to optimize the loss function
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(Y, 1)), tf.float32)) #To be used to calculate val. loss
        accuracy = tf.identity(accuracy, name="Accuracy")

        #Training NN
        tf.set_random_seed(123)

        # Best validation accuracy seen so far.
        best_validation_accuracy = 0.0

        # Iteration-number for last improvement to validation accuracy.
        last_improvement = 0

        # Stop optimization if no improvement found in this many iterations.
        patience = 20

        # Start session
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            for e in range(epochs):
                avg_cost = 0
                for i in range(Xtrain.shape[0]):
                    feed_dict = {X: Xtrain[i], Y: Ytrain[i]}
                    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                    avg_cost += c / Xtrain.shape[0]
                feed_dict = {X: Xval, Y: Yval}
                acc = sess.run(accuracy, feed_dict=feed_dict) #Validation Accuracy
                print('Epoch : ' + str(e+1) + '/' + str(epochs) + '    Loss: ' + str(avg_cost) + '   Validation Accuracy : ' + str(acc*100) + '%')
                if acc > best_validation_accuracy:
                    last_improvement = e
                    best_validation_accuracy = acc
                    saver.save(sess, 'weights/'+model_name+'/'+hidden_unit+'/NN') 
                if e - last_improvement > patience:
                    print("Early stopping ...")
                    break
            print('\n')
            print('Best Validation Accuracy Achieved : ' + str(best_validation_accuracy*100) + '%')

    if argv[2] == "--test":
        # Testing Block

        # Loading Data
        testDataLoader = DataLoader()
        testDataLoader.load_data('test')
        Xtest, Ytest = testDataLoader.images, testDataLoader.labels

        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('weights/'+model_name+'/'+hidden_unit+'/NN.meta')
            saver.restore(sess, tf.train.latest_checkpoint('weights/'+model_name+'/'+hidden_unit))
            feed_dict = {'X:0': Xtest, 'Y:0': Ytest}
            final_acc = sess.run('Accuracy:0', feed_dict=feed_dict)
            print('Final Test Accuracy : ' + str(final_acc*100) + '%')