import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def ReLu(x):
    return np.where(x<0, 0, x)

def gradReLu(x):
    return np.where(x<=0, 0, 1)

def softmax(x):
    axis = 1
    ax_sum = np.expand_dims(np.sum(np.exp(x), axis = axis), axis)
    return np.exp(x)/ax_sum


def compute(X, W, b):
    return np.matmul(X, W) + b

def averageCE(target, prediction):
    total = (np.multiply(target, np.log(prediction))).sum()
    avg = -1*total /target.shape[0]
    return avg

def gradCE(target, prediction):
    total = np.sum(target * 1/prediction)
    gradCE = total * (-1/len(target))
    return gradCE

def total_gradCE_mult_der_softmax(target, prediction):
    return np.subtract(prediction, target)

def gradSoftmax(s):
    s = softmax(s).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, np.transpose(s))


class myNN():
    def __init__(self, num_hidden):
        super(myNN, self).__init__()
        
        n_x = 784 # size of input layer`
        self.k = num_hidden
        n_y = 10 # size of output layer
        
        #initialize parameters
        self.W1 = np.random.randn(n_x, self.k) * (2 / (n_x + self.k))**(1/2.0) #Wh
        self.b1 = np.zeros(shape=(1,self.k))
        self.W2 = np.random.randn(self.k,n_y) * (2 / (self.k + n_y))**(1/2.0) #Wo
        self.b2 = np.zeros(shape=(1,n_y))
        
        #for back propagation
        self.V1_w = np.full((n_x, self.k), 1e-5)
        self.V2_w = np.full((self.k, n_y), 1e-5)
        self.V1_b = np.full((1, self.k), 1e-5)
        self.V2_b = np.full((1, n_y), 1e-5)
        self.gamma = 0.9
        self.alpha = 0.1
        
    def forward(self, img):
        self.X0 = img=
        self.S1 = compute(self.X0, self.W1, self.b1)
        self.X1 = ReLu(self.S1)    
        self.S2 = compute(self.X1, self.W2, self.b2)
        self.X2 = softmax(self.S2)
        return self.X2
    
    def backward(self, target):
        N = target.shape[0]
        delta2 = np.subtract(softmax(self.S2), target)   
        delta1 = np.multiply(np.matmul(delta2, self.W2.T), gradReLu(self.S1))
        
        d_W2 = np.matmul(self.X1.T,delta2)/N #k * 10
        d_b2 = delta2.sum(axis = 0)/N #1 * 10
        d_b2 = np.reshape(d_b2, (1, 10))
        d_W1 = np.matmul(self.X0.T,delta1)/N #784 * k
        d_b1 = delta1.sum(axis = 0)/N #1 * k
        d_b1 = np.reshape(d_b1, (1, self.k))        
        
        self.V2_w = self.gamma * self.V2_w + self.alpha * d_W2
        self.V2_b = self.gamma * self.V2_b + self.alpha * d_b2
        
        self.V1_w = self.gamma * self.V1_w + self.alpha * d_W1
        self.V1_b = self.gamma * self.V1_b + self.alpha * d_b1
        
        
        return 
    
    def update(self):
        self.W1 = self.W1 - self.V1_w
        self.b1 = self.b1 - self.V1_b
        self.W2 = self.W2 - self.V2_w
        self.b2 = self.b2 - self.V2_b
        return

def get_accuracy(model, data, target):
    correct = 0
    total = 0
    N = data.shape[0]
    labels = np.reshape(target, (N, 10))
    images = np.reshape(data, (N, 784))
    output = model.forward(images)
    pred = np.argmax(output, axis=1)
    label = np.argmax(labels, axis=1)
    correct += (np.sum(pred == label))
    total += pred.shape[0]
    return correct / total

def train(model,trainData, validData, testData, trainTarget, validTarget, testTarget, num_epochs = 10):
    # initialize all the weights
    losses, valid_losses, test_losses, train_acc, valid_acc, test_acc = [], [], [], [], [], []
    epochs = []
    
    assert trainData.shape[0] == 10000
    assert validData.shape[0] == 6000

    
    N_train = trainData.shape[0]
    N_valid = validData.shape[0]
    N_test = testData.shape[0]

    print(trainData.shape[0], validData.shape[0])
    
    train_labels = np.reshape(trainTarget, (N_train, 10))
    train_images = np.reshape(trainData, (N_train, 784))
    
    valid_images = np.reshape(validData, (N_valid, 784))
    valid_labels = np.reshape(validTarget, (N_valid, 10))
    
    test_images = np.reshape(testData, (N_test, 784))
    test_labels = np.reshape(testTarget, (N_test, 10))
    
    for epoch in range(num_epochs):
        train_loss = 0
        
        pred = model.forward(train_images)
        loss = averageCE(train_labels, pred)
        
        train_loss += loss
        model.backward(train_labels)
        model.update()
        losses.append(float(train_loss))
    
        valid_loss = 0
        pred = model.forward(valid_images)
        valid_loss += averageCE(valid_labels, pred)
        valid_losses.append(float(valid_loss)) 

        test_loss = 0
        pred = model.forward(test_images)
        test_loss += averageCE(test_labels, pred)
        test_losses.append(float(test_loss)) 

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, trainData, trainTarget))
        valid_acc.append(get_accuracy(model, validData, validTarget))
        test_acc.append(get_accuracy(model, testData, testTarget))

        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f; Test Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1], test_acc[-1]))
        
     # plotting
    plt.title("Training Curves")
    plt.plot(epochs, losses, label="Train")
    plt.plot(epochs, valid_losses, label="Validation")
    plt.plot(epochs, test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Accuracy Curves")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.plot(epochs, test_acc, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()



def tensor_model(keep_prob = 1, reg = 0, padding='SAME'): 
    tf.set_random_seed(421)
    
    X = tf.placeholder(tf.float64, [None, 28, 28, 1], name="x")
    y_target = tf.placeholder(tf.float64, [None, 10], name="y_target")
    filter = tf.get_variable("conv_filter", shape=[3, 3, 1, 32], dtype="float64",  initializer=tf.contrib.layers.xavier_initializer(uniform=False)) 
    
    weights = {
        'w1': tf.get_variable("W1", shape=[14*14*32, 784], dtype="float64", initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
        'w2': tf.get_variable("W2", shape=[784, 10], dtype="float64", initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    }
    bias = {
        'b0': tf.get_variable("b0", shape=[32], dtype="float64", initializer=tf.zeros_initializer()),
        'b1': tf.get_variable("b1", shape=[784], dtype="float64", initializer=tf.zeros_initializer()),
        'b2': tf.get_variable("b2", shape=[10], dtype="float64", initializer=tf.zeros_initializer()),
    }
    
    layer_1 = tf.nn.conv2d(X, filter = filter, strides = [1, 1, 1, 1], padding=padding)
    layer_1 += bias['b0']
    layer_1 = tf.nn.relu(layer_1)
    mean, variance = tf.nn.moments(layer_1, axes=[0])
    layer_1 = tf.nn.batch_normalization(layer_1, mean, variance, None, None, 1e-9)
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
    layer_1 = tf.layers.flatten(layer_1)
    layer_2 = tf.matmul(layer_1, weights['w1']) + bias['b1']
    layer_2 = tf.nn.relu(tf.nn.dropout(layer_2, keep_prob))
    layer_3 = tf.matmul(layer_2, weights['w2']) + bias['b2']
    
    y_predicted = tf.nn.softmax(layer_3, name = "y_predicted")
    loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=layer_3)) + reg*tf.nn.l2_loss(conv_filter) + reg*tf.nn.l2_loss(weights['w1']) + reg*tf.nn.l2_loss(weights['w2'])
    train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=loss)
    correct_prediction = tf.equal(tf.argmax(y_predicted, dimension = 1), tf.argmax(y_target, dimension = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return weights, bias, X, y_target, y_predicted, loss, train, accuracy



def SGD(batch_size, epochs):
    tf.reset_default_graph()
    
    weights, bias, X, y_target, y_predicted, loss, train, accuracy = tensor_model()
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    trainData,trainTarget = shuffle(trainData, trainTarget)
    
    B = batch_size
    b_count = 0
    cur_epoch = 0
    train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc = [], [], [], [], [], []
    while cur_epoch < epochs:              
        if trainData[b_count * B : (b_count+1) * B].shape == (32, 28, 28):
            feeddict = {X: np.reshape(trainData[b_count * B : (b_count+1) * B], [B, 28, 28, 1]), y_target: trainTarget[b_count * B : (b_count+1) * B]}
            sess.run([train, loss, weights, bias, y_predicted, accuracy], feed_dict=feeddict)
        else:
            b_count = 0
            trainData, trainTarget = shuffle(trainData, trainTarget)
            
            err = loss.eval(feed_dict={X:np.reshape(trainData, [trainData.shape[0], 28, 28, 1]), y_target:trainTarget})
            acc = accuracy.eval(feed_dict={X:np.reshape(trainData, [trainData.shape[0], 28, 28, 1]), y_target:trainTarget})
            train_loss.append(err)
            train_acc.append(acc)
            
            err = loss.eval(feed_dict={X:np.reshape(validData, [validData.shape[0], 28, 28, 1]), y_target:validTarget})
            acc = accuracy.eval(feed_dict={X:np.reshape(validData, [validData.shape[0], 28, 28, 1]), y_target:validTarget})
            valid_loss.append(err)
            valid_acc.append(acc)  
            
            err = loss.eval(feed_dict={X:np.reshape(testData, [testData.shape[0], 28, 28, 1]), y_target:testTarget})
            acc = accuracy.eval(feed_dict={X:np.reshape(testData, [testData.shape[0], 28, 28, 1]), y_target:testTarget})
            test_loss.append(err)
            test_acc.append(acc)
            
            print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f; Test Acc %f" % (cur_epoch+1, train_loss[-1], train_acc[-1], valid_acc[-1], test_acc[-1]))
            cur_epoch += 1 
        b_count += 1       
    sess.close()
    return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc


