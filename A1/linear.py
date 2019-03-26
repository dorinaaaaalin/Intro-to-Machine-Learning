import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):

    x = np.transpose(x)
    pred = np.dot(np.transpose(W), x)+b
    return np.squeeze(np.add(np.mean((y - pred) ** 2), np.dot(np.transpose(W), W) * reg / 2), axis=0)

def gradMSE(W, b, x, y, reg):

    x = np.transpose(x)
    pred = np.dot(np.transpose(W), x)+b
    grad_W = -2*np.mean(np.multiply(y-pred, x), axis=1)
    grad_W = np.expand_dims(grad_W, axis=1)
    grad_W += 2*reg*W
    # grad_W = np.expand_dims(grad_W, axis=1)
    grad_b = -2*np.mean(y-pred)
    grad_b = np.expand_dims(grad_b, axis=1)
    return grad_W, grad_b


def sigmoid(x):

    return (1/(1+np.exp(-x)))


def crossEntropyLoss(W, b, x, y, reg):

    x = np.transpose(x)
    pred = sigmoid(np.dot(np.transpose(W), x) + b)
    return np.squeeze(np.add(np.mean(-1*y*np.log(pred)-(1-y)*np.log(1-pred)), np.dot(np.transpose(W), W)*reg*0.5), axis=0)


def gradCE(W, b, x, y, reg):

    x = np.transpose(x)
    pred = sigmoid(np.dot(np.transpose(W), x)+b)
    grad_W = np.mean(np.multiply(pred-y, x), axis=1)
    grad_W = np.expand_dims(grad_W, axis=1)
    grad_W += 2*reg*W
    grad_b = np.mean(pred-y)
    grad_b = np.expand_dims(grad_b, axis=1)
    return grad_W, grad_b

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):

    # Initialize the old weight matrix
    if lossType == None:
        print("No loss specified")
        return
    training_losses = []
    iters = []
    W_old = W - 10*EPS
    W_new = W.copy()
    b_new = b.copy()
    current_iter = 0
    while np.linalg.norm(W_new-W_old) > EPS and current_iter < iterations:
        W_old = W_new.copy()
        if lossType == "MSE":
            grad_W, grad_b = gradMSE(W_new, b_new, trainingData, trainingLabels, reg)
            W_new -= alpha*grad_W
            b_new -= alpha*grad_b
            if current_iter % 100 == 0:
                training_loss = MSE(W_new, b, trainingData, trainingLabels, reg)

                print("Current Epoch: %d" % current_iter)
                print("Training MSE:  %.4f" % training_loss)

                iters.append(current_iter)
                training_losses.append(training_loss)

            current_iter += 1
        elif lossType == "CE":
            grad_W, grad_b = gradCE(W_new, b_new, trainingData, trainingLabels, reg)
            W_new -= alpha*grad_W
            b_new -= alpha*grad_b
            if current_iter % 100 == 0:
                training_loss = crossEntropyLoss(W_new, b, trainingData, trainingLabels, reg)

                print("Current Epoch: %d" % current_iter)
                print("Training CE:  %.4f" % training_loss)

                iters.append(current_iter)
                training_losses.append(training_loss)

            current_iter += 1

    return W_new, b_new, iters, training_losses

def testGD():
    #Test program for gradient, start by seeding the RNG
    np.random.seed(421)
    h = 0.00001
    #initialize some noise
    x = np.random.normal(0, 0.5, (10, 3))
    #initialize the weights
    W = np.random.normal(0.1, 0.4, (3, 1))
    test = np.array([0, h, 0])
    test = test[:, np.newaxis]
    plus = W+test
    minus = W-test
    #make up some labels
    y = np.ones((5))
    y = np.hstack((np.zeros(5), y))
    np.random.shuffle(y)

    positive = MSE(plus, 0, x, y)
    negative = MSE(minus, 0, x, y)
    print((positive-negative)/(2*h))
    print(gradMSE(W, 0, x, y))
    b = np.zeros((1,1))

    W_new, b_new, iters, losses = grad_descent(W, b, x, y, 0.0001, 1000, 0.1, 1e-7)

def batchGD():
    #---Batch Gradient Descent---#
    #Set the random seed
    np.random.seed(421)
    #Get the data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((-1, 28*28))
    validData = validData.reshape((-1, 28*28))
    testData = testData.reshape((-1, 28*28))
    trainTarget = np.transpose(trainTarget)
    validTarget = np.transpose(validTarget)
    testTarget = np.transpose(testTarget)

    # W = np.ones((784, 1))
    W = np.random.normal(loc=0, scale=0.5, size=(784, 1))
    b = np.zeros((1, 1))

    # Solutions to "Tuning the Learning Rate"

    # W_new, b, iters1, losses1 = grad_descent(W, b, trainData, trainTarget, 0.0001, 5001, 0, 1e-7, lossType="MSE")
    # W_new, b, iters2, losses2 = grad_descent(W, b, trainData, trainTarget, 0.001, 5001, 0, 1e-7, lossType="MSE")
    # W_new, b, iters3, losses3 = grad_descent(W, b, trainData, trainTarget, 0.005, 5001, 0, 1e-7, lossType="MSE")
    # plt.xlabel("Epochs")
    # plt.ylabel("Training loss")
    # plt.title("Gradient Descent for notMNIST")
    # plt.plot(iters1, losses1, 'r--', label=r"Training Loss, $\alpha=0.0001$")
    # plt.plot(iters2, losses2, 'b--', label=r"Training Loss, $\alpha=0.001$")
    # plt.plot(iters3, losses3, 'y--', label=r"Training Loss, $\alpha=0.005$")
    # plt.legend()
    # plt.savefig("GD_learning_rate.jpg")

    # t_prediction = np.dot(np.transpose(W_new), np.transpose(trainData)) + b
    # t_prediction = np.where(t_prediction > 0.5, 1, 0)
    # print("Final Training Accuracy: %.4f" % np.mean(np.equal(t_prediction, trainTarget)))
    #
    # v_prediction = np.dot(np.transpose(W_new), np.transpose(validData)) + b
    # v_prediction = np.where(v_prediction > 0.5, 1, 0)
    # print("Final Validation Accuracy: %.4f" % np.mean(np.equal(v_prediction, validTarget)))
    #
    # t_prediction = np.dot(np.transpose(W_new), np.transpose(testData)) + b
    # t_prediction = np.where(t_prediction > 0.5, 1, 0)
    # print("Final Test Accuracy: %.4f" % np.mean((np.equal(t_prediction, testTarget))))

    # Solutions to "Generalization"

    W_new_1, b_1, iters1, losses1 = grad_descent(W, b, trainData, trainTarget, 0.005, 5001, 0.001, 1e-7, lossType="MSE")
    W_new_2, b_2, iters2, losses = grad_descent(W, b, trainData, trainTarget,  0.005, 5001, 0.01, 1e-7, lossType="MSE")
    W_new_3, b_3, iters3, losses3 = grad_descent(W, b, trainData, trainTarget, 0.005, 5001, 0.1, 1e-7, lossType="MSE")

    v_prediction = np.dot(np.transpose(W_new_1), np.transpose(validData)) + b_1
    v_prediction = np.where(v_prediction > 0.5, 1, 0)
    print("Validation Accuracy for lambda=0.001: %.4f" % np.mean(np.equal(v_prediction, validTarget)))
    v_prediction = np.dot(np.transpose(W_new_2), np.transpose(validData)) + b_2
    v_prediction = np.where(v_prediction > 0.5, 1, 0)
    print("Validation Accuracy for lambda=0.01: %.4f" % np.mean(np.equal(v_prediction, validTarget)))
    v_prediction = np.dot(np.transpose(W_new_3), np.transpose(validData)) + b_3
    v_prediction = np.where(v_prediction > 0.5, 1, 0)
    print("Validation Accuracy for lambda=0.1: %.4f" % np.mean(np.equal(v_prediction, validTarget)))
    t_prediction = np.dot(np.transpose(W_new_1), np.transpose(testData)) + b_1
    t_prediction = np.where(t_prediction > 0.5, 1, 0)
    print("Final Test Accuracy for lambda=0.001: %.4f" % np.mean((np.equal(t_prediction, testTarget))))
    t_prediction = np.dot(np.transpose(W_new_2), np.transpose(testData)) + b_2
    t_prediction = np.where(t_prediction > 0.5, 1, 0)
    print("Final Test Accuracy for lambda=0.01: %.4f" % np.mean((np.equal(t_prediction, testTarget))))
    t_prediction = np.dot(np.transpose(W_new_3), np.transpose(testData)) + b_3
    t_prediction = np.where(t_prediction > 0.5, 1, 0)
    print("Final Test Accuracy for lambda=0.1: %.4f" % np.mean((np.equal(t_prediction, testTarget))))

    # Solutions to "Comparing Batch GD with normal equation"

    # normal_W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(trainData), trainData)), np.transpose(trainData)), np.transpose(trainTarget))
    # t_prediction = np.dot(np.transpose(normal_W), np.transpose(trainData)) + b
    # t_prediction = np.where(t_prediction > 0.5, 1, 0)
    # print("Normal Equation MSE: %.4f" % np.mean((np.equal(t_prediction, trainTarget))))

    # Solutions to "Comparison to Linear Regression"
    # W_1, b_1, iters1, losses1 = grad_descent(W, b, trainData, trainTarget, 0.005, 5001, 0.001, 1e-7, lossType="MSE")
    # W_2, b_2, iters2, losses2 = grad_descent(W, b, trainData, trainTarget, 0.005, 5001, 0.001, 1e-7, lossType="CE")
    # plt.xlabel("Epochs")
    # plt.ylabel("Training loss")
    # plt.title("Gradient Descent for notMNIST")
    # plt.plot(iters1, losses1, 'r--', label=r"Training Loss, MSE")
    # plt.plot(iters2, losses2, 'b--', label=r"Training Loss, CE")
    # plt.legend()
    # plt.savefig("GD_Comparison.jpg")


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):

    if lossType == None:
        print("No loss type specified")
        return
    # Variable creation
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(shape=[28*28,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
    X_flatten = tf.reshape(X, [-1, 28*28])
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    Lambda = tf.placeholder("float32", name='Lambda')
 
    # Graph definition
    y_predicted = tf.matmul(X_flatten, W) + b

    weight_loss = tf.nn.l2_loss(W) * Lambda
    # Error definition
    if lossType == "MSE":
        loss = tf.losses.mean_squared_error(predictions=y_predicted, labels=y_target)
        loss = loss + weight_loss
    elif lossType == "CE":
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_target, logits=y_predicted)
        loss = loss + weight_loss

    # Training mechanism
    if learning_rate is None:
        learning_rate = 0.001
    if beta1 is None and beta2 is None and epsilon is None:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif beta1:
        optimizer = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
    elif beta2:
        optimizer = tf.train.AdamOptimizer(beta2=beta2, learning_rate=learning_rate)
    elif epsilon:
        optimizer = tf.train.AdamOptimizer(epsilon=epsilon, learning_rate=learning_rate)

    train = optimizer.minimize(loss=loss)
    return W, b, X, y_target, y_predicted, loss, train, Lambda

def stochasticGD(batch_size, epochs, lossfilename=None, accfilename=None, lossType=None, beta1=None, beta2=None, epsilon=None):
    if lossType is None or lossType == "MSE":
        W, b, X, y_target, y_predicted, meanSquaredError, train, Lambda = buildGraph(beta1=beta1, beta2=beta2,
                                                                                     epsilon=epsilon, lossType="MSE")
    elif lossType == "CE":
        W, b, X, y_target, y_predicted, meanSquaredError, train, Lambda = buildGraph(beta1=beta1, beta2=beta2,
                                                                                     epsilon=epsilon, lossType="CE")
    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    print(trainData.shape)
 
    ## Training hyper-parameters
    B = batch_size
    wd_lambda = 0
    wList = []
    trainLoss_list = []
    trainAcc_list = []
    validLoss_list = []
    validAcc_list = []
    testLoss_list = []
    testAcc_list = []
    epochs = epochs
    current_epoch = 0
    i = 0
    while current_epoch <= epochs:
        if (i*B) % trainData.shape[0] == 0 and i != 0:
            i = 0
            randIdx = np.arange(len(trainData))
            np.random.shuffle(randIdx)
            trainData = trainData[randIdx]
            trainTarget = trainTarget[randIdx]

            err = meanSquaredError.eval(feed_dict={X: trainData, y_target: trainTarget, Lambda: wd_lambda})
            acc = np.mean((y_predicted.eval(feed_dict={X: trainData}) > 0.5) == trainTarget)
            trainLoss_list.append(err)
            trainAcc_list.append(acc)

            err = meanSquaredError.eval(feed_dict={X: validData, y_target: validTarget, Lambda: wd_lambda})
            acc = np.mean((y_predicted.eval(feed_dict={X: validData}) > 0.5) == validTarget)
            validLoss_list.append(err)
            validAcc_list.append(acc)

            err = meanSquaredError.eval(feed_dict={X: testData, y_target: testTarget, Lambda: wd_lambda})
            acc = np.mean((y_predicted.eval(feed_dict={X: testData}) > 0.5) == testTarget)
            testLoss_list.append(err)
            testAcc_list.append(acc)
            current_epoch += 1

        feeddict = {X: trainData[i * B:(i + 1) * B], y_target: trainTarget[i * B:(i + 1) * B], Lambda: wd_lambda}
        ## Update model parameters
        sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict=feeddict)
        i += 1

    validation_dict = {"train":(trainData, trainTarget), "valid":(validData, validTarget), "test":(testData, testTarget)}
    for dataset in validation_dict:
        data, target = validation_dict[dataset]
        err = sess.run(meanSquaredError, feed_dict={X: data, y_target: target, Lambda: wd_lambda})
        acc = np.mean((y_predicted.eval(feed_dict={X: data}) > 0.5) == target)
        print("Final %s Loss: %.3f, acc: %.3f"%(dataset, err, acc))

    if lossfilename is None or accfilename is None:
        sess.close()
        return

    plt.xlabel("Epochs")
    if lossType == None:
        plt.ylabel("L2 Regularized MSE Loss")
    elif lossType == "CE":
        plt.ylabel("L2 Regularized Mean CE Loss")
    plt.title("Stochastic Gradient Descent for notMNIST")
    plt.plot(validLoss_list, 'g--', label="Validation Loss")
    plt.plot(trainLoss_list, 'r--', label="Training Loss")
    plt.plot(testLoss_list, 'y--', label="Test Loss")
    plt.legend()
    plt.savefig(lossfilename)
    plt.clf()
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.title("Stochastic Gradient Descent for notMNIST")
    plt.plot(trainAcc_list, 'r--', label="Training Accuracy")
    plt.plot(validAcc_list, 'g--', label="Validation Accuracy")
    plt.plot(testAcc_list, 'y--', label="Test Accuracy")
    plt.legend()
    plt.savefig(accfilename)
    plt.clf()
    sess.close()


def main():
    # testGD()
    # batchGD()
    # stochasticGD(500, 700, "SGD_MSE_500.jpg", "SGD_Accuracy_500.jpg")
    # stochasticGD(100, 700, "SGD_MSE_100.jpg", "SGD_Accuracy_100.jpg")
    # stochasticGD(700, 700, "SGD_MSE_700.jpg", "SGD_Accuracy_700.jpg")
    # stochasticGD(1750, 700, "SGD_MSE_1750.jpg", "SGD_Accuracy_1750.jpg")
    # stochasticGD(100, 700, "SGD_CE_100.jpg", "SGD_CE_Accuracy_100.jpg", lossType="CE")
    # stochasticGD(700, 700, "SGD_CE_700.jpg", "SGD_CE_Accuracy_700.jpg", lossType="CE")
    # stochasticGD(1750, 700, "SGD_CE_1750.jpg", "SGD_CE_Accuracy_1750.jpg", lossType="CE")
    stochasticGD(500, 700, beta1=0.95)
    stochasticGD(500, 700, beta1=0.99)
    stochasticGD(500, 700, beta2=0.99)
    stochasticGD(500, 700, beta2=0.9999)
    stochasticGD(500, 700, epsilon=1e-9)
    stochasticGD(500, 700, epsilon=1e-4)
    stochasticGD(500, 700, beta1=0.95, lossType="CE")
    stochasticGD(500, 700, beta1=0.99, lossType="CE")
    stochasticGD(500, 700, beta2=0.99, lossType="CE")
    stochasticGD(500, 700, beta2=0.9999, lossType="CE")
    stochasticGD(500, 700, epsilon=1e-9, lossType="CE")
    stochasticGD(500, 700, epsilon=1e-4, lossType="CE")



if __name__ == "__main__":

    main()