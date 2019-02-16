from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf


def run(dataset_name, x_train, y_train, x_valid, y_valid, x_test, y_test):
    # start timer
    start = timeit.default_timer()
    np.random.seed(0)
    # get predicioned classes
    predicted_y_test = logistic_regression(dataset_name, x_train, y_train, x_valid, y_valid, x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start
    # compute results
    y_test = y_test.flatten()
    predicted_y_test = np.asarray(predicted_y_test).flatten()
    correct_predict = (y_test == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    return (correct_predict, accuracy, run_time)

#Convenient class for iterating through train set randomly
class DatasetIterator:
    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.b_sz = batch_size
        self.b_pt = 0
        self.d_sz = len(x)
        self.idx = None
        self.randomize()

    # randomize indexs in dataset
    def randomize(self):
        self.idx = np.random.permutation(self.d_sz)
        self.b_pt = 0

    # get the next batch of the dataset
    def next_batch(self):
        start = self.b_pt
        end = self.b_pt + self.b_sz
        idx = self.idx[start:end]
        x = self.x[idx]
        y = self.y[idx]

        self.b_pt += self.b_sz
        if self.b_pt >= self.d_sz:
            self.randomize()

        return x, y

# Convert numpy array to one hot
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# perform logistic regression on input dataset
def logistic_regression(dataset_name, x_train, y_train, x_valid, y_valid, x_test):
    if dataset_name == "MNIST":
        # length of x_train, x_valid, x_test
        n_inputs = 784
    elif dataset_name == "CIFAR10":
        # length of x_train, x_valid, x_test
        n_inputs = 3072
        # reshape input
        x_train = np.reshape(x_train, (40000, 3072))
        y_train = np.reshape(y_train, (40000))
        x_valid = np.reshape(x_valid, (10000, 3072))
        y_valid = np.reshape(y_valid, (10000))
        x_test = np.reshape(x_test, (10000, 3072))
        # change data type of input to float32
        x_train = x_train.astype('float32')
        x_valid = x_valid.astype('float32')
        x_test = x_test.astype('float32')
        # scale data (0,1)
        for i in range(len(x_train)):
            x_train[i] = np.true_divide(x_train[i], 256)
        for i in range(len(x_valid)):
            x_valid[i] = np.true_divide(x_valid[i], 256)
            x_test[i] = np.true_divide(x_test[i], 256)

    Xmean = np.mean(x_train,axis=0) # mean of data vector
    ymean = np.mean(y_train,axis=0) # bias of the model

    X = tf.placeholder(tf.float32,[None, n_inputs])
    Xm = X - Xmean # data centering
    y = tf.placeholder(tf.float32,[None, 10]) #actual distribution (one-hot vector)

    lr = 0.001 # learning rate
    lam_val = 1 # regularization parameter
    theta = tf.Variable(tf.random_normal([n_inputs,10])) # parameter of the linear model (weights)

    logits= tf.matmul(Xm,theta)+ymean # scores
    yp    = tf.nn.softmax(logits) #predicted probability distribution (0 to 1)
    #loss function (cross_entropy)
    cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(yp, 1e-10, 1.0))) + lam_val * tf.reduce_mean(tf.square(theta))

    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = lr) # Adam optimizer

    training_op = optimizer.minimize(cross_entropy)

    y1 = tf.placeholder(tf.float32,[None, 10])
    y2 = tf.placeholder(tf.float32,[None, 10])
    acc = 100.0*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y1,axis=1),tf.argmax(y2,axis=1)),tf.float32))

    init = tf.global_variables_initializer()

    n_epochs = 100
    batch_size = 200

    # use convenient class for iterating over dataset randomly
    d_iterator = DatasetIterator(x_train, y_train, batch_size)

    # number of batches
    n_batches = int(len(x_train) / batch_size)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            # compute model
            for iteration in range(n_batches):
                # get next batch
                X_batch,y_batch = d_iterator.next_batch()
                y_batch = one_hot(y_batch, 10)
                sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

            if epoch%10 == 0:

                theta_value = theta.eval()

                # prediction on training set
                yp_train = yp.eval(feed_dict={X:x_train, theta:theta_value})
                acc_train = acc.eval(feed_dict={y1:one_hot(y_train, 10), y2:yp_train})
                regerr_train = tf.reduce_mean(tf.square(yp_train-one_hot(y_train, 10))).eval()

                # prediction on validation set
                yp_validation = yp.eval(feed_dict={X:x_valid, theta:theta_value})
                acc_validation = acc.eval(feed_dict={y1:one_hot(y_valid, 10), y2:yp_validation})
                regerr_validation = tf.reduce_mean(tf.square(yp_validation-one_hot(y_valid, 10))).eval()

            theta_value = theta.eval()
        # Now that the model is trained, it is the test time!
        yp_test = yp.eval(feed_dict={X:x_test, theta:theta_value})

    # return predicted labels from x_test
    return np.argmax(yp_test, axis=1)

# load input dataset and run logistic regression on it
def run_on_dataset(dataset_name):
    if dataset_name == "MNIST":
        # load MNIST dataset
        mnist = read_data_sets('data', one_hot=False)
        x_train, y_train = (mnist.train._images, mnist.train._labels)
        x_test, y_test = (mnist.test._images, mnist.test.labels)
    elif dataset_name == "CIFAR10":
        # load CIFAR10 dataset
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 10000 validation samples
    x_valid, y_valid = x_train[-10000:], y_train[-10000:]
    # 10000 training samples
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    # get results
    correct_predict, accuracy, run_time = run(dataset_name,
                                              x_train, y_train, x_valid, y_valid, x_test, y_test)

    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy,
                         run_time=run_time)
    return result


def main():
    result_all = OrderedDict()
    # run logistic regression on both MNIST and CIFAR10
    for dataset_name in ["MNIST", "CIFAR10"]:
        result_all[dataset_name] = run_on_dataset(dataset_name)
    # write results to txt file
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))

main()
