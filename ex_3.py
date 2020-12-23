import sys
import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=5)

NUN_OF_LABELS = 10


def sigmoid(x):
    ret = 1 / (1 + np.exp(-x))
    return ret


def convert_y_one_hot_encoding(train_y_file):
    train_y = []
    with open(train_y_file) as fp:
        lines = fp.readlines()
    for label in lines:
        train_y.append(int(label))
    # Label y foe each example x
    examples = len(train_y)
    hot_labels = []
    for i in range(examples):
        y = np.zeros((1, NUN_OF_LABELS))
        y[0][train_y[i]] = 1
        hot_labels.append(y)
    return hot_labels


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    dz2 = (h2 - np.array(y).transpose())  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 dz2) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def init_neural_network(dim_input):
    layers = [dim_input, 128, NUN_OF_LABELS]  # 128x784=w1 * 784x1=x = 350x1=h --> 10x350=w2 * 350x1 = h = 10x1
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.uniform(-0.01, 0.01, (layers[i + 1], layers[i])))
        biases.append(np.random.uniform(-0.01, 0.01, (layers[i + 1], 1)))
    params = {'W1': weights[0], 'b1': biases[0], 'W2': weights[1], 'b2': biases[1]}
    return params


def training_process(train_x, train_y):
    learning_rate = 0.01
    x_shape = train_x.shape[1]
    params = init_neural_network(train_x.shape[1])
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    for epoch in range(20):
        print(epoch)
        zip_x_y = list(zip(train_x, train_y))
        np.random.shuffle(zip_x_y)
        train_x, train_y = zip(*zip_x_y)
        for x, y in zip(train_x, train_y):
            # convert x shape to (len(x), 1)
            # fprop
            x = x.reshape((x_shape, 1))
            z1 = np.dot(W1, x) + b1
            h1 = sigmoid(z1)
            z2 = np.dot(W2, h1) + b2
            h2 = softmax(z2)
            y_trans = np.array(y).transpose()
            loss = -(np.sum(y_trans * np.log(h2)))
            fprop_cache = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss, 'W1': W1, 'b1': b1,
                           'W2': W2, 'b2': b2}
            bprop_cache = bprop(fprop_cache)
            W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
            db1, dW1, db2, dW2 = [bprop_cache[key] for key in ('b1', 'W1', 'b2', 'W2')]
            # update w
            params['W1'] = W1 - learning_rate * dW1
            params['b1'] = b1 - learning_rate * db1
            params['W2'] = W2 - learning_rate * dW2
            params['b2'] = b2 - learning_rate * db2
        # next x
    # next epoch
    return params


def test_classification(params_train, test_x):
    classification = []
    test_x_shape = test_x.shape[1]
    for x in test_x:
        W1, b1, W2, b2 = [params_train[key] for key in ('W1', 'b1', 'W2', 'b2')]
        # convert x shape to (len(x), 1)
        x = x.reshape((test_x_shape, 1))
        z1 = np.dot(W1, x) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        classification.append(np.argmax(h2))
    with open('test_y', 'w') as f:
        for y_hat in classification:
            f.write("%s\n" % y_hat)


if __name__ == '__main__':
    training_examples_x, train_labels, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    training_examples_x = np.loadtxt(training_examples_x)
    train_labels = convert_y_one_hot_encoding(train_labels)
    test_x = np.loadtxt(test_x)
    # normalization
    training_examples_x /= 255
    test_x /= 255
    params_train = training_process(training_examples_x, train_labels)
    test_classification(params_train, test_x)
