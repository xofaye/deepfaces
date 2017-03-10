import numpy as np
from matplotlib.pyplot import *


def one_hot_encode(classes):
    one_hot = np.zeros([len(classes), max(classes) + 1])
    for i, j in enumerate(classes):
        one_hot[i, j] = 1
    return one_hot


def make_sets(data, classes, train=5000, validation=1000):
    np.random.seed(0)
    source = zip(data, classes)
    np.random.shuffle(source)
    data, classes = zip(*source)
#    data = np.array(np.array(data).astype('float32')/np.max(data))
    classes = np.array(classes)

    x_train = data[:train]
    t_train = classes[:train]

    mid = validation + train
    x_validation = data[train:mid]
    t_validation = classes[train:mid]

    x_test = data[mid:mid+validation]
    t_test = classes[mid:mid+validation]

    return x_train, t_train, x_validation, t_validation, x_test, t_test


def forward_prop(inputs, weights, biases, softmax):
    layers = len(weights)
    outputs = []
    activations = []
    for l in range(layers):
        curr = np.dot(inputs, weights[l]) + biases[l]
        outputs.append(curr)
        inputs = softmax[l](curr)
        activations.append(curr)
    final_activation = inputs
    return final_activation, outputs, activations


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    # np.exp(y)/tile(sum(exp(y),0), (len(y),1))
    return np.exp(y) / (np.exp(y).sum(axis=1))[:, None]


def cross_entropy(x, w, b, funcs, y):
    # return -sum(y_*log(y))
    output = forward_prop(x, w, b, funcs)[0]
    return (1.0/len(y)) * -np.sum(y*np.log(output))


def calculate_accuracy(x, w, b, y):
    output = forward_prop(x, [w], [b], [softmax])[0]
    successes = np.equal(np.argmax(output, 1), np.argmax(y, 1))
    accuracy = np.sum(successes).astype('float32') / len(y)
    return accuracy


def calculate_cost(x, w, b, y):
    cost = cross_entropy(x, [w], [b], [softmax], y)
    return np.sum(cost)


def visualize_weights(w):
    np.random.seed(5)
    tags = np.random.choice(w.shape[1], 10, False)
    tags.sort()
    fig, axs = subplots(nrows=2, ncols=5)
    fig.suptitle('Weight Visualization')
    for i, ax in enumerate(axs.flat):
        heatmap = ax.imshow(w[:, tags[i]].reshape((28, 28)), cmap=cm.coolwarm)
        ax.set_title(i)
        ax.set_axis_off()
    fig.subplots_adjust(right=0.7)
    fig.colorbar(heatmap, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
    savefig('images/part4_weight.png')
    return tags


def make_sample_sets(data, classes):
    train = 600
    validation = 200
    np.random.seed(20)
    source = zip(data, classes)
    np.random.shuffle(source)
    data, classes = zip(*source)
    data = np.array(np.array(data).astype('float32')/np.max(data))
    classes = np.array(classes)

    x_train = data[:train]
    t_train = classes[:train]

    mid = validation + train
    x_validation = data[train:mid]
    t_validation = classes[train:mid]

    x_test = data[mid:mid+validation]
    t_test = classes[mid:mid+validation]
    x_noise = [img + np.random.normal(0, 0.2, 784) for img in x_test]

    return x_train, t_train, x_validation, t_validation, x_noise, t_test


def grad_descent(f, df, x, y, init_t, alpha):
    """
    Code from CSC411 Course Website
    """
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 40000
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
#        if iter % 500 == 0:
#            print "Iter", iter
#            print "f(x) = %.2f" % f(x, y, t)
        iter += 1
    return t


def linear_classification(test, theta):
    predictions = []
    for row in test:
        inner = []
        for i in range(10):
            p = theta[0][i]
            for j in range(len(row)):
                p += np.dot(theta[j+1][i], row[j])
            inner.append(p)
        predictions.append(inner)
    return predictions


def linear_accuracy(test, actual):
    successes = np.equal(np.argmax(test, 1), np.argmax(actual, 1))
    accuracy = np.sum(successes).astype('float32') / len(actual)
    return accuracy * 100


def rgb2gray(rgb):
    """
    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
