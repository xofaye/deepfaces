import numpy as np
import tensorflow as tf
from helper import one_hot_encode, make_sets, rgb2gray
from glob import glob
from scipy.misc import imread, imresize
from copy import deepcopy
import time, datetime
import os
from operator import mul
from AlexNet.myalexnet_forward import alexnet_main
import cPickle

CROPPED_DIR = 'cropped/'
UNCROPPED_DIR = 'uncropped'
ALL_ACTORS = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
SKIP = ['baldwin132.jpg', 'chenoweth85.jpg''drescher92.jpg', 'hader101.jpg']
conv = True

if not os.path.exists("results"):
    os.makedirs("results")


def read_images(actors):
    faces = []
    classes = []
    codes = one_hot_encode([0, 1, 2, 3, 4, 5])
    for k, name in enumerate(actors):
        current = sorted(set(glob(CROPPED_DIR + name + "*")))
        for i in range(len(current)):
            face = imread(current[i])
            faces.append(face)
            classes.append(codes[k])
    return faces, classes


def reshape_faces(faces):
    for i, face in enumerate(faces):
        try:
            face = imresize(face, (227, 227))
            img = (np.random.random((1, 227, 227, 3)) / 255.).astype('float32')
            img[0, :, :, :] = face[:, :, :3]
            img = img - np.mean(img)
            faces[i] = img
        except:
            pass
    return faces


def train_neural_net(sets, hidden_units, fxns, batch_size=150, dropout=0.8, index=0, lmbda=0,
                     max_iter=1500, best=float('inf'), conv_input=False, save=False, title=None):

    x_train, t_train, x_validation, t_validation, x_test, t_test = sets

    in_size = reduce(mul, list(x_train[0].shape))
    EPS = 1e-5

    x = tf.placeholder(tf.float32, [None, in_size], name="Input")
    y = tf.placeholder(tf.float32, [None, 6], name="Expected_Output")
    keep_prob = tf.placeholder(tf.float32, name="Keep_Probability")

    ''' NEURAL NETWORK TOPOLOGY '''
    # Hidden Layer
    with tf.name_scope("Hidden_Layer") as scope:
        w0 = tf.Variable(tf.truncated_normal([in_size, hidden_units[0]], mean=0, stddev=0.01), name="Weight")
        b0 = tf.Variable(tf.ones([hidden_units[0]])*0.1, name="Bias")
        a0 = fxns[0](tf.matmul(x, w0) + b0)
        d0 = tf.nn.dropout(a0, keep_prob)
        out = d0

    # Second hidden layer
    if len(hidden_units) > 1:
        with tf.name_scope("2nd_Hidden_Layer") as scope:
            w1 = tf.Variable(tf.truncated_normal([hidden_units[0], hidden_units[1]], mean=0, stddev=0.01), name="Weight")
            b1 = tf.Variable(tf.ones([hidden_units[1]])*0.1, name="Bias")
            a1 = fxns[1](tf.matmul(out, w1) + b1)
            d1 = tf.nn.dropout(a1, keep_prob)
            out = d1

    # Output Layer
    with tf.name_scope("Output_Layer") as scope:
        wout = tf.Variable(tf.truncated_normal([hidden_units[-1], 6], mean=0, stddev=0.01), name="Weight")
        bout = tf.Variable(tf.ones([6])*0.1, name="Bias")
        logits = tf.matmul(out, wout) + bout
        output = tf.nn.softmax(logits)
    ''' END NEURAL NETWORK TOPOLOGY '''


    ''' OBJECTIVE PARAMETERS '''
    #Training Specification
    with tf.name_scope("Training") as scope:
        iter_var = tf.Variable(0)
        with tf.name_scope("Regularization") as scope:
            regularizer = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) if len(hidden_units) > 1 else tf.nn.l2_loss(w0)
            l1_regularizer = tf.reduce_mean(tf.abs(w0)) + tf.reduce_mean(tf.abs(w1)) if len(hidden_units) > 1 else tf.reduce_mean(tf.abs(w0))
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        cost = tf.reduce_mean(cost_batch) + lmbda * regularizer
        optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost, global_step=iter_var)

    # Test accuracy
    with tf.name_scope("Evaluation") as scope:
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
    ''' END OBJECTIVE PARAMETERS '''

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ''' TENSORBOARD CONFIGURATION '''
    writer = tf.summary.FileWriter("tboard/%d" %(index), sess.graph_def)

    # Histograms
    w_hist = tf.summary.histogram("1st_Weights", w0)
    b_hist = tf.summary.histogram("1st_Biases", b0)
    y_hist = tf.summary.histogram("Output", output)
    hists = tf.summary.merge([w_hist, b_hist, y_hist])
    if len(hidden_units) > 1:
        w1_hist = tf.summary.histogram("2nd_Weights", w1)
        b1_hist = tf.summary.histogram("2nd_Biases", b1)
        hists = tf.summary.merge([hists, w1_hist, b1_hist])

    # Summaries
    train_accuracy_summary = tf.summary.scalar("Train_Accuracy", accuracy)
    validation_accuracy_summary = tf.summary.scalar("Validation_Accuracy", accuracy)
    test_accuracy_summary = tf.summary.scalar("Test_Accuracy", accuracy)

    train_cost_summary = tf.summary.scalar("Train_Cost", cost)
    validation_cost_summary = tf.summary.scalar("Validation_Cost", cost)
    test_cost_summary = tf.summary.scalar("Test_Cost", cost)

    train_ops = [tf.summary.merge([hists, train_accuracy_summary, train_cost_summary]), cost, accuracy]
    validation_ops = [tf.summary.merge([validation_accuracy_summary, validation_cost_summary]), cost, accuracy]
    test_ops = [tf.summary.merge([test_accuracy_summary, test_cost_summary]), cost, accuracy]
    '''END TENSORBOARD CONFIGURATION'''

    last_cost = 0
    last_val = 0
    costs, accuracies = [], []
    with sess.as_default():
        # Training cycle
        total_batches = int(360 / batch_size)
        while True:
            iter = iter_var.eval()/total_batches+1
            # Complete training epoch
            for i in range(total_batches):
                batch_xs = x_train[i*batch_size:(i+1)*batch_size]
                batch_ys = t_train[i*batch_size:(i+1)*batch_size]
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            # Evaluate network
            if iter % 5 == 0:
                train = sess.run(train_ops, feed_dict={x: x_train, y: t_train, keep_prob: 1})
                writer.add_summary(train[0], iter)

                validation = sess.run(validation_ops, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
                writer.add_summary(validation[0], iter)

                test = sess.run(test_ops, feed_dict={x: x_test, y: t_test, keep_prob: 1})
                writer.add_summary(test[0], iter)
                costs.append([train[1], validation[1], test[1]])
                accuracies.append([train[2], validation[2], test[2]])
                if abs(train[1] - last_cost) < EPS:
                    print "Converged!"
                    break

                if validation[2] > last_val:
                    print "New best generation:"
                    print "\t Accuracy: %4.2f%%" % (validation[2])
                    bench_val = [train[1], validation[1], test[1]], [train[2], validation[2], test[2]]
                    last_val = validation[2]
                    if len(hidden_units) > 1:
                        params = [w0.eval(), w1.eval(), b0.eval(), b1.eval(), wout.eval(), bout.eval()]
                    else:
                        params = [w0.eval(), b0.eval(), wout.eval(), bout.eval()]

                last_cost = train[1]

                if iter % 50 == 0:

                    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                    print "[%s] Iteration: %3d" % (st, iter)
                    print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train[1], train[2])
                    print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation[1], validation[2])
                    print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test[1], test[2])

                    if iter >= 500 and last_val < 50:
                        break

            if iter >= max_iter:
                print "Reached max iterations."
                break

    if validation[2] > best or save:
        with open('results/neural_network_%s%s.pickle' %('conv' if conv_input else 'ff', '_'+str(title) if title else ''), 'wb') as f:
            cPickle.dump(params, f)

    sess.close()

    print "Optimization Finished!"

    return bench_val, costs, accuracies


faces, classes = read_images(ALL_ACTORS)

dim = (100, 100)
dimf = dim[0] * dim[1]
faces_cp = deepcopy(faces)
for img in faces:
    try:
        img = rgb2gray(img)
        img = imresize(img, dim)
        img = img.reshape(dimf)/255.0
    except:
        pass

sets = make_sets(faces_cp, classes, 360, 120)
i = 0
best_val_accuracy = 0

try:
    faces = reshape_faces(deepcopy(faces_cp))
    maxlayer = 1
    while True:
        i += 1
        np.random.seed(5)
        nlayer = np.random.randint(1,maxlayer+1)
        funcs_n = np.array(['relu', 'tanh'])
        funcs = np.array([tf.nn.relu, tf.nn.tanh])
        hu = np.random.randint(200, 850, nlayer)
        idx = np.random.choice(2, nlayer)
        funs = funcs_n[idx]
        fun = funcs[idx]
        lmbda = np.random.randint(0, 10) * 10 ** np.random.randint(-5, -2)
        convlayer = None
        if conv:
            convlayer = np.random.randint(1, 6)
            print "Preparing AlexNet outputs:"
            set = make_sets(alexnet_main(faces, convlayer), classes, 360, 120)
            print "\n"
            print "Model %d:" %i
            print "\t", hu, funs, lmbda, convlayer
            bench,cs,acs = train_neural_net(set, hu, fun, dropout=0.5, batch_size=50, index=i, lmbda=lmbda,
                                                best=best_val_accuracy, max_iter=1500, conv_input=conv)
            c,a = bench
            # Found new model according to validation accuracy
            if a[1] > best_val_accuracy:
                best_val_accuracy = a[1]
                print "Found new best model!"
                print "\tCost (Accuracy):"
                print "\t\t %f (%4.2f%%)" %(best_val_accuracy, a[1])
                print "\tModel:"
                print "\t\t", hu, funs, lmbda, "\n\n"
                if conv:
                    with open('results/models_conv_best.csv', 'w+') as fl:
                        fl.write("%s, %s, %s, %d, %5.4f, "
                                 "%5.4f, %5.4f, %5.2f, %5.2f, %5.2f\n"
                                 %(hu, funs, lmbda, convlayer, c[0], c[1], c[2], a[0], a[1], a[2]))
                else:
                    with open('results/models_best.csv', 'w+') as fl:
                        fl.write(" %s, %s, %s, %5.4f, %5.4f,"
                             " %5.4f, %5.2f, %5.2f, %5.2f\n" %(hu, funs, lmbda,
                                                               c[0], c[1], c[2], a[0],a[1], a[2]))
            if conv:
                with open('results/models_conv.csv', 'a') as fl:
                    fl.write("%d, %s, %s, %s, %d, %5.4f, "
                             "%5.4f, %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs,
                                lmbda, convlayer, c[0], c[1], c[2], a[0], a[1], a[2]))

            else:
                with open('results/models.csv', 'a') as outfile:
                    outfile.write("%d, %s, %s, %s, %5.4f, %5.4f,"
                    " %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs, lmbda, c[0], c[1], c[2], a[0], a[1], a[2]))

            if i >= 25:
                break
except KeyboardInterrupt:
    print "Stopping hyper-parameter search"