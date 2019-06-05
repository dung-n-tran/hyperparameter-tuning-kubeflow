import os

import numpy as np
import tensorflow as tf


def train(X_train, y_train, X_valid=None, y_valid=None, logger=None, **kwargs):
    print("TensorFlow version:", tf.VERSION)

    training_set_size = X_train.shape[0]

    n_inputs = 28 * 28
    n_h1 = kwargs['n_hidden_1']
    n_h2 = kwargs['n_hidden_2']
    n_outputs = 10
    if kwargs['lr_scale'] == 'org':
        learning_rate = kwargs['learning_rate']
    elif kwargs['lr_scale'] == 'log':
        learning_rate = np.exp(kwargs['learning_rate'])
    else:
        raise ValueError("Invalid value of lr_scale! Should be either 'org' or 'log'.")
    n_epochs = 20
    batch_size = kwargs['batch_size']

    with tf.name_scope('network'):
        # construct the DNN
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int64, shape=None, name='y')
        h1 = tf.layers.dense(X, n_h1, activation=tf.nn.relu, name='h1')
        h2 = tf.layers.dense(h1, n_h2, activation=tf.nn.relu, name='h2')
        output = tf.layers.dense(h2, n_outputs, name='output')

    with tf.name_scope('train'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(output, y, 1)
        acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):

            # randomly shuffle training set
            indices = np.random.permutation(training_set_size)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # batch index
            b_start = 0
            b_end = b_start + batch_size
            for _ in range(training_set_size // batch_size):
                # get a batch
                X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]

                # update batch index for the next batch
                b_start = b_start + batch_size
                b_end = min(b_start + batch_size, training_set_size)

                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

            # evaluate training set TODO instead of evaluation, why don't we just get tf.equal(tf.argmax(y, 1), y_)?
            acc_train = acc_op.eval(feed_dict={X: X_batch, y: y_batch})
            # evaluate validation set
            acc_val = acc_op.eval(feed_dict={X: X_valid, y: y_valid})

            # log accuracies
            if logger is not None:
                logger.log('training_acc', np.float(acc_train))
                logger.log('validation_acc', np.float(acc_val))
            print(epoch, "-- Training accuracy:", acc_train, "; Validation accuracy:", acc_val)

        if logger is not None:
            logger.log('final_acc', np.float(acc_val))

        model_path = os.path.join(kwargs['model_dir'], 'mnist-tf.model')
        os.makedirs(kwargs['model_dir'], exist_ok=True)
        saver.save(sess, model_path)
        print("Model saved at {}".format(kwargs['model_dir']))


def test(meta_path, model_path, X_test, y_test, logger=None, verbose=True):
    tf.reset_default_graph()

    # Load the saved TensorFlow graph.
    saver = tf.train.import_meta_graph(meta_path)
    graph = tf.get_default_graph()

    if verbose:
        for op in graph.get_operations():
            if op.name.startswith('network'):
                print(op.name)
        print('\n')

    # Feed test dataset to the persisted model to get predictions.
    # Input tensor. This is an array of 784 elements, each representing the intensity of a pixel in the digit image.
    X = tf.get_default_graph().get_tensor_by_name("network/X:0")
    # Output tensor. This is an array of 10 elements, each representing the probability of predicted value of the digit.
    output = tf.get_default_graph().get_tensor_by_name("network/output/MatMul:0")

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        k = output.eval(feed_dict={X: X_test})
    # Get the prediction, which is the index of the element that has the largest probability value.
    y_hat = np.argmax(k, axis=1)

    # Calculate the overall accuracy
    test_acc = np.average(y_hat == y_test)
    if verbose:
        print("Accuracy on the test set:", test_acc)

    if logger is not None:
        logger.log('test_acc', test_acc)

    return test_acc
