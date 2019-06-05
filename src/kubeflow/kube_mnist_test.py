import argparse
import os
import logging
from urllib.request import urlretrieve

import numpy as np

from src.tf_mnist import test
from src.utils import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--blob-path', type=str, dest='blob_path', help="blob path for dataset load")
parser.add_argument('--model-dir', type=str, dest='model_dir', default='model', help="Directory for loading model")

args, _ = parser.parse_known_args()

# Download and load data
mnist_path = os.path.join('data', 'mnist')
os.makedirs(mnist_path, exist_ok=True)
X_test_path = os.path.join(mnist_path, 'X_test.npy')
y_test_path = os.path.join(mnist_path, 'y_test.npy')

if not args.blob_path:
    raise ValueError("Data path should be provided")
else:  
    # Download if not exist already
    if not os.path.isfile(X_test_path):
        urlretrieve(args.blob_path + "/X_test.npy", X_test_path)
    if not os.path.isfile(y_test_path):
        urlretrieve(args.blob_path + "/y_test.npy", y_test_path)

X_t = np.load(X_test_path)
y_t = np.load(y_test_path)

test_acc = test(
    os.path.join(args.model_dir, "mnist-tf.model.meta"),
    os.path.join(args.model_dir, "mnist-tf.model"),
    X_t,
    y_t,
    logger=Logger(logging, 'python'),
    verbose=False,
)
