import argparse
import os
import time
from urllib.request import urlretrieve

import numpy as np

from src.tf_mnist import train
from src.utils import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--blob-path', type=str, dest='blob_path', help="blob path for dataset load")

parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help="mini batch size for training")
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help="# of neurons in the first layer")
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help="# of neurons in the second layer")
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=-6, help="learning rate")
parser.add_argument('--lr-scale', type=str, dest='lr_scale', default='log', help="learning rate scale")

parser.add_argument('--model-dir', type=str, dest='model_dir', help="Directory for storing model")
parser.add_argument('--log-dir', type=str, dest='log_dir', help="Summaries log directory")
args, _ = parser.parse_known_args()


# Download and load data
mnist_path = os.path.join('data', 'mnist')
os.makedirs(mnist_path, exist_ok=True)
X_train_path = os.path.join(mnist_path, 'X_train.npy')
y_train_path = os.path.join(mnist_path, 'y_train.npy')
X_valid_path = os.path.join(mnist_path, 'X_valid.npy')
y_valid_path = os.path.join(mnist_path, 'y_valid.npy')

if not args.blob_path:
    raise ValueError("Data path should be provided")
else:  
    # Download if not exist already
    if not os.path.isfile(X_train_path):
        urlretrieve(args.blob_path + "/X_train.npy", X_train_path)
    if not os.path.isfile(y_train_path):
        urlretrieve(args.blob_path + "/y_train.npy", y_train_path)
    if not os.path.isfile(X_valid_path):
        urlretrieve(args.blob_path + "/X_valid.npy", X_valid_path)
    if not os.path.isfile(y_valid_path):
        urlretrieve(args.blob_path + "/y_valid.npy", y_valid_path)

X_t = np.load(X_train_path)
y_t = np.load(y_train_path)
X_v = np.load(X_valid_path)
y_v = np.load(y_valid_path)

params = vars(args)

mnt_path = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow')  # azurefile mount path
ts = int(round(time.time() * 1000))
params['model_dir'] = os.path.join(mnt_path, '{}_model'.format(ts))
params['log_dir'] = os.path.join(mnt_path, '{}_logs'.format(ts))

logger = Logger(None, 'katib')
logger.log('model_id', ts)  # This is hack, storing the model id as a metric in order to record it.

train(X_t, y_t, X_v, y_v, logger=logger, **params)
