from __future__ import print_function
from __future__ import division

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from skimage.io import imsave
import json
from datetime import datetime
import traceback

from old_model import Unet
from old_preprocess import read_image

# Handle command line arguments
parser = argparse.ArgumentParser(description='Run a complete training pipeline based on a given configuration file')
parser.add_argument('--config', dest='config_filepath',
                    help='Configuration .json file (optional). Overwrites existing command-line args')
parser.add_argument('--output_dir', default=os.getcwd(),
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--data_dir', default='./PH2Dataset/PH2Dataset/PH2Datasetimages',
                    help='Data directory')
parser.add_argument('--debug_size', type=int,
                    help='For rapid testing purposes (e.g. debugging), limit training set to a small random sample')
parser.add_argument('--name', dest='experiment_name', default='',
                    help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
parser.add_argument('--no_timestamp',action='store_true',
                    help='If set, a timestamp will not be appended to the output directory name')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
args = parser.parse_args()


def train(model, train_image, train_labels, train_step, sess_, batch_size=3):

    ind = np.arange(0, train_labels.shape[0])
    np.random.shuffle(ind)
    inputs = train_image[ind]
    labels = train_labels[ind]

    for i in range(0, train_labels.shape[0], batch_size):
        input_ = inputs[i: i + batch_size, :, :, :]
        label = labels[i: i + batch_size, :]

        sess_.run(train_step, feed_dict={model.x_: input_, model.y_: label})


def evaluate(model, test_image, test_label, output, sess_, epoch):

    if not os.path.exists('./results/{}'.format(epoch)):
        os.mkdir('./results/{}'.format(epoch))
    ind = np.arange(0, test_label.shape[0])
    np.random.shuffle(ind)
    inputs = test_image
    labels = test_label

    tp = tn = fp = fn = 0
    image_list = []
    for label, input in zip(labels, inputs):

        image = sess_.run(output, feed_dict={model.x_: np.expand_dims(input, 0)})

        image[image > 0.25] = 1.0
        image[image < 0.25] = 0.0

        image = image.astype(np.bool).astype(np.uint8)
        lbl = label.astype(np.bool).astype(np.uint8)

        tp += np.sum(np.logical_and(image == 1, lbl == 1))
        tn += np.sum(np.logical_and(image == 0, lbl == 0))
        fp += np.sum(np.logical_and(image == 1, lbl == 0))
        fn += np.sum(np.logical_and(image == 0, lbl == 1))

        to_save = (np.hstack((image[0], label)) * 255).astype(np.uint8)

        image_list.append(to_save)

    for i, image in enumerate(image_list):
        imsave('./results/{}/{}.jpg'.format(epoch, i), image)

    dice = (2 * tp) / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)

    return dice, accuracy


def setup(args):

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()


    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if not config['no_timestamp'] or len(config['experiment_name']) == 0:
        output_dir += "_" + formatted_timestamp
    create_dirs([output_dir])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def main():

    config = setup(args)

    tf.set_random_seed(1020202)
    sess = tf.Session()

    model = Unet()

    logits, output = model.get_logits_and_segmentation()
    loss = model.get_loss(logits)
    train_step = model.get_train_step(loss)

    sess.run(tf.global_variables_initializer())

    train_inputs_, train_labels_ = read_image(config['data_dir'])

    start_time = time.time()

    for epoch in range(1, config['epochs']):
        train(model, train_inputs_, train_labels_, train_step, sess, batch_size=3)
        dice, accuracy = evaluate(model, train_inputs_, train_labels_, output, sess, epoch)
        print('Epoch: {}, Dice: {:.3f}, Accuracy: {:.3f}'.format(epoch, dice, accuracy))

    logger.info('All Done!')

    total_runtime = time.time() - start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(total_runtime // 3600, (total_runtime // 60) % 60,
                                                                   total_runtime % 60))


if __name__ == '__main__':
    main()
