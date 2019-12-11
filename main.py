from __future__ import print_function
from __future__ import division

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

import os
import sys
import tensorflow as tf
import numpy as np
import preprocess
from skimage.io import imsave
import model_config
from tqdm import tqdm
import xlrd
import xlwt
from xlutils.copy import copy
import json
from datetime import datetime
import time
import traceback
import argparse
import warnings


# Handle command line arguments
parser = argparse.ArgumentParser(description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')
parser.add_argument('--config', dest='config_filepath',
                    help='Configuration .json file (optional). Overwrites existing command-line args!')
parser.add_argument('--output_dir', default='/users/gzerveas/data/gzerveas/pruned_med_img_seg/output',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--data_dir', default='/users/gzerveas/data/gzerveas/PH2Dataset/images',
                    help='Data directory')
parser.add_argument('--debug_size', type=int,
                    help='For rapid testing purposes (e.g. debugging), limit training set to a small random sample')
parser.add_argument('--name', dest='experiment_name', default='',
                    help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
parser.add_argument('--no_timestamp', action='store_true',
                    help='If set, a timestamp will not be appended to the output directory name')
parser.add_argument('--records_file', default='/users/gzerveas/data/gzerveas/pruned_med_img_seg/records.xls',
                    help='Excel file keeping all records of experiments')

parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Training batch size')
parser.add_argument('--use_patches', action='store_true',
                    help='If set, training will use patches instead of full images')
parser.add_argument('--prune_at', nargs='+', type=int,
                    help="Space separated values that correspond to the epochs at which pruning will occur")
parser.add_argument('--backup_distance', type=int, default=1,
                    help='Number of training epochs before pruning epoch from which to initialize weights')
parser.add_argument('--prune_param', type=float, default=0.5,
                    help='Pruning parameter which, multiplied by the weight matrix std, gives the pruning threshold')

args = parser.parse_args()

if args.prune_at is None:
    args.prune_at = [50]


class Model(tf.keras.Model):
    def __init__(self, layers_list):

        super(Model, self).__init__()

        self.layers_list = layers_list
        self.loss_op = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.init_values = list()
        self.num_trainable_variables = len(self.trainable_variables)

    @tf.function
    def call(self, inputs):

        prev = inputs
        temp_output_storage = dict()
        for layer in self.layers_list:
            if layer.name != 'concat':
                prev = layer(prev)
                temp_output_storage[layer.name] = prev
            else:
                tensors_to_concat = [temp_output_storage[_name] for _name in layer.input_tensor_names]
                prev = layer(*tensors_to_concat)

        return prev

    def take_back_up(self):
        var_val = [var.numpy() for var in self.trainable_variables]
        self.init_values = var_val

    @tf.function
    def loss(self, logits, labels):

        return tf.reduce_mean(self.loss_op(y_pred=logits, y_true=labels))

    def compute_mask(self, layer_to_prune, threshold):

        masks = list()
        for var in self.trainable_variables:
            if var.name.split('/')[0] in layer_to_prune:
                value = np.abs(var.numpy())
                mask = value > (np.std(value) * threshold)
                masks.append(mask.astype(np.float32))
            else:
                masks.append(np.ones_like(var.numpy()))

        return masks

    def prune_connections(self, mask__):

        for var, value, var_mask in zip(self.trainable_variables, self.init_values, mask__):
            var.assign(np.multiply(value, var_mask))


def train(model, train_iterator, num_steps, mask_):

    steps_taken = 0
    if mask_ is not None:
        num_vars = len(mask_)
    for input_, label in train_iterator:
        steps_taken += 1
        with tf.GradientTape() as tape:
            predictions = model.call(input_)
            loss = model.loss(predictions, label)

        gradients = tape.gradient(loss, model.trainable_variables)
        if mask_ is not None:
            op_vars = model.optimizer.weights
            for i, (v, m, grad, msk) in enumerate(zip(op_vars[num_vars + 1:], op_vars[1:num_vars + 1], gradients, mask_)):
                gradients[i] = tf.multiply(grad, msk)
                v.assign(tf.multiply(v, msk))
                m.assign(tf.multiply(m, msk))
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if steps_taken >= num_steps:
            break


def assemble_tiles(tiles, num_images):
    """Assembles the 4 tiles/patches corresponding to an image, into a single image."""
    images = []
    for i in range(num_images):
        up = np.hstack(tiles[i * 4:i * 4 + 2])
        down = np.hstack(tiles[i * 4 + 2:(i + 1) * 4])
        images.append(np.vstack((up, down)))
    return np.array(images)


def evaluate(model, test_iterator, num_batches, use_patches, epoch, out_dir):

    tp = tn = fp = fn = 0
    batches_processed = 0
    all_output_masks = []
    all_label_masks = []
    for input_, label in test_iterator:
        batch_size = len(label)
        batches_processed += batch_size

        image = model.call(input_).numpy()

        image[image >= 0.5] = 1.0
        image[image < 0.5] = 0.0

        image = image.astype(np.bool).astype(np.uint8)
        lbl = label.astype(np.bool).astype(np.uint8)

        image = np.squeeze(image)
        lbl = np.squeeze(lbl)

        tp += np.sum(np.logical_and(image == 1, lbl == 1))
        tn += np.sum(np.logical_and(image == 0, lbl == 0))
        fp += np.sum(np.logical_and(image == 1, lbl == 0))
        fn += np.sum(np.logical_and(image == 0, lbl == 1))

        if use_patches:
            image = assemble_tiles(image, int(batch_size/4))  # (set_size, 256, 256)
            lbl = assemble_tiles(lbl, int(batch_size/4))  # (set_size, 256, 256)

        all_output_masks.append(image)  # extends list
        all_label_masks.append(lbl)  # extends list

        if batches_processed >= num_batches:
            break

    save_mask_images(all_output_masks, all_label_masks, out_dir, epoch)

    dice = (2 * tp) / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)

    return dice, accuracy, precision, recall


def save_mask_images(pred_masks, lbl_masks, out_dir, epoch):

    pred_masks = np.concatenate(pred_masks)
    lbl_masks = np.concatenate(lbl_masks)

    for i, (image, label) in enumerate(zip(pred_masks, lbl_masks)):
        pred_img = (image * 255).astype(np.uint8)
        lbl_img = (label * 255).astype(np.uint8)

        directory = os.path.join(out_dir, "sample_{}".format(i))
        if not os.path.exists(directory):
            os.makedirs(directory)

        with warnings.catch_warnings():  # stop complaining about low contrast
            warnings.simplefilter("ignore")
            imsave(os.path.join(directory, "pred_{}_epoch_{}.jpg".format(i, epoch)), pred_img)
            imsave(os.path.join(directory, "lbl_{}_epoch_{}.jpg".format(i, epoch)), lbl_img)
            # Save as side-by-side panels
            imsave(os.path.join(directory, "both_{}_epoch_{}.jpg".format(i, epoch)), np.hstack((pred_img, lbl_img)))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

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
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        output_dir += "_" + formatted_timestamp
    create_dirs([output_dir])
    config['output_dir'] = output_dir

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


def export_performance_metrics(filepath, metrics_table, header, sheet_name="metrics"):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    book = xlwt.Workbook()  # excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds the best and final metrics of this experiment as a record in an excel sheet with other experiment records."""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)

    logger.info("Exported performance record to '{}'".format(filepath))


def main():

    start_time = time.time()

    config = setup(args)

    tf.random.set_seed(1020202)
    model = Model(model_config.layers)

    logger.info("Loading and preprocessing data ...")
    train_images, train_labels, test_images, test_labels = preprocess.read_images(config["data_dir"], image_size=256, testset_ratio=0.1, use_patches=config["use_patches"])

    if config["debug_size"]:
        train_images = train_images[:config["debug_size"], :, :, :]
        train_labels = train_labels[:config["debug_size"], :, :, :]

    logger.info("Train images shape: {}".format(train_images.shape))
    logger.info("Train label masks shape: {}".format(train_labels.shape))

    logger.info("Test images shape: {}".format(test_images.shape))
    logger.info("Test label masks shape: {}".format(test_labels.shape))

    num_test_samples = test_labels.shape[0]  # number of test samples (can be whole images or patches)
    num_train_samples = train_labels.shape[0]  # number of train samples (can be whole images or patches)
    num_test_batches = num_test_samples
    num_train_batches = int(num_train_samples / config["batch_size"])

    train_iterator = preprocess.build_iterator(train_images, train_labels, config["batch_size"], shuffle=True)
    test_iterator = preprocess.build_iterator(test_images, test_labels, num_test_samples, shuffle=False)

    prune_at = config["prune_at"]
    backup_distance = config["backup_distance"]

    # initialize loop variables
    max_dice = max_acc = max_prec = max_rec = 0
    total_model_size = total_count_nonzero_mask = 1
    layers_mask = None
    latest_pruning = -1
    it = tqdm(range(1, config["epochs"] + 1), desc='Training', ncols=0)

    metrics = []  # list of lists: for each epoch, stores metrics like accuracy, DICE, non-pruned weights ratio

    for epoch in it:

        train(model, train_iterator, num_train_batches, layers_mask)

        if (epoch + backup_distance) in prune_at:
            model.take_back_up()

        if epoch in prune_at:
            layers_mask = model.compute_mask(model_config.prune_layers, config["prune_param"])
            model.prune_connections(layers_mask)
            latest_pruning = epoch
            print(f'max dice before pruning: {max_dice}')

        if layers_mask is not None:
            total_count_nonzero = 0
            total_count_nonzero_mask = 0
            total_model_size = 0
            for var, mask in zip(model.trainable_variables, layers_mask):
                if 'conv' in var.name:
                    total_model_size += tf.size(var).numpy()
                    total_count_nonzero += tf.math.count_nonzero(var).numpy()
                    total_count_nonzero_mask += np.count_nonzero(mask)
            assert total_count_nonzero == total_count_nonzero_mask

        dice, accuracy, precision, recall = evaluate(model, test_iterator, num_test_batches, config["use_patches"], epoch, out_dir=config["output_dir"])

        pruned_ratio = (total_model_size - total_count_nonzero_mask) / total_model_size
        metrics.append([epoch, accuracy, dice, precision, recall, pruned_ratio])

        max_dice = max([max_dice, dice])
        max_acc = max([max_acc, accuracy])
        max_prec = max([max_prec, precision])
        max_rec = max([max_rec, recall])
        it.set_postfix(
            Accuracy='{:.3f}%'.format(accuracy * 100),
            Max_Accuracy='{:.3f}%'.format(max_acc * 100),
            Dice='{:.3f}%'.format(dice * 100),
            Max_Dice='{:.3f}%'.format(max_dice * 100),
            Precision='{:.3f}%'.format(precision * 100),
            Max_Precision='{:.3f}%'.format(max_prec * 100),
            Recall='{:.3f}%'.format(recall * 100),
            Max_Recall='{:.3f}%'.format(max_rec * 100),
            Last_pruning='{:2d}'.format(latest_pruning),
            Pruned='{:.2f}%'.format(pruned_ratio * 100))

    # Export evolution of metrics over epoch
    header = ["Epoch", "Accuracy", "DICE", "Precision", "Recall", "Pruned ratio"]
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    export_performance_metrics(metrics_filepath, metrics, header)

    # Export record metrics to a file accumulating records from all experiments
    metrics = np.array(metrics)
    best_inds = np.argmax(metrics, axis=0)
    row_values = [config["initial_timestamp"], config["experiment_name"],
                  metrics[best_inds[2], 2], metrics[best_inds[2], 0], metrics[best_inds[2], 5], metrics[-1, 2], metrics[-1, 5], metrics[-1, 0],
                  metrics[best_inds[1], 1],  metrics[-1, 1],  metrics[best_inds[3], 3], metrics[-1, 3],
                  metrics[best_inds[4], 4], metrics[-1, 4]]

    if not os.path.exists(config["records_file"]):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...")
        directory = os.path.dirname(config["records_file"])
        if not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "BEST DICE", "Epoch at BEST", "PrunedR at BEST", "Final DICE", "Final Pruned Ratio", "Final Epoch",
                  "Best Accuracy", "Final Accuracy", "Best Precision", "Final Precision", "Best Recall", "Final Recall"]
        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(config["records_file"])
    else:
        try:
            export_record(config["records_file"], row_values)
        except:
            alt_path = os.path.join(os.path.dirname(config["records_file"]), "record_" + config["experiment_name"])
            logger.error("Failed saving in: '{}'! Will save here instead: {}".format(config["records_file"], alt_path))
            export_record(alt_path, row_values)


    logger.info('All Done!')

    total_runtime = time.time() - start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(total_runtime // 3600, (total_runtime // 60) % 60,
                                                                   total_runtime % 60))


if __name__ == '__main__':

    main()

