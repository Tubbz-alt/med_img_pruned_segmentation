import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate
from preprocess import read_image
from skimage.io import imsave
import model_config as config
from tqdm import tqdm


class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.layers_list = config.layers
        self.loss_op = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.batch_size = 3
        self.init_values = list()
        self.num_trainable_variables = len(self.trainable_variables)

    def call(self, inputs):

        prev = inputs
        temp_output_storage = dict()
        for layer in config.layers:
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


def attach_tiles(tiles, num_images):
    images = []
    for i in range(num_images):
        up = np.hstack(tiles[i * 4:i * 4 + 2])
        down = np.hstack(tiles[i * 4 + 2:(i + 1) * 4])
        images.append(np.vstack((up, down)))
    return images

def evaluate(model, test_iterator, num_steps, full_image, epoch):
    if not os.path.exists(f'./results/{epoch}'):
        os.mkdir(f'./results/{epoch}')

    tp = tn = fp = fn = 0
    steps_taken = 0
    for input_, label in test_iterator:
        steps_taken += config.eval_batch_size

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

        if not full_image:
            image = attach_tiles(image, 20)
            lbl = attach_tiles(lbl, 20)

        if steps_taken >= num_steps:
            break

    for i, (image, label) in enumerate(zip(image, lbl)):
        image_to_save = (np.hstack((image, label)) * 255).astype(np.uint8)
        imsave(f'./results/{epoch}/{i}.jpg', image_to_save)

    dice = (2 * tp) / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)

    return dice, accuracy


def main():

    tf.random.set_seed(1020202)
    model = Model()
    full_image = True
    train_iterator, num_train_steps, test_iterator, num_test_steps = read_image('./../PH2Dataset/PH2Dataset/PH2Datasetimages', size=256, full_image=full_image)
    ## You can add different data augmentations in the read_image function when making the image generator. (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#__init__)

    epochs = 100

    max_dice = 0
    max_acc = 0
    total_size = total_count_nonzero_mask = 1
    layers_mask = None
    epochs_to_prune = [50]
    latest_pruning = -1
    back_up_distance = 1
    it = tqdm(range(1, epochs + 1), desc='Training', ncols=0)
    for epoch in it:

        train(model, train_iterator, num_train_steps, layers_mask)

        if (epoch - back_up_distance) in epochs_to_prune:
            model.take_back_up()

        if epoch in epochs_to_prune:
            model.take_back_up()
            layers_mask = model.compute_mask(config.prune_layers, 0.5)
            model.prune_connections(layers_mask)
            latest_pruning = epoch
            print(f'max dice before pruning: {max_dice}')
            max_dice = 0

        if layers_mask is not None:
            total_count_nonzero = 0
            total_count_nonzero_mask = 0
            total_size = 0
            for ivar, imask in zip(model.trainable_variables, layers_mask):
                if 'conv' in ivar.name:
                    total_size += tf.size(ivar)
                    total_count_nonzero += tf.math.count_nonzero(ivar)
                    total_count_nonzero_mask += np.count_nonzero(imask)
            assert total_count_nonzero == total_count_nonzero_mask

        dice, accuracy = evaluate(model, test_iterator, num_test_steps, full_image, epoch)

        max_dice = max([max_dice, dice])
        max_acc = max([max_acc, accuracy])
        it.set_postfix(
            Accuracy='{:.3f}'.format(accuracy * 100),
            Max_Accuracy='{:.3f}'.format(max_acc * 100),
            Dice='{:.3f}'.format(dice * 100),
            Max_Dice='{:.3f}'.format(max_dice * 100),
            Recent_pruning='{:2d}'.format(latest_pruning),
            Nonzero='{:.2f}'.format((total_count_nonzero_mask / total_size) * 100))


if __name__ == '__main__':

    main()

