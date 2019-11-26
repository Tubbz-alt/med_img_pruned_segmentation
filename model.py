import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate
from preprocess import read_image
from skimage.io import imsave


class Unet(object):
    def __init__(self):

        self.x_ = tf.placeholder(name='x', shape=[None, 256, 256, 3], dtype=tf.float32)
        self.y_ = tf.placeholder(name='y', shape=[None, 256, 256, 1], dtype=tf.float32)

    def get_logits_and_segmentation(self):

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(self.x_)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        logit = Conv2D(1, 1)(conv9)
        output = tf.nn.sigmoid(logit)

        return logit, output

    def get_loss(self, logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=logits))

    def get_train_step(self, loss_):
        return tf.train.AdamOptimizer(1e-4).minimize(loss_)



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

    if not os.path.exists(f'./results/{epoch}'):
        os.mkdir(f'./results/{epoch}')
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
        imsave(f'./results/{epoch}/{i}.jpg', image)

    dice = (2 * tp) / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)

    return dice, accuracy


def main():

    tf.compat.v2.random.set_seed(1020202)
    sess = tf.Session()

    model = Unet()

    logits, output = model.get_logits_and_segmentation()
    loss = model.get_loss(logits)
    train_step = model.get_train_step(loss)

    sess.run(tf.global_variables_initializer())


    train_inputs_, train_labels_ = read_image('./PH2Dataset/PH2Dataset/PH2Datasetimages')

    epochs = 1000
    for epoch in range(1, epochs):
        train(model, train_inputs_, train_labels_, train_step, sess, batch_size=3)
        dice, accuracy = evaluate(model, train_inputs_, train_labels_, output, sess, epoch)
        print(f'Epoch: {epoch}, Dice: {dice: .3f}, Accuracy: {accuracy: .3f}')


if __name__ == '__main__':
    main()
