import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate


class concat_wrapper:
    def __init__(self, tensor1_name, tensor2_name, op_name='concat', axis=3):
        self.name = op_name
        self.axis = axis
        self.input_tensor_names = [tensor1_name, tensor2_name]

    def __call__(self, a, b):
        return concatenate([a, b], axis=self.axis)


tf.keras.backend.set_floatx('float32')

conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv1_1')
conv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv1_2')
pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')
conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv2_1')
conv2_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv2_2')
pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')
conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv3_1')
conv3_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv3_2')
pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')
conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv4_1')
conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv4_2')
drop4 = Dropout(0.5, name='drop4')
pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')
conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv5_1')
conv5_2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='down_conv5_2')
drop5 = Dropout(0.5, name='drop5')
upsampling6 = UpSampling2D(size=(2, 2), name='upsampling6')
upconv6_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv6_1')
merge6 = concat_wrapper('drop4', 'up_conv6_1')
upconv6_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv6_2')
upconv6_3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv6_3')
upsampling7 = UpSampling2D(size=(2, 2), name='upsampling7')
upconv7_1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv7_1')
merge7 = concat_wrapper('down_conv3_2', 'up_conv7_1')
upconv7_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv7_2')
upconv7_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv7_3')
upsampling8 = UpSampling2D(size=(2, 2), name='upsampling8')
upconv8_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv8_1')
merge8 = concat_wrapper('down_conv2_2', 'up_conv8_1')
upconv8_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv8_2')
upconv8_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv8_3')
upsampling9 = UpSampling2D(size=(2, 2), name='upsampling9')
upconv9_1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv9_1')
merge9 = concat_wrapper('up_conv9_1', 'down_conv1_2')
upconv9_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv9_2')
upconv9_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv9_3')
upconv9_4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up_conv9_4')
output = Conv2D(1, 1, activation='sigmoid', name='logits')


layers = [
conv1_1,
conv1_2,
pool1,
conv2_1,
conv2_2,
pool2,
conv3_1,
conv3_2,
pool3,
conv4_1,
conv4_2,
drop4,
pool4,
conv5_1,
conv5_2,
drop5,
upsampling6,
upconv6_1,
merge6,
upconv6_2,
upconv6_3,
upsampling7,
upconv7_1,
merge7,
upconv7_2,
upconv7_3,
upsampling8,
upconv8_1,
merge8,
upconv8_2,
upconv8_3,
upsampling9,
upconv9_1,
merge9,
upconv9_2,
upconv9_3,
upconv9_4,
output
]

prune_layers = list()
for layer in layers:
    if 'conv' in layer.name:
        prune_layers.append(layer.name)