from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Lambda, MaxPooling2D, Input
from keras.layers.merge import concatenate
import os
import keras
from core.activation import Mish, Mish6
from keras.models import Model
from keras.utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
darknet tools
'''


def _darknet_conv(input, conv):
    x = input
    if conv['stride'] == 1:
        padding = "same"
    else:
        padding = "valid"

    if conv['stride'] > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # unlike tensorflow darknet prefer left and top paddings
    x = Conv2D(conv['filter'],
               conv['kernel'],
               strides=conv['stride'],
               padding=padding,
               # unlike tensorflow darknet prefer left and top paddings
               name='conv_' + str(conv['layer_idx']),
               use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']:
        x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['activation'] == "leaky":
        x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
    elif conv['activation'] == "mish":
        x = Mish(name='mish_' + str(conv['layer_idx']))(x)
    else:
        pass
    return x


def _darknet_conv_block(input, convs):
    x = input
    for conv in convs:
        x = _darknet_conv(input=x, conv=conv)
    return x


def slice_channel(x, index):
    return x[:, :, :, 0: index]


def _csp_darknet_tiny_res_block(input, convs):
    x = input
    x = _darknet_conv(x, convs[0])
    channel = keras.backend.int_shape(x)[-1]
    x1 = Lambda(slice_channel, arguments={'index': int(channel/2)})(x)
    x2 = _darknet_conv(x1, convs[1])
    x3 = _darknet_conv(x2, convs[2])
    route = concatenate([x2, x3])
    x4 = _darknet_conv(route, convs[3])
    y = concatenate([x, x4])
    return y, x4


def yolov4_tiny_backbone(input_image):

    # stage1:layer:0-1(conv)
    x = _darknet_conv_block(input=input_image, convs=[
        {'filter': 32, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 0},
        {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 1}])

    # stage2:layer:2-6(conv, maxpool)
    x, _ = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 2},
        {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 3},
        {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 4},
        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 5}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # stage3:layer:7-11(conv, maxpool)
    x, _ = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 7},
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 8},
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 9},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 10}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # stage3:layer:12-16(conv, maxpool)
    x, route = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 12},
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 13},
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 14},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 15}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    return route, x


# input_image = Input(shape=(None, None, 3))
# x, route = yolov4_tiny_backbone(input_image)
# model = Model(inputs=input_image, outputs=x, name='darknet')
# model.summary()
# plot_model(model, "yolov4_tiny_backbone_model.png", show_shapes=True, show_layer_names=True)



