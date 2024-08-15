import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv3D, BatchNormalization, AveragePooling3D, LeakyReLU, ELU, Activation, \
    Input, average, concatenate
from keras.models import Model
from keras.utils import get_custom_objects
from keras.initializers import lecun_normal

from activations import swish, gelu

class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

def sub_model_3d_00a(in_layer, act_func, all_shape):
    out_layer = Conv3D(32, kernel_size=(3, 3, 5), padding='same', strides=(1, 1, 2), kernel_initializer=lecun_normal(), activation=act_func)(in_layer)
    out_layer = Dropout(0.5)(out_layer)
    out_layer = BatchNormalization()(out_layer)
    out_layer = AveragePooling3D(pool_size=(all_shape[1], all_shape[2], 1))(out_layer)
    out_layer = SqueezeLayer(axis=1)(out_layer)
    out_layer = SqueezeLayer(axis=1)(out_layer)
    out_layer = Conv1D(128, kernel_size=65, padding='same', strides=2, kernel_initializer=lecun_normal(), activation=act_func)(out_layer)
    return out_layer

def model_3d_00(X, Y, psa, workdir):
    num_classes = 2
    all_shape = X.shape
    input_shape = np.shape(X[0])
    input_shape_psa = np.shape(psa[0])
    
    get_custom_objects().update({'swish': Activation(swish)})
    get_custom_objects().update({'gelu': Activation(gelu)})
    get_custom_objects().update({'leaky-relu': Activation(LeakyReLU())})
    # get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(negative_slope=0.2))})
    act_func = ELU(alpha=0.8)

    # Input layers
    in_layer1 = Input(shape=input_shape)
    in_layer2 = Input(shape=input_shape)
    in_layer3 = Input(shape=input_shape)
    in_layer4 = Input(shape=input_shape)
    in_layer5 = Input(shape=input_shape)
    in_layer6 = Input(shape=input_shape)
    in_layer7 = Input(shape=input_shape_psa)
    in_layer = [in_layer1, in_layer2, in_layer3, in_layer4, in_layer5, in_layer6, in_layer7]
    
    # Mid layers
    out_layer1 = sub_model_3d_00a(in_layer1, act_func, all_shape)
    out_layer2 = sub_model_3d_00a(in_layer2, act_func, all_shape)
    out_layer3 = sub_model_3d_00a(in_layer3, act_func, all_shape)
    out_layer4 = sub_model_3d_00a(in_layer4, act_func, all_shape)
    out_layer5 = sub_model_3d_00a(in_layer5, act_func, all_shape)
    out_layer6 = sub_model_3d_00a(in_layer6, act_func, all_shape)
    out_layer = [out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6]
    
    # End layers
    end_layer = average(out_layer)
    end_layer = Conv1D(384, kernel_size=33, padding='same', strides=1, kernel_initializer=lecun_normal(), activation=act_func)(end_layer)
    end_layer = Dropout(0.5)(end_layer)
    end_layer = Flatten()(end_layer)
    
    # Add PSA
    in_layer7 = Dense(32, activation=act_func)(in_layer7)
    end_layer = concatenate([end_layer, in_layer7])
    
    # Finish
    end_layer = Dense(num_classes, activation='sigmoid')(end_layer)
    
    # Summarize model
    model = Model(inputs=in_layer, outputs=end_layer)
    # model.summary()
    
    # Plot model architecture
    # plot_model(model, show_shapes=True, to_file=workdir + 'Module3D03' + "-{}".format(int(time.time())) + '.png', expand_nested=True, show_layer_names=True)
    
    return model