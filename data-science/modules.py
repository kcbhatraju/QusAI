from keras.layers import Activation, AveragePooling3D, Conv3D, MaxPooling3D, add, concatenate
from keras.initializers import lecun_normal
from keras.regularizers import l2

# function for creating an identity or projection residual module
def residual_module_3d_01(layer_in, n_filters, kernel_siz, activation_func, strd):
    merge_input = layer_in
    
    # check if the number of filters needs to be increased, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv3D(n_filters, (1, 1, 1), padding='same', strides=strd, kernel_regularizer=l2(0.01), kernel_initializer='glorot_normal')(layer_in)
   
    conv1 = Conv3D(n_filters, kernel_siz, padding='same', strides=strd, activation=activation_func, kernel_regularizer=l2(0.01), kernel_initializer='glorot_normal')(layer_in)
    conv2 = Conv3D(n_filters, kernel_siz, padding='same', kernel_regularizer=l2(0.01), kernel_initializer='glorot_normal')(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation(activation_func)(layer_out)
    return layer_out

def residual_module_3d_02(layer_in, n_filters, kernel_siz, activation_func, strd):
    merge_input = layer_in
    
    # check if the number of filters needs to be increased, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv3D(n_filters, (1, 1, 1), padding='same', kernel_initializer=lecun_normal(seed=None))(layer_in)
   
    conv1 = Conv3D(n_filters, kernel_siz, padding='same', activation=activation_func, kernel_initializer=lecun_normal(seed=None))(layer_in)
    conv2 = Conv3D(n_filters, kernel_siz, padding='same', kernel_initializer=lecun_normal(seed=None))(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation(activation_func)(layer_out)
    return layer_out

def inception_module_3d_org(layer_in, act_func, n_filters, n_strides):
    merge_input = layer_in
        
    tower_0 = Conv3D(n_filters, (1, 1, 1), padding='same', activation=act_func, strides=n_strides)(merge_input)
        
    tower_1 = Conv3D(n_filters, (1, 1, 1), padding='same', activation=act_func)(merge_input)
    tower_1 = Conv3D(n_filters, (3, 3, 7), padding='same', activation=act_func, strides=n_strides)(merge_input)
    
    tower_2 = Conv3D(n_filters, (1, 1, 1), padding='same', activation=act_func)(merge_input)
    tower_2 = Conv3D(n_filters, (3, 3, 5), padding='same', activation=act_func, strides=n_strides)(merge_input)
        
    tower_3 = AveragePooling3D((1, 1, 2), padding='same')(merge_input)
    tower_3 = Conv3D(n_filters, (1, 1, 1), padding='same', activation=act_func)(tower_3)

    layer_out = concatenate([tower_0, tower_1, tower_2, tower_3], axis=4)
    
    return layer_out

def inception_module_3d_00(layer_in, n_kernel, n_strides):
    merge_input = layer_in
        
    tower_3 = MaxPooling3D(n_kernel, padding='same', strides=n_strides)(merge_input)
    tower_4 = AveragePooling3D(n_kernel, padding='same', strides=n_strides)(merge_input)

    layer_out = concatenate([tower_4, tower_3], axis=4)
    
    return layer_out

def inception_module_3d_00a(layer_in, n_kernel, n_strides):
    merge_input = layer_in
        
    tower_1 = MaxPooling3D(pool_size=n_kernel)(merge_input)
    tower_2 = AveragePooling3D(pool_size=n_kernel)(merge_input)
    tower_3 = AveragePooling3D(pool_size=(1, 1, 3), padding='same', strides=n_strides)(merge_input)
    tower_4 = AveragePooling3D(pool_size=(3, 3, 5), padding='same', strides=n_strides)(merge_input)
    tower_5 = AveragePooling3D(pool_size=(5, 5, 7), padding='same', strides=n_strides)(merge_input)
    
    layer_out = concatenate([tower_1, tower_2], axis=4)
    
    return layer_out

def inception_module_3d_01(layer_in, n_filters, act_func):
    merge_input = layer_in
    
    tower_00 = Conv3D(n_filters, (1, 1, 1), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
        
    tower_0 = Conv3D(n_filters, (5, 5, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
    tower_1 = Conv3D(n_filters, (3, 3, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
    tower_2 = Conv3D(n_filters, (2, 2, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
    tower_3 = Conv3D(n_filters, (5, 5, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
    tower_4 = Conv3D(n_filters, (3, 3, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
    tower_5 = Conv3D(n_filters, (2, 2, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer=lecun_normal(seed=10))(merge_input)
            
    tower_6 = AveragePooling3D((1, 1, 4), padding='same')(merge_input)
    tower_6 = Conv3D(n_filters, (9, 9, 9), padding='same', activation=act_func, kernel_initializer=lecun_normal(seed=10))(tower_6)
    
    tower_7 = MaxPooling3D((1, 1, 4), padding='same')(merge_input)
    tower_7 = Conv3D(n_filters, (9, 9, 9), padding='same', activation=act_func, kernel_initializer=lecun_normal(seed=10))(tower_7)

    layer_out = concatenate([tower_2, tower_1, tower_3], axis=4)
    
    return layer_out

def inception_module_3d_02(layer_in, n_filters, act_func):
    merge_input = layer_in
    
    tower_0 = Conv3D(n_filters, (5, 5, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_1 = Conv3D(n_filters, (3, 3, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_2 = Conv3D(n_filters, (2, 2, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_4 = Conv3D(n_filters, (3, 3, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_5 = Conv3D(n_filters, (2, 2, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
            
    tower_6 = AveragePooling3D((1, 1, 4), padding='same')(merge_input)
    tower_6 = Conv3D(n_filters, (7, 7, 7), padding='same', activation=act_func, kernel_initializer='lecun_normal')(tower_6)
    
    tower_7 = MaxPooling3D((1, 1, 4), padding='same')(merge_input)
    tower_7 = Conv3D(n_filters, (7, 7, 7), padding='same', activation=act_func, kernel_initializer='lecun_normal')(tower_7)

    layer_out = concatenate([tower_0, tower_1, tower_2, tower_4], axis=4)
    
    return layer_out

def inception_module_3d_02b(layer_in, n_filters, act_func):
    merge_input = layer_in
    
    tower_0 = Conv3D(n_filters, (3, 3, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_1 = Conv3D(n_filters, (3, 3, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_2 = Conv3D(n_filters, (2, 2, 35), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_4 = Conv3D(n_filters, (3, 3, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
    tower_5 = Conv3D(n_filters, (2, 2, 65), padding='same', activation=act_func, strides=(1, 1, 4), kernel_initializer='lecun_normal', kernel_regularizer=l2(0.008))(merge_input)
            
    tower_6 = AveragePooling3D((1, 1, 4), padding='same')(merge_input)
    tower_6 = Conv3D(n_filters, (7, 7, 7), padding='same', activation=act_func, kernel_initializer='lecun_normal')(tower_6)
    
    tower_7 = MaxPooling3D((1, 1, 4), padding='same')(merge_input)
    tower_7 = Conv3D(n_filters, (7, 7, 7), padding='same', activation=act_func, kernel_initializer='lecun_normal')(tower_7)

    layer_out = concatenate([tower_0, tower_1, tower_2, tower_4], axis=4)
    
    return layer_out