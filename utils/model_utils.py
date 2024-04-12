import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
from keras.models import save_model, load_model
from keras import layers, models
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Activation, Conv2D, Dropout, BatchNormalization, Flatten, Input, MaxPooling2D, ReLU, DepthwiseConv2D, GlobalAveragePooling2D

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def FedAvg(global_model, list_client_model_weight, list_client_scales):
    """
    This function perform the FedAvg.
    * Parameters:
        global_model (keras model): the global model.
        list_client_model_weight (python list): each element indicate the weight of each client's model.
        list_client_scales (python list): each element indicate the scale of client.

    * Return:
        avg_weights (python list): weight of global model after FedAvg
    """

    num_selected_clients = len(list_client_model_weight)

    avg_weights = []
    for idx_layer in range(len(global_model.get_weights())):
        layer_weights = np.array([list_client_model_weight[j][idx_layer]  for j in range(num_selected_clients)])
        average_layer_weights = np.average(layer_weights, weights=list_client_scales, axis=0)
        avg_weights.append(average_layer_weights)

    return avg_weights


def Define_Simple_CNN_Model(input_shape, output_shape,\
                 list_number_kernel, kernel_size=3,\
                 max_pooling_step=2, dropout_rate=0.1, model_name=None):
    """
    This function create the simple CNN model. 
    """
    
    model = models.Sequential(name=model_name)
    model.add(Input(input_shape))

    for number_kernel in list_number_kernel:
        model.add(Conv2D(number_kernel, (kernel_size, kernel_size), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=max_pooling_step, strides=max_pooling_step))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation='softmax', name='classifier'))
    return model


def define_simple_model(name, input_shape):
    model = models.Sequential(name=name)

    model.add(layers.Dense(100, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(10, activation='softmax', name='classifier'))
    return model



def calculate_zero_weights_percentage(model):
    for layer in model.layers:
        if isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv2D): 
            weights = layer.get_weights()[0] 
            zero_weights_percentage = np.mean(weights == 0) * 100
            print(f"Layer {layer.name}: {zero_weights_percentage:.2f}% of weights are 0")



def get_flops_keras_model(model):
    '''
    Calculate FLOPS
    Parameters
    ----------
    model : tf.keras.Model
        Model for calculating FLOPS.

    Returns
    -------
    flops.total_float_ops : int
        Calculated FLOPS for the model
    '''
    
    batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops





# ================================ MOBILE NET =================================
def mobilnet_block (x, filters, strides):
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def define_mobilenet(input_shape, output_dims):

    input_layer = Input(shape = input_shape)
    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # main part of the model
    x = mobilnet_block(x, filters = 64, strides = 1)
    x = mobilnet_block(x, filters = 128, strides = 2)
    x = mobilnet_block(x, filters = 128, strides = 1)
    x = mobilnet_block(x, filters = 256, strides = 2)
    x = mobilnet_block(x, filters = 256, strides = 1)
    x = mobilnet_block(x, filters = 512, strides = 2)
    for _ in range (5):
        x = mobilnet_block(x, filters = 512, strides = 1)
        
    x = mobilnet_block(x, filters = 1024, strides = 2)
    x = mobilnet_block(x, filters = 1024, strides = 1)
    x = GlobalAveragePooling2D()(x)
    
    output_layer = Dense (units = output_dims, activation = 'softmax')(x)
    mobilenet = models.Model(inputs=input_layer, outputs=output_layer)
    return mobilenet