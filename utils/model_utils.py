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
from keras.utils import plot_model
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



def Define_Simple_CNN_Model(input_shape, output_shape, list_number_filters, max_pooling_step=2, model_name=None):
    """
    This function create the simple CNN model. 
    """

    kernel_size = 5
    model = models.Sequential(name=model_name)
    model.add(Input(input_shape))

    for (idx_filter, number_filter) in enumerate(list_number_filters):
        model.add(Conv2D(number_filter, (kernel_size, kernel_size), name=f'prunable_conv_{idx_filter}'))
        # model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=max_pooling_step, strides=max_pooling_step))

    # model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D())
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



def get_flops_keras_model(model):
    '''
    Calculate FLOPS
    Parameters
    ----------
    model (tf.keras.Model): Model for calculating FLOPS.

    Returns
    -------
    flops.total_float_ops (int): Calculated FLOPS for the model
    '''
    
    batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops




# ===================================================== RESNET =====================================================

def residual_block(input_tensor, num_filters_1, num_filters_2, kernel_size=3, strides=2, idx_residual_block=None):
    """
    This function define the residual block, which is the architecture using in the ResNet
    """

    shortcut = input_tensor
    
    # First block
    x = layers.Conv2D(num_filters_1, kernel_size, strides=strides, padding='same', name=f'prunable_conv_{idx_residual_block+1}')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second block
    num_filters_2 = input_tensor.shape[-1]
    x = layers.Conv2D(num_filters_2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjusting shortcut dimension if necessary
    if strides != 1 or shortcut.shape[-1] != num_filters_2:
        shortcut = layers.Conv2D(num_filters_2, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Adding shortcut to main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def Define_ResNet_Model(input_shape, output_shape, list_number_filters=[8, 8, 16, 32, 64], max_pooling_step=2, model_name=None):
    """
    This function create the simple Residual Network model. 
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolutional Layer
    x = layers.Conv2D(list_number_filters[0], kernel_size=3, strides=2, padding='same', name=f'prunable_conv_0')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=max_pooling_step, strides=max_pooling_step, padding='same')(x)
    
    # Residual Blocks
    for (idx_residual_block, number_filters) in enumerate(list_number_filters[1:]):
        x = residual_block(x, num_filters_1=number_filters, num_filters_2=number_filters, strides=(2, 2), idx_residual_block=idx_residual_block)
    
    # Final Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(output_shape, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model



def Get_Model(model_type, input_shape, output_shape, list_number_filters, max_pooling_step=2, model_name=None):
    """
    This function take the model_type and return the corresponding model

    * Parameters:
        model_type (str): indicate the define model type, including ['vanilla_conv', 'resnet', 'xception'] 

    * Return:
        model (keras.Model): the selected model.
    """

    if model_type == 'vanilla_conv':
        model = Define_Simple_CNN_Model(input_shape, output_shape, list_number_filters, max_pooling_step, model_name)
    elif model_type == 'resnet':
        model = Define_ResNet_Model(input_shape, output_shape, list_number_filters, max_pooling_step, model_name)
    elif model_type == 'xception':
        model = Define_ResNet_Model(input_shape, output_shape, list_number_filters, max_pooling_step, model_name)
    else:
        model = None

    return model    