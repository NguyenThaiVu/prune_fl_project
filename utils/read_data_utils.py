import os
import numpy as np
import cv2
from tensorflow import keras
from keras.utils import to_categorical


def cut_redundant_background(image, left_margin = 5, right_margin = 20):
    image = image[left_margin:right_margin, left_margin:right_margin]
    return image


def Get_Image_and_Label_from_Client(client_dataset, dataset_name='mnist', target_size=None):
    """
    This function extract list_images and list_labels from client_dataset
    Parameters:
        client_dataset (python list): each element is a dictionary, which include:
                                    - If dataset_name is mnist: ['pixels', 'label']
                                    - If dataset_name is celeb: ['image', 'male']

        dataset_name (str): the name of dataset (mnist or celeb)

    Return:
        list_X (numpy array): list images in this client.
        list_y (numpy array): list label in this client.

    """

    list_X = []
    list_y = []

    for (idx_sample, sample) in enumerate(client_dataset):
        if dataset_name.lower() == 'mnist':
            X = sample['pixels'].numpy()
            y = sample['label'].numpy()
    
            original_shape = X.shape
            X = cut_redundant_background(X)
            X = cv2.resize(X, original_shape)

        if dataset_name.lower() == 'celeb': 
            X = sample['image'].numpy().astype(np.uint8)
            y = sample['male'].numpy().astype(np.int8)

        list_X.append(X)
        list_y.append(y)
    
    return (np.array(list_X), np.array(list_y))



def Create_Clients_Data(tff_dataset, dataset_name='mnist'):
    """
    This function read the tff_dataset and return the list of client data

    * Parameters:
        tff_dataset (tff dataset object): 
        dataset_name (str): the name of dataset - mnist (10 categories) or celeb (2 categories)

    * Returns:
        list_clients_data (python list): each element is a dictionary, containing ['client_name', 'list_X', 'list_y']
    """

    list_clients_data = []
    num_users = len(tff_dataset.client_ids)

    for idx_client in range(0, num_users):
        client_name = tff_dataset.client_ids[idx_client]

        client_dataset = list(tff_dataset.create_tf_dataset_for_client(tff_dataset.client_ids[idx_client]))
        (list_X, list_y) = Get_Image_and_Label_from_Client(client_dataset, dataset_name)
        
        if list_X.ndim == 3: list_X = np.expand_dims(list_X, axis=-1)  # Just in case the input image is gray scale (28,28)

        if dataset_name.lower() == 'mnist':  list_y = to_categorical(list_y, num_classes=62)
        if dataset_name.lower() == 'celeb':  list_y = to_categorical(list_y, num_classes=2)
        
        client_data = {'client_name': client_name, 'list_X': list_X, 'list_y': list_y}
        list_clients_data.append(client_data)

    return list_clients_data