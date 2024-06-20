import os

# =========================================================
# Dataset Hyper-parameter
DATASET_NAME = 'mnist'  # mnist

IMAGE_DIMENSION = 28
INPUT_SHAPE = (IMAGE_DIMENSION, IMAGE_DIMENSION, 1)

OUPUT_SHAPE = 62 # 


# =========================================================
# Model Hyper-parameter
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

LIST_NUMBER_FILTERS = [32, 64]
FILTER_SIZE = 5

MODEL_TYPE = "vanilla_conv" # ['vanilla_conv', 'resnet', 'xception']
PATH_GLOBAL_MODEL = os.path.join("models", "global_model_femnist_prune.h5")


# =========================================================
# Training Hyper-parameter
NUM_ROUNDS = 500
NUM_SELECTED_CLIENT = 50

LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 32

MAX_PRUNED_ROUND = 50
IS_STILL_PRUNE = True
PRUNE_PATIENCE = 0
MAX_PRUNE_PATIENCE = 3

STD_THRESHOLD_PRUNE = 2.2