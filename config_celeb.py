import os

# =========================================================
# Dataset Hyper-parameter
DATASET_NAME = 'celeb'  # mnist or celeb

IMAGE_DIMENSION = 84
INPUT_SHAPE = (IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
OUPUT_SHAPE = 2
LABEL_NAME = {0: 'female', 1: 'male'}    # Gender classification 


# =========================================================
# Model Hyper-parameter
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

MODEL_TYPE = "vanilla_conv" # ['vanilla_conv', 'resnet', 'xception']

LIST_NUMBER_FILTERS = [32, 32, 32, 32]  # for 'vanilla_conv'
KERNEL_SIZE = 5
DROPOUT_RATE = 0.1
PATH_GLOBAL_MODEL = os.path.join("models", "global_model_celeb_prune.h5")


# =========================================================
# Training Hyper-parameter
NUM_ROUNDS = 500
NUM_SELECTED_CLIENT = 50

LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 32

IS_STILL_PRUNE = True
MAX_PRUNED_ROUND = 50
PRUNE_PATIENCE = 0
MAX_PRUNE_PATIENCE = 3

STD_THRESHOLD_PRUNE = 2.0