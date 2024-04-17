# =========================================================
# Dataset Hyper-parameter
DATASET_NAME = 'celeb'  # mnist or celeb

IMAGE_DIMENSION = 112
INPUT_SHAPE = (IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
OUPUT_SHAPE = 2

LABEL_NAME = {0: 'female', 1: 'male'}    # Gender classification 


# =========================================================
# Model Hyper-parameter
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

MODEL_TYPE = "vanilla_conv" # ['vanilla_conv', 'resnet', 'xception']

LIST_NUMBER_FILTERS = [32, 64, 96, 128]  # for 'vanilla_conv'
# LIST_NUMBER_FILTERS = [32, 32, 64, 128, 256]  # for 'resnet'
KERNEL_SIZE = 5
DROPOUT_RATE = 0.1


# =========================================================
# Training Hyper-parameter
NUM_ROUNDS = 200
SELECTED_PERCENT_CLIENT = 0.25

LOCAL_EPOCHS = 10
LOCAL_BATCH_SIZE = 32

MAX_PRUNED_ROUND = 40
IS_STILL_PRUNE = True
PRUNE_PATIENCE = 0
MAX_PRUNE_PATIENCE = 3

STD_THRESHOLD_PRUNE = 2.5