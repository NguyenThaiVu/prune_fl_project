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
DROPOUT_RATE = 0.1

MODEL_TYPE = "vanilla_conv" # ['vanilla_conv', 'resnet', 'xception']


# =========================================================
# Training Hyper-parameter
NUM_ROUNDS = 1_000
MAX_SELECTED_PERCENT_CLIENT = 0.1

LOCAL_EPOCHS = 10
LOCAL_BATCH_SIZE = 32

MAX_PRUNED_ROUND = 50
IS_STILL_PRUNE = True
PRUNE_PATIENCE = 0
MAX_PRUNE_PATIENCE = 3

STD_THRESHOLD_PRUNE = 2.5