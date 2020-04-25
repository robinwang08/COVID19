import os
import logging

class Config(object):
    IMAGE_SIZE = 224

    TRIALS = 1
    BATCH_SIZE = 64

    EPOCHS = 50

    PATIENCE = 7
    SAMPLES_VALIDATION = 300
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    RAW_NRRD_ROOT = "/media/user1/data"
    INPUT_FORM = "t1"

    train = '/media/user1//train'
    validation = '/media/validation'
    test ='/media/user1/test'

    DATA = "/media/user1/data/"
    PREPROCESSED_DIR = os.path.join(DATA, "preprocessed")

    FEATURES_DIR = "./features"

    OUTPUT = "/media/user1/data/output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")
    DATASET_RECORDS = os.path.join(OUTPUT, "datasets")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 5
    SPLIT_TRAINING_INTO_VALIDATION = 0.1

    SEED = 'c07386a3-ce2e-4714-aa1b-3ba39836e82f'

config = Config()


