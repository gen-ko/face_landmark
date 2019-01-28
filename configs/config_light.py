from face_landmark.configs.config import Config

class ConfigLight(Config):
    NAME = '3D_FAN-Light-128'
    INPUT_SIZE = 128
    
    NUM_MODULES = 1

    EXPORT_DIR = '/barn1/yuan/saved_models/3d_fan_light_128_2019-01-04'
    LOG_FILENAME = '/barn1/yuan/saved_models/3d_fan_light_128_2019-01-04/training_log.txt'

    PLOT_DIR = '/barn1/yuan/saved_models/3d_fan_light_128_2019-01-04/figures'

    INPUT_SIZE = 128

    STEP_LIMIT = 0

    USE_IMAGE_GRADIENT = True

    CONV1_CHANNELS = 32
    CONV2_CHANNELS = 64
    CONV3_CHANNELS = 96
    CONV4_CHANNELS = 128
    INITIAL_CHECKPOINT_PATH = None