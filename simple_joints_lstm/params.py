HIDDEN_NODES = 128
LSTM_LAYERS = 5
CUDA = True
EPOCHS = 1
DATASET_PATH = "/data/lisa/data/sim2real/mujoco_data4.h5"
MODEL_PATH = "./trained_models/simple_lstm4_v1_5l_128.pt"
MODEL_PATH_BEST = "./trained_models/simple_lstm4_v1_5l_128_best.pt"
TRAIN = True
CONTINUE = True
JOINT_LIMITS = [
    (-150, 150),
    (-125, 125),
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90)
]