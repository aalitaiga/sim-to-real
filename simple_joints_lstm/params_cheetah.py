HIDDEN_NODES = 256
LSTM_LAYERS = 5
EXPERIMENT = 4
# CUDA = False
CUDA = True
EPOCHS = 1
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH_REL = "/lindata/sim2real/"
MODEL = "cheetah"
RELEVANT_COLUMNS = list(range(3,9)) + list(range(12,18))
DATASET_PATH = DATASET_PATH_REL + "mujoco_data1_{}.h5".format(MODEL)
MODEL_PATH = "./trained_models/simple_lstm_{}{}_v1_{}l_{}.pt".format(
    MODEL,
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/simple_lstm_{}{}_v1_{}l_{}_best.pt".format(
    MODEL,
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
TRAIN = True
CONTINUE = True
