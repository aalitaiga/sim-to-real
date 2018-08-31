import poppy_helpers
import gym
from sklearn.externals import joblib
import numpy as np

EPISODES = 20
MODEL = "models/gp2_1000.pkl"

inf = joblib.load(MODEL)

dummy = np.expand_dims(np.array([-0.00166667, 0.00488889, 0.00811111, -0.00166667, 0.00166667,
                      0.00166667, 0., 0., 0., 0.,
                      0., 0., -0.00166667, 0.132, -0.11888889,
                      -0.03422222, -0.02122222, -0.08955556, 0., 0.,
                      0.00888002, 0., 0., 0.,0,0,0,0,0,0]),axis=0)

print (dummy.shape)

out = inf.predict(dummy)
print (out)
print (out.shape)

# env = gym.make("ErgoFight-Live-Shield-Move-HalfRand-NoComp-v0")
# obs = env.reset()
# print (obs)

# rewards = []
# while True:
