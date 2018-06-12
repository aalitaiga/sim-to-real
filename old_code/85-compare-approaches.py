import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

files = {
    "sim": "HalfCheetah2-v0-run41", #
    "simplus-v2": "HalfCheetah2Plus-v0-run38",
    "simplus-v3": "HalfCheetah2Plus-v0-run39",
    "simplus-v4": "HalfCheetah2Plus-v0-run40",
    "simplus-v2-r": "HalfCheetah2Plus-v0-run51",
    "simplus-v3-r": "HalfCheetah2Plus-v0-run52",
    "simplus-v4-r": "HalfCheetah2Plus-v0-run53",
    "real": "HalfCheetah2-v0-run34-real" #
}

LOGS_DIR = "rl-logs"

data = {}
means = {}
stds = {}

x = np.linspace(-700, 700, 1400)


for key,val in sorted(files.items()):
    data[key] = np.loadtxt("./{}/{}/eval.log".format(LOGS_DIR, val))
    mean,std=norm.fit(data[key])
    means[key] = mean
    stds[key] = std
    print ("{}\t{}\t{}".format(key, mean, std))
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, label=key)


plt.legend()

plt.xlim((-700, 700))
plt.ylim((0,0.05))
# plt.hist(data["sim"], bins=10, normed=True)



plt.show()
