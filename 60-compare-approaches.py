import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

files = {
    "sim": "Reacher2-v1-run2",
    "simplus-v1": "Reacher2Plus-v1-run5",
    "simplus-v2": "Reacher2Plus-v1-run8",
    "simplus-v3": "Reacher2PlusBig-v1-run11",
    "real": "Reacher2-v1-run12"
}

LOGS_DIR = "rl-logs"

data = {}
means = {}
stds = {}

RANGE = 1100

x = np.linspace(-RANGE, 0, RANGE)


for key,val in files.items():
    data[key] = np.loadtxt("./{}/{}/eval.log".format(LOGS_DIR, val))
    mean,std=norm.fit(data[key])
    means[key] = mean
    stds[key] = std
    print (key, mean, std)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, label=key)


plt.legend()


# plt.hist(data["sim"], bins=10, normed=True)



plt.show()
