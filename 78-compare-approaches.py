import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

files = {
    "sim": "Pusher2-v0-run22",
    # "simplus-v1": "Pusher2Plus-v0-run33",
    "simplus-v2": "Pusher2Plus-v0-run26",
    "simplus-v3": "Pusher2Plus-v0-run27",
    "simplus-v4": "Pusher2Plus-v0-run28",

    "simplus-v1-r": "Pusher2Plus-v0-run47",
    "simplus-v2-r": "Pusher2Plus-v0-run48",
    "simplus-v3-r": "Pusher2Plus-v0-run49",
    "simplus-v4-r": "Pusher2Plus-v0-run50",
    "real": "Pusher2-v0-run24"
}

LOGS_DIR = "rl-logs"

data = {}
means = {}
stds = {}

RANGE = 2600

x = np.linspace(-RANGE, 0, RANGE)


for key,val in sorted(files.items()):
    data[key] = np.loadtxt("./{}/{}/eval.log".format(LOGS_DIR, val))
    mean,std=norm.fit(data[key])
    means[key] = mean
    stds[key] = std
    print ("{}\t{}\t{}".format(key, mean, std))
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, label=key)


plt.legend()

plt.xlim((-2700, -1100))
plt.ylim((0,0.003))
# plt.hist(data["sim"], bins=10, normed=True)



plt.show()
