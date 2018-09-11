import numpy as np
import torch

np.random.seed(2)

SAMPLES = 1000
STEPS_PER_SAMPLE = 100
SAMPLES_PER_BATCH = 10

space = np.linspace(-10,10,1000)
x = np.sin(space)
y = np.sin(space+1)

out_x = []
out_y = []
for _ in range(SAMPLES):
    start = np.random.randint(low=0,high=len(x)-STEPS_PER_SAMPLE)
    out_x.append(x[start:start+STEPS_PER_SAMPLE])
    out_y.append(y[start:start+STEPS_PER_SAMPLE])

out_x = np.array(out_x).reshape(-1,SAMPLES_PER_BATCH,STEPS_PER_SAMPLE)
out_y = np.array(out_y).reshape(-1,SAMPLES_PER_BATCH,STEPS_PER_SAMPLE)

torch.save(out_x, open('data_x.pt', 'wb'))
torch.save(out_y, open('data_y.pt', 'wb'))
torch.save(x, open('full_x.pt', 'wb'))
torch.save(y, open('full_y.pt', 'wb'))
