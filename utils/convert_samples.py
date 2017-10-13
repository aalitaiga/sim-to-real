from glob import glob
import skvideo.io
import numpy as np

dir = glob('/Tmp/alitaiga/sim-to-real/2017-05-25/*.npz')
print dir
for i, file_ in enumerate(sorted(dir)):
    array = np.load(file_)
    sample = array['generated']
    sample = (sample[:,0,:,:,:] + 1) * 127.5
    sample = sample.astype('uint8')
    sample = np.transpose(sample, (0, 2, 3, 1))
    skvideo.io.vwrite("/u/alitaiga/repositories/samples/epoch_{}.mp4".format(i), sample)
import ipdb; ipdb.set_trace()
