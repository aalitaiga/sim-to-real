import numpy as np

diffs = np.load("../results/diff-data-afterPaper.npz")

d_sim = diffs["diff_gt_sim"] / 299
d_nosim = diffs["diff_gt_nosim"] / 299
d_simplus = diffs["diff_gt_simplus"] / 299
d_gp = diffs["diff_gt_gp"] / 299

print("Sim Mean:", np.mean(d_sim), np.percentile(d_sim, 25), np.percentile(d_sim, 75))
print("Nosim Mean:", np.mean(d_nosim), np.percentile(d_nosim, 25), np.percentile(d_nosim, 75))
print("Simplus Mean:", np.mean(d_simplus), np.percentile(d_simplus, 25), np.percentile(d_simplus, 75))
print("GP Mean:", np.mean(d_gp), np.percentile(d_gp, 25), np.percentile(d_gp, 75))
