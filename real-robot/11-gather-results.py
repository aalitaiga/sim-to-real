import numpy as np

sim_file = [
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0-180530232307-rewards-25-180601104449.npz",
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0-180530210954-rewards-20-180614215122.npz",
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0-180529135104-rewards-25-180530113054.npz",
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-v0-180530211020-rewards-20-180614165501.npz"
]
print ("sim")
res = []
for f in sim_file:
    data = np.load(f)["rewards"]
    res.append(np.mean(data))
print (res)



real_file = [
    "../results/ErgoFight-Live-Shield-Move-HalfRand-v0-180508145449-rewards-25-180510233930.npz",
    "../results/ErgoFight-Live-Shield-Move-HalfRand-v0-180509120904.pt-rewards-25-180510-232208.npz"
]
print ("real")
res = []
for f in real_file:
    data = np.load(f)["rewards"]
    res.append(np.mean(data))
print(res)



simplus_file = [
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Half-v0-180614142942-rewards-20-180614223401.npz",
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Half-v0-180614160111-rewards-20-180614204335.npz"
]
res = []
print ("simplus")
for f in simplus_file:
    data = np.load(f)["rewards"]
    res.append(np.mean(data))
print(res)



nosim_file = [
    "../results/ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-NoSim-v0-180614184611-rewards-20-180614201131.npz"
]
res = []
print ("nosim")
for f in nosim_file:
    data = np.load(f)["rewards"]
    res.append(np.mean(data))
print(res)


