# Sim2Real Reading List

## Evolutionary Approaches

Bongard J, Lipson H.
**Once more unto the breach: Co-evolving a robot and its simulator.**
In Proceedings of the Ninth International Conference on the Simulation and Synthesis of Living Systems (ALIFE9) 2004 (pp. 57-62).

- Introduces idea of parameter-estimation in simulation by cyclical revalutaion.

Zagal JC, Ruiz-del-Solar J, Vallejos P. **Back to reality: Crossing the reality gap in evolutionary robotics.** IFAC Proceedings Volumes. 2004 Jul 1;37(8):834-9.

- Introducing "BTR", a real-world-cyclic search algorithm for minimizing differences in fitness

Zagal JC, Ruiz-Del-Solar J. **Combining simulation and reality in evolutionary robotics.** Journal of Intelligent and Robotic Systems. 2007 Sep 1;50(1):19-39.

- Extension of "BTR" to simple dynamics, decreases rollout number

Koos S, Mouret JB, Doncieux S. **The transferability approach: Crossing the reality gap in evolutionary robotics.** IEEE Transactions on Evolutionary Computation. 2013 Feb;17(1):122-45.

- Brings transfer difference directly into the optimization function.

## Domain Randomization

Peng XB, Andrychowicz M, Zaremba W, Abbeel P. **Sim-to-real transfer of robotic control with dynamics randomization.** arXiv preprint arXiv:1710.06537. 2017 Oct 18.

- Introduces Domain Randomization over robot dynamics

Pinto L, Andrychowicz M, Welinder P, Zaremba W, Abbeel P. **symmetric actor critic for image-based robot learning.** arXiv preprint arXiv:1710.06542. 2017 Oct 18.

- Introduces Domain Randomization over images for simple visual servoing.

Tobin J, Zaremba W, Abbeel P. **Domain randomization and generative models for robotic grasping.** arXiv preprint arXiv:1710.06425. 2017 Oct 17.

- Introduces Domain Randomization over object shapes for grasping.

## Model-Building Approaches

Higgins I, Pal A, Rusu AA, Matthey L, Burgess CP, Pritzel A, Botvinick M, Blundell C, Lerchner A. **Darla: Improving zero-shot transfer in reinforcement learning.** arXiv preprint arXiv:1707.08475. 2017 Jul 26.

- Policy is learned on a b-VAE disentangled representation of the environment

Yang L, Liang X, Xing E. **Unsupervised Real-to-Virtual Domain Unification for End-to-End Highway Driving.** arXiv preprint arXiv:1801.03458. 2018 Jan 10.

- Learns transformation of real photos into photos that look like the ones that come out of the simulator. This way the driving policy only has to be learned on simulator-like images.

Zhang F, Leitner J, Ge Z, Milford M, Corke P. **Adversarial Discriminative Sim-to-real Transfer of Visuo-motor Policies.** May 2018

- Learns an image transformation that can adapt a simulator policy to a real policy via adversarial loss.

Golemo F, Ali Taiga A, Oudeyer PY, Courville A.
**Sim-to-Real Transfer with Neural-Augmented RobotSimulation.** Accepted at CoRL 2018, soon to be released

- If you take a real robot recording and roll out the simulation with the same timesteps, you can learn a dynamics model and use that to transform the simulator output.

## Misc Works

Rusu AA, Vecerik M, Rothörl T, Heess N, Pascanu R, Hadsell R. **Sim-to-real robot learning from pixels with progressive nets.** arXiv preprint arXiv:1610.04286. 2016 Oct 13.

- If you learn the policy in simulation, you can few-shot adapt to the real robot or other envs via progressively growing the network ( = adding new hidden nodes to all layers and freezing the existing weights, only training the newly added weights).

Marco A, Berkenkamp F, Hennig P, Schoellig AP, Krause A, Schaal S, Trimpe S. **Virtual vs. real: Trading off simulations and physical experiments in reinforcement learning with Bayesian optimization.** InRobotics and Automation (ICRA), 2017 IEEE International Conference on 2017 May 29 (pp. 1557-1563). IEEE.

- Uses Bayesian optimization on Gaussian Processes to find low-confidence points along policy trajectory and re-sample real robot there. Similar what Jean-Baptiste Mouret did 2015 with MAP-Elites.

Tai L, Paolo G, Liu M. **Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation.** InIntelligent Robots and Systems (IROS), 2017 IEEE/RSJ International Conference on 2017 Sep 24 (pp. 31-36). IEEE.

- If you train your policy on something abstract like range finder point cloud, the policy learned in simulation can accidentally work on the real robot without adjustment.

Bruce J, Sünderhauf N, Mirowski P, Hadsell R, Milford M. **One-shot reinforcement learning for robot navigation with interactive replay.** arXiv preprint arXiv:1711.10137. 2017 Nov 28.

- Better version of the previous paper: uses slight DomainRand. and prertrained feature detectors to learn policy in simulation based on automatically discretized environment.
