# Game Based Deep Reinforcement Learning for Target Tracking

In pursuit-evasion games, we have an evader whose role is to escape from a pursuer. In this work we extend the visibility problem to a pursuit-evasion game where a player (the evader) tries to step out from the field of view of an antagonist player (a pursuer).

We start our approach in the discrete state space determining motion policies for both players using Dynamic Programing (DP). We call these policies **DMP** (for Deterministic Motion Policies). Then, we move progressively into the continous state space first by using Artificial Neural Networks (ANN) on a classification manner to learn the policies applied on the DP part which we call **SMP** (for Supervised Motion Policies). We finally refine what was learned with the ANN by using Deep Reinforcement Learning (DRL). In the image below we show this progress.

![Methodology](./Images/methodology%202.PNG)

On the DRL part, we have a total of three more policies. On **RMP** (for Reinforced Motion Policy) we use DRL with no prior knowledge, the neural network starts learning with random generated weights. **IRMP** stands for Initialized Reinforced Motion Policy, in it the weights on the SMP are used to initialize the weights on the DRL process. Finally, on the **MRMP**, which stands for Master Reinforced Motion Policy, we use the SMP as a master policy on the DRL provess.

We test our policies on several environments where both players are considered as point robots whose movement stays in a four-connectivity neighborhood with a fixed step. The pursuer has an unbounded field of view only blocked by obstacles. In the video below we compare every of the policies on a given environment. We evaluate every game on the amount of steps the evader (in red) needs to step out the pursuer's (in blue) field of view (in green). Note how at the end the pursuer is able to keep the evader always inside its field of view by using the MRMP policy.

![Video example](./Images/videoExample.mp4)

The codes used for this project can be found in the attached folders. They are organized as follows:

- Environment Discretization and Deterministic Motion Policies with Discrete Optimal Planing
- Data set Generation and Motion Policies in the Continuous State-Space with Artificial Neural Networks
- Motion Policies Improvement with Deep Reinforcement Learning
