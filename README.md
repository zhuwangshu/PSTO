# PSTO: Learning Energy-Efficient Locomotion for Quadruped Robots
This project is based on pybullet-gym: https://github.com/benelot/pybullet-gym.

You should first download the pybullet-gym environment and put the learning codes into the pybullet-gym folder.

Run train.py to train the policy.

Run seeTrainedResult.py to check the learned result.

The Arduino code TrajectoryControl.ino is used to control the Ant robot, a custom built quadruped robot with eight servomotors by the learned policy. Each leg of the Ant consists of two carbon fiber tubes and two servo motors controlled by an Arduino nano with an Adafruit 16-Channel PWM/Servo Driver. Zhu, J., Li, S., Wang, Z., \& Rosendo, A. (2019, July). Bayesian optimization of a quadruped robot during 3-dimensional locomotion. In Conference on Biomimetic and Biohybrid Systems (pp. 295-306). Springer, Cham.

Additionally, main.py and Arduino code BT2servo.ino can control the robot by the learned policy.
