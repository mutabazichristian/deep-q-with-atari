# DEEP Q-Learning Project (DQN)
----
## 1. Project overview
In order to practice more about DQN, we have been tasked to trains a reinforcement learning agent that plays an Atari game using DQN. The main goal is to explore how different hyper-parameters affect training performance and compare policy types (MLP vs CNN) using ```Stable Baselines3``` and ```Gymnasium``. The agent is later evaluated by watching it play the game using a ε-greedy policy.

## 2. Project structure
```bash
deep-q-with-atari/
    ├── train.py       # Train the agent
    ├── play.py        # Play with trained agent
    ├── dqn_model.zip  # Saved trained model
    ├── README.md      # Project overview
    ├── models/        # Trained models (CNN and MLP)
    └── logs/          # Training logs for TensorBoard
```
#### Environment used:
``` bash
env = gym.make("ALE/Breakout-v5", render_mode="human")
```
## Agent Behavior Evaluation (Tuning Table)

| Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| lr=5e-4, gamma=0.90, batch=64, epsilon_start=1.0, epsilon_end=0.05 | Observ |
| lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | your observ |
| lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay=| your observ |



## 3. Installation:
1. Creating virtual environment.
2. Clone repo from main-branch:
```bash
git clone https://github.com/mutabazichristian/deep-q-with-atari.git
```
3. Install project requirements:
```bash
pip install -r requirements.txt
```
alternatively to install requirements
```bash
pip install "gymnasium[atari]" ale-py stable-baselines3[extra]
```
4. After installing the requirements, Run this files:
```bash
python train.py
```
and then;
```bash
python play.py
```

## 4. Done by:
+ Christian M.
+ Willy K.
+ Audry A. Chivanga
+ Pascal M.
