# DEEP Q-Learning Project (DQN)
--
## Project overview
In order to practice more about DQN, we have been tasked to trains a reinforcement learning agent that plays an Atari game using DQN. The main goal is to explore how different hyperparameters affect training performance and compare policy types (MLP vs CNN) using ```Stable Baselines3``` and ```Gymnasium``. The agent is later evaluated by watching it play the game using a ε-greedy policy.

## Project structure
deep-q-with-atari/
    ├── train.py       # Train the agent
    ├── play.py        # Play with trained agent
    ├── dqn_model.zip  # Saved trained model
    ├── README.md      # Project overview
    ├── models/        # Trained models (CNN and MLP)
    └── logs/          # Training logs for TensorBoard

#### Environment used:
``` bash
env = gym.make("ALE/Breakout-v5", render_mode="human")
```
## Agent Behavior evaluation

| Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| lr=5e-4, gamma=0.90, batch=64, epsilon_start=1.0, epsilon_end=0.05 | Observ |
| lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | your observ |
| lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay=| your observ |



## Installation:
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
* 
```bash
    python train.py
```
* 
```bash
python play.py
```

##  Group Collaboration & Contributions

###  Overview
We worked in **pairs** to implement and compare two DQN agents (CNN and MLP policies) on the Atari game *Breakout*. Our focus was on training, evaluating, and selecting the best-performing model using Stable-Baselines3.

###  Individual Contributions

- **Christian & Willy**  
  - Developed `train.py` which sets up the environment, trains policies, and evaluates them.
  - Selected the game (Breakout-v5), set up project structure, and wrote model training logic.

- **Audry & Pascal**  
  - Focused on model import and evaluation logic in `play.py`, integrating the final trained agent and building a gameplay loop.
  - Helped with documentation and tuning hyperparameters during training and evaluation.

###  Team Collaboration
Although we worked in pairs, we had **frequent short sync-ups** to support each other and unblock issues. This included:
- Debugging training instability
- Sharing results and evaluation metrics
- Deciding which policy to save as final (`dqn_model.zip`)

###  Challenges Faced
- Handling unstable training with MLP due to poor performance on spatial data
- Navigating and wrapping the Atari environments correctly using Gym and ALE
- Rendering slowed down gameplay, making tuning and evaluation slightly harder
- Resolving environment compatibility between `gymnasium`, `stable_baselines3`, and `ale_py`

### Key Files

- `train.py`: Full training pipeline with CNN vs MLP comparison, model saving, and logging.
- `play.py`: Loads `dqn_model.zip` and runs a 1000-step simulation with performance output.

## Done by:
+ Christian M.
+ Audry A. Chivanga
+ Pascal M.
+ Willy K.
