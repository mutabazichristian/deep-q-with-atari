# DEEP Q-Learning Project (DQN)
----

## 1. Project overview
In order to practice more about DQN, we have been tasked to train a reinforcement learning agent that plays an Atari game using DQN. The main goal is to explore how different hyperparameters affect training performance and compare policy types (MLP vs CNN) using `Stable Baselines3` and `Gymnasium`. The agent is later evaluated by watching it play the game using an ε-greedy policy.

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
```python
env = gym.make("ALE/Breakout-v5", render_mode="human")
```

## 3. Agent Behavior Evaluation (Tuning Table)

This section captures and compares the agent's performance using different hyperparameter settings during training. Each row summarizes how the agent behaved when evaluated with a trained model.

| Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| `lr=1e-5`, `gamma=0.90`, `batch_size=32`, `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay` | The agent was able to finish episodes and earned some rewards, indicating it learned the basic mechanics. However, the rewards were mostly `0`, with a maximum of `+4`, and performance was inconsistent. The agent sometimes survived longer, but survival duration did not correlate with high reward. |
| `lr=5e-4`, `gamma=0.90`, `batch_size=64`, `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay` | The agent finished episodes quickly and earned near-zero rewards. Its actions were inconsistent, suggesting undertraining or ineffective learning due to possibly insufficient timesteps. |
| `lr=2e-5`, `gamma=0.99`, `batch_size=32`, `epsilon_start=1.0`, `epsilon_end=0.05`, `exploration_fraction=0.3` (with `timesteps=150000`) | The agent demonstrated improved learning. It was able to reach rewards of up to `+5`, with more frequent scores of `+2` and `+1`, showing a noticeable improvement in stability and performance. While still inconsistent in places, this setup yielded the most balanced results. |

[![img-1.jpg](https://i.postimg.cc/TYyFfF5M/img-1.jpg)](https://postimg.cc/YhHbxXxb)
[![img-2.jpg](https://i.postimg.cc/mZpKxBDy/img-2.jpg)](https://postimg.cc/23v0hpn1)

## 4. Hyperparameter Tuning Discussion

From the hyperparameter experiments summarized above, several key insights were gathered:

- **Learning Rate (`lr`)**: A lower learning rate like `1e-5` allowed the agent to learn slowly but steadily. When we increased it to `5e-4`, learning became too unstable, possibly overshooting optimal Q-values. The best results were observed with `2e-5`, which struck a balance between learning speed and stability.

- **Discount Factor (`gamma`)**: Setting `gamma` to `0.99` helped the agent focus more on long-term rewards, which aligns well with the nature of the Breakout game where sequences of actions lead to better outcomes. In contrast, `gamma=0.90` made the agent focus too much on immediate rewards, which likely stunted its performance.

- **Batch Size**: While batch size `64` is typically more stable in many settings, it didn’t show improvements here—possibly because of the small replay buffer size or early training instability. Batch size `32` yielded more consistent learning when combined with a lower learning rate.

- **Exploration Strategy**: The ε-greedy policy with gradual decay (`epsilon_start=1.0` to `epsilon_end=0.05`) allowed the agent to explore broadly in the early stages and then exploit its knowledge later. The best-performing setup used `exploration_fraction=0.3`, which allowed a smoother transition from exploration to exploitation.

- **Training Duration (Timesteps)**: One of the most important discoveries was that the **number of timesteps heavily influences agent performance**. The more timesteps we allocated for training (e.g., `150,000+`), the better the agent performed. This suggests that for a complex task like Breakout, the agent requires a long training horizon to sufficiently explore the environment and converge toward optimal actions.

In conclusion, tuning hyperparameters in DQN requires careful balancing of learning stability, exploration, and discounting future rewards. It was clear from our experiments that **training longer with well-chosen hyperparameters is key to better agent performance**.

## 5. Installation:

1. Create a virtual environment (optional but recommended).
2. Clone the repository from the main branch:
```bash
git clone https://github.com/mutabazichristian/deep-q-with-atari.git
```
3. Install project requirements:
```bash
pip install -r requirements.txt
```
Alternatively, install requirements directly with:
```bash
pip install "gymnasium[atari]" ale-py stable-baselines3[extra]
```

4. After installing the requirements, run the following:
```bash
python train.py
```
Then to test the trained model:
```bash
python play.py
```

## 6. Demo Video

[![Watch the Demo](https://img.youtube.com/vi/214xwcZ8j5g/0.jpg)](https://www.youtube.com/watch?feature=shared&v=214xwcZ8j5g)

Click the image above or watch directly [on YouTube](https://www.youtube.com/watch?feature=shared&v=214xwcZ8j5g).


## 7. Group Collaboration & Contributions

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



## 8. Done by:
+ Christian M.
+ Willy K.
+ Audry A. Chivanga
+ Pascal M.
