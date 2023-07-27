# rl-bee-multimodal-sensing - Relevance of sensory modalities for spatial navigation
Reinforcement Learning Model Training and Prediction for Neuromatch group project  This repository contains Python scripts and utilities for training a custom Reinforcement Learning (RL) model using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. The trained model can be used for predicting actions in a custom Gym environment.

Agent in complex environment:
https://github.com/alTeska/rl-bee-multimodal-sensing/assets/17547288/c404de51-9ece-4c3b-beaa-3ce0a81f8b44


## Files

The repository contains the following files:

0. `bee.py`: Environment (gym) setup - agents action, rewards and space definition for the RL model.
1. `train_model.py`: The main script to train the RL model.
2. `render_model.py`: A script to generate and display a video of the RL model's predictions.
3. `config.yaml`: The configuration file for training the RL model.
4. `model.py`: A Python module containing utility functions for initializing and loading the RL model.
5. `utils.py`: A Python module containing utility functions for creating directories, saving the configuration, and more.
6. `gym_run.py`: Demonstration file for testing gym changes


## Usage

1. Configure the `config.yaml` file with the desired parameters for training the RL model.
2. Run the training script:

```python train_model.py --config_path config.yaml```

You can have a set of consecutive training rounds, make sure to specify each in a separate config.yaml and set proper alias setting. This is demonstrated in:

```bash run-multiround-training.sh```

3. After training, you can generate a video of the model's predictions using the `render_model.py` script or evaluate the model and save the episodes into a test log:

```python render_model.py --config_path configs/config.yaml```
```python render_model.py --config_path configs/test-config.yaml```


4. You can also use the convenience jupyter-notebooks, useful for working in google colab.



## Model specifics:
1. logging, early stopping, model managment
2. model re-training
3. walls/obstacles
1. Testing different architectures
2. Make goal smaller
3. Reward testing 
   1. Adding senses
   2. Time passed punishment
   3. wall hitting punishment
4. Increasing/gradually shrinking the goal size whilst training
5. multiple goals

Training example: 
https://github.com/alTeska/rl-bee-multimodal-sensing/assets/17547288/df3a1da3-7378-475b-8ffa-c9957835bb79



## References

- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- Gymnasium: https://github.com/openai/gym

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
