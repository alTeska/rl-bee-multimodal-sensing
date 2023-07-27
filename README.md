# Relevance of sensory modalities for spatial navigation [rl-bee-multimodal-sensing]
Reinforcement Learning Model Training and Prediction for Neuromatch group project  This repository contains Python scripts and utilities for training a custom Reinforcement Learning (RL) model using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. The trained model can be used for predicting actions in a custom Gym environment.

The problem is goal-oriented spatial navigation with multi-modal sensory inputs. Two sensory inputs, that can be knocked-out (added noise) in order to investigate the importance of the senses: How do different modalities contribute to 
goal-oriented spatial navigation?


Agent in complex environment:

https://github.com/alTeska/rl-bee-multimodal-sensing/assets/17547288/c404de51-9ece-4c3b-beaa-3ce0a81f8b44


## Model specifics:

Environment Design:
- 2D env (Gymnasium) with continuous action space
- action space: 
  - acceleration 
  - angular acceleration 
- rewards: 
  - 1000 points for finding the goal
  - negative reward for energy expenditure
  - negative reward for distance from the goal


Observation space (6 inputs): 
- sensory inputs:
  - vision  - one hot encoding (0/1) modeled with limited angle cone, (wall vision included - based on ray casting)
  - smell - modeled as euclidean distance from the goal
- time elapsed 
- velocity, angular velocity  
- at each time step agent receives `(5, 6)` matrix containing  a current time frame and a memory of previous 4 frames


Training specific:
1. multiple-round training integration with model management and custom logging
2. early stopping if no training progress
3. goal size can be changed, custom walls/obstacles

For our sample training we gradually decreased the size of the goal and added the walls.


Training example: 


https://github.com/alTeska/rl-bee-multimodal-sensing/assets/17547288/df3a1da3-7378-475b-8ffa-c9957835bb79


## Files

The repository contains the following files:

0. `bee.py`: Environment (gym) setup - agents action, rewards and space definition for the RL model.
1. `model.py`: A Python module containing utility functions for initializing and loading the RL model.
2. `utils.py`: A Python module containing utility functions for creating directories, saving the configuration, and more.
3 `train_model.py`: The main script to train the RL model.
4. `evaluate_model.py`: evaluation script for sensory input knockout
5. `render_model.py`: A script to generate and display a video of the RL model's predictions.
6. `gym_run.py`: Demonstration file for testing gym changes


#### Folders:
1. `configs` - contain all of the sample testing and training configurations
2. `notebooks` - will configs used for training and analysis visualization 
3. `figures`
   

## Usage

1. Configure the `config.yaml` file with the desired parameters for training the RL model.
2. Run the training script:

   ```python train_model.py --config_path config.yaml```

   You can have a set of consecutive training rounds, make sure to specify each in a separate config.yaml and set proper alias setting. This is demonstrated in:

   ```bash run-multiround-training.sh```

3. Sensory knock-out experiments can be performed and saved:
   
   ```python evaluate_model.py --config_path configs/test-config.yaml```

4. After training, you can generate a video of the model's predictions using the `render_model.py` script or evaluate the model and save the episodes into a test log:
   ```python render_model.py --config_path configs/config.yaml```
   
5. Analysis of the models performance
   1. `sense-knockout-analysis.ipynb` - plot failure rate and episodes lengths under each sensory condition
   2. `training-visualization.ipynb` - plot multi-round metrics 
   
6.  You can also use the convenience jupyter-notebooks, useful for working in google colab.


## References

- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- Gymnasium: https://github.com/openai/gym

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
