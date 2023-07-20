from test import test
from train import train

# common variables
gymName = "BeeWorld"
base_path = "drive/MyDrive/neuromatch/"
model_algo = "TD3"

# train the model
train(
    gymName=gymName,
    base_path=base_path,
    model_algo=model_algo,
    timesteps=10000,
    iters_max=10,
)

# test the model
test(gymName=gymName, base_path=base_path, model_algo=model_algo, range_max=1000)
