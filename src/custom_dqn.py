#? comentarle al tutor la decision de esto
#? Usar el repo que usaron ellos (mismos hyperparemtros, pero codigo desactualizado) o stable baseline (cambios en hyperparametros, pero mas modelos y mas sencillo de utilizar)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(NatureCNN):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)

    def forward(self, x):
        conv_outputs = []
        for layer in self.cnn:  # self.cnn is nn.Sequential of conv + relu
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                conv_outputs.append(x)  # store output after each ReLU
        x = x.flatten(start_dim=1)
        return x, conv_outputs
    


class MyPolicy(DQN.policy_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace default features_extractor
        self.q_net.features_extractor = CustomCNN(self.observation_space)

        Entrenar modelo en Kaggle y buscar como hacer para que no se acabe la sesion