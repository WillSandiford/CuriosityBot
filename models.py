import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cpu')

# Extracts features relevant for traversing the environment
class FeatureEncoder(nn.Module):
    def __init__(self, input_shape):
        super(FeatureEncoder, self).__init__()
        
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(1, -1)
        return x

    def feature_size(self):
        return self.forward(torch.zeros(1, *self.input_shape)).size(1)


# Predicts which action was taken given the 
# encoded current state and encoded next state
class InverseModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(InverseModel, self).__init__()

        self.fc1 = nn.Linear(2 * input_shape, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, num_actions)
        
    def forward(self, state, next_state):
        x = torch.cat((state.detach(), next_state.detach()), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# Predicts what the encoded next state will be given the 
# encoded current state and one-hot encoded actions
class ForwardModel(nn.Module):
    def __init__(self, input_size, action_size):
        super(ForwardModel, self).__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(input_size + action_size, 3000)
        self.fc2 = nn.Linear(3000, 2000)
        self.fc3 = nn.Linear(2000, input_size)
        
    def forward(self, state, action):
        action = action.unsqueeze(0)
        x = torch.cat((state.detach(), action.detach()), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def one_hot(self, action):
        action_one_hot = torch.FloatTensor(torch.zeros(self.action_size))
        action_one_hot[action] = 1
        return action_one_hot


# Combines the encoder, forward model and inverse model to produce a
# curiosity reward signal and error to train all three models
class ICM(nn.Module):
    def __init__(self, num_inputs, num_actions, encoder):
        super(ICM, self).__init__()
        self.encoder = encoder
        self.forward_model = ForwardModel(self.encoder.feature_size(), num_actions)
        self.inverse_model = InverseModel(self.encoder.feature_size(), num_actions)
        self.MSE = nn.MSELoss()
        self.beta = 0.66

    def encode(self, state):
        return self.encoder.forward(state)
    
    def forward(self, state, action):
        state = self.encode(state)
        return self.forward_model.forward(state, action)
    
    def inverse(self, state, next_state):
        state = self.encode(state)
        next_state = self.encode(next_state)
        return self.inverse_model.forward(state, next_state)

    def error(self, state, action, next_state):
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        action = self.forward_model.one_hot(action)

        predicted_state = self.forward(state, action)
        predicted_action = self.inverse(state, next_state)

        next_state = self.encode(next_state)

        forward_error = self.MSE(next_state, predicted_state)
        inverse_error = self.MSE(action, predicted_action)
        total_error = (1 - self.beta) * inverse_error + self.beta * forward_error

        return total_error

    def reward(self, state, action, next_state):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = self.forward_model.one_hot(action)

        with torch.no_grad():
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            next_state = self.encode(next_state)

            predicted_state = self.forward(state, action)
            reward = self.MSE(next_state, predicted_state)
        
        return reward


# Neural Network that takes a state, encodes it and returns a 
# probability distribution of which action is likely to be best
class MLPPolicy(nn.Module):
    def __init__(self, input_size, action_size, encoder):
        super(MLPPolicy, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, state):
        state = self.encoder.forward(state)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

