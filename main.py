from csuite.environments import dancing_catch
from csuite.environments import catch
from torch import nn
import torch
from collections import deque
import numpy as np
import random
import itertools
import pickle

import warnings
warnings.filterwarnings('ignore')

# Hyperparameters
GAMMA = 0.9                   #Care more about recent steps
BATCH_SIZE = 32
BUFFER_SIZE = 1000
MIN_REPLAY_SIZE = 32          #To Do #1000 => Care less about previous ones
MAX_STEPS = 2000000
TARGET_UPDATE_FREQ = 128
HIDDEN_LAYERS_SIZE = 64       #To Do 32 for crelu
LEARNING_RATE = 0.01
SWAP_EVERY = 10000
SEED = 6555 #should be loop
AVERAGE_WINDOW_SIZE = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#huber loss for now
criterion = nn.functional.smooth_l1_loss

def set_random_seed():
    random_number = np.random.randint(SEED)
    print("Random Seed is: ", random_number)
    random.seed(random_number)
    np.random.seed(random_number)
    torch.random.manual_seed(random_number)
    torch.cuda.manual_seed(random_number)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DQN(nn.Module):
    def __init__(self, env, hidden_size=128,name=''):
        super(DQN, self).__init__()
        in_features = int(np.product(env.observation_spec().shape))
        n_actions = env.action_spec().num_values

        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_size).to(device)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=n_actions).to(device)  # multiply by 2 because of the CRelu

        self.plate = None

        self.name=name

        self.fc1_input=None
        self.fc1_output=None
        self.act_output=None
        self.output_output=None

        self.to(device)
    

    def crelu(self, x):
        # Apply ReLU separately to positive and negative parts
        return torch.cat((torch.relu(x), torch.relu(-x)), dim=1)  # Concatenate along dim=1

    def forward(self, x):

        self.fc1_input=x

        output = self.fc1(x)

        self.fc1_output=output
        

        # Relu
        output = nn.ReLU()(output)

        self.act_output=output
        

        # Leaky Relu
        # output = nn.LeakyReLU(0.1)(output)

        # CRelu
        # output = self.crelu(output)

        self.plate = output



        output = self.fc2(output)

        self.output_output=output

        return output

    def act(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self(obs)

            max_q_idx = torch.argmax(q_values, dim=1).squeeze() # this needs random tie breaking
            action = max_q_idx.detach().item()

        return action
    

def sample_action(env):
    actions = np.arange(env.action_spec().num_values)
    return np.random.choice(actions)

env = dancing_catch.DancingCatch(swap_every=SWAP_EVERY)

replay_memory = deque(maxlen=BUFFER_SIZE)

online_network = DQN(env, hidden_size=HIDDEN_LAYERS_SIZE, name='online')
target_network = DQN(env, hidden_size=HIDDEN_LAYERS_SIZE, name='target')

target_network.load_state_dict(online_network.state_dict())
optimizer = torch.optim.Adam(online_network.parameters(), lr=LEARNING_RATE) # change to regular  beta1: 0.9, beta2 0.99

#initilize Replay Memory
obs = env.start().flatten()
for _ in range(MIN_REPLAY_SIZE):
    action = sample_action(env)

    new_obs, reward = env.step(action)

    new_obs = new_obs.flatten()
    transition = (obs, action, reward, new_obs)
    replay_memory.append(transition)
    obs = new_obs


from tqdm import tqdm


def write_list_to_file(my_list, file_name):
    try:
        with open(file_name, 'w') as file:
            for item in my_list:
                file.write(f"{item}\n")
        print(f"List has been written to {file_name} successfully.")
    except IOError:
        print("Error: Unable to write to file.")


hidden_layer_weight = []
hidden_layer_gradient =[]
activations = []
train_losses = []
train_losses2 = []
l2train_losses=[]
rewards_log = []
catch_or_miss = [0]
p_bar = tqdm(range(MAX_STEPS))
gloss = 0
total_reward = 0

set_random_seed()

for step in p_bar:
   
    #Epsilon greedy with decaying epsilon between [EPSILON_START, EPSILON_END]
    epsilon = 0.1 #fix
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = sample_action(env)
    else:
        action = online_network.act(obs)

    new_obs, reward = env.step(action)
    new_obs = new_obs.flatten()
    transition = (obs, action, reward, new_obs)
    replay_memory.append(transition)
    obs = new_obs

    if (reward == 1):
        catch_or_miss.append(1)
    elif(reward == -1):
        catch_or_miss.append(0)

    avg_window = catch_or_miss[max(0, len(catch_or_miss) - AVERAGE_WINDOW_SIZE): ]
    reward_avg = sum(avg_window) / len(avg_window)
    rewards_log.append(reward_avg)


    transitions = random.sample(replay_memory, BATCH_SIZE)

    # Deep Learning part
    observations = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    new_observations = np.asarray([t[3] for t in transitions])

    observations = torch.tensor(observations, dtype=torch.float32).to(device) #(BATCH_SIZE, 50)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device) #(BATCH_SIZE, 1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device) #(BATCH_SIZE, 1)
    new_observations = torch.tensor(new_observations, dtype=torch.float32).to(device) #(BATCH_SIZE, 50)

    target_q = target_network(new_observations) #(BATCH_SIZE, 3)
    max_target_q = target_q.max(dim=1, keepdim=True)[0]

    targets = rewards + GAMMA * max_target_q

    #Compute loss
    q_values = online_network(observations)

    action_q_values = torch.gather(q_values, dim=1, index=actions).to(device)

    loss = criterion(action_q_values, targets)
    mean_loss = loss.detach().cpu().item() / BATCH_SIZE


    gloss += mean_loss
    train_losses2.append(mean_loss)
    if (step + 1 ) % 5000 == 0:
        train_losses.append(mean_loss)

    p_bar.set_description(f"gLoss: {gloss / (step + 1) * 1000:0.5f}, reward: {reward_avg:0.5f}, mean loss: {mean_loss:.5f}")

    #Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    #Clip gradients
    for param in online_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


   
    # Update target net
    if (step + 1) % TARGET_UPDATE_FREQ == 0:
        target_network.load_state_dict(online_network.state_dict())

    if (step + 1 ) % 5000 == 0:

        hidden_layer_weight.append(online_network.fc1.weight.data.detach().clone())  

      # Get gradients of hidden layer weights
        hidden_layer_gradient.append(online_network.fc1.weight.grad.detach().clone())

        
        activations.append(online_network.plate)

    # print("online_network.fc1_input: ", online_network.fc1_input.shape)
    # print("online_network.fc1_input: ", online_network.fc1_output.shape)
    # print("online_network.fc1_input: ",online_network.act_output.shape)
    # print("online_network.output_output: ", online_network.output_output.shape)






activationFunction ='Relu'
myseed =1

path = f"seed{myseed}_maxsteps{MAX_STEPS}_swap{SWAP_EVERY}_{activationFunction}_"


with open(path + 'activations.pkl', 'wb') as f:
        pickle.dump(activations, f)

with open(path + 'hidden_layer_gradient.pkl', 'wb') as f:
        pickle.dump(hidden_layer_gradient, f)

with open(path + 'hidden_layer_weight.pkl', 'wb') as f:
        pickle.dump(hidden_layer_weight, f)

with open(path + 'loss.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

with open(path + 'reward.pkl', 'wb') as f:
        pickle.dump(rewards_log, f)
        
        
print('Saved!')

