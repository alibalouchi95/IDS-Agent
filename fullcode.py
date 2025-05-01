"""### define the custom environment for IDS (Intrusion Detection System)"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# !pip install keras-rl2

action_mapper = {
    0: "BENIGN",
    1: "DoS slowloris",
    2: "DoS Slowhttptest",
    3: "DoS Hulk",
    4: "DoS GoldenEye",
    5: "Heartbleed",
    6: "DDoS",
    7: "PortScan",
    8: "Bot",
    9: "Infiltration",
    10: "Web Attack � Brute Force",
    11: "Web Attack � XSS",
    12: "Web Attack � Sql Injection",
    13: "FTP-Patator",
    14: "SSH-Patator",
}

attack_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
benign_action = 0


class IntrusionDetectionEnv(gym.Env):

    def __init__(self, dataset):
        self.dataset = dataset  # The Dataset that we are using in this episode
        self.clock = 0
        # Observations are an array of 69 (number of features) numbers between -1 and 1
        self.observation_space = spaces.Box(-1, 1, shape=(69, 1), dtype=float)

        # We have 15 actions, corresponding to {
        #     0:'BENIGN',
        #     1:'DoS slowloris',
        #     2:'DoS Slowhttptest',
        #     3:'DoS Hulk',
        #     4:'DoS GoldenEye',
        #     5:'Heartbleed',
        #     6:'DDoS',
        #     7:'PortScan',
        #     8:'Bot',
        #     9:'Infiltration',
        #     10:'Web Attack � Brute Force',
        #     11:'Web Attack � XSS',
        #     12:'Web Attack � Sql Injection',
        #     13:'FTP-Patator',
        #     14:'SSH-Patator',
        #     }
        self.action_space = spaces.Discrete(15)

        self.actual_predicted_tuples = {}

    def _reward_function(self, action, label):
        reward = 0
        if action in attack_actions:
            if label == benign_action:
                reward = -1
            else:
                if action == label:
                    reward = 1
                else:
                    reward = 0.2
        else:
            if label == action:
                reward = 0.5
            else:
                reward = -1.5
        self.reward = reward
        return reward

    def _get_next_obs(self, clock):
        result = self.dataset.iloc[clock : clock + 1]
        return result

    def _get_obs(self):
        self.clock += 1
        self.observation = self._get_next_obs(self.clock)
        current_observation = self._get_next_obs(self.clock).drop(columns=[" Label"])
        return current_observation

    def _set_info(self, actual_predicted_tuple):
        if actual_predicted_tuple in self.actual_predicted_tuples.keys():
            self.actual_predicted_tuples[actual_predicted_tuple] += 1
        else:
            self.actual_predicted_tuples[actual_predicted_tuple] = 1

    def _get_info(self):
        return {
            "number of predictions (actual, predicted)": self.actual_predicted_tuples
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.reward = 0  # The initial reward for this episode
        self.clock = 0  # The initial value for the clock
        # The first observation is the first row in the dataset
        observation = self._get_obs().values[0]

        return observation

    def step(self, action):
        label = self.observation[" Label"].values[0]

        if self.clock % 1000 == 0:
            print(
                "\n this agent's reward from the beginning to time step:",
                self.clock,
                "is",
                self.reward,
            )
            print("\n", self._get_info())

        # An episode is done if the agent has reached the target (number of data in the dataset)
        terminated = self.clock == len(self.dataset) - 1
        reward = self._reward_function(action, label)

        observation = self._get_obs().values[0]
        info = self.reward

        return observation, reward, terminated, info


def new_env(size_of_data=10000):
    _attacks = all_attacks.sample(frac=1)[: int(size_of_data * 0.3)]
    _benigns = all_benigns.sample(frac=1)[: int(size_of_data * 0.7)]

    new_dataset = pd.concat([_attacks, _benigns]).sample(frac=1)

    env = IntrusionDetectionEnv(new_dataset)

    return env


"""### Prepare the normalized dataset"""


def prepare_dataset():
    dataset_label = " Label"

    print("Reading The dataset...")
    # this dataset should be downloaded and if not the README.md file explains it fully
    dataset = pd.read_csv("./dataset/normalized_dataset.csv")
    dataset = dataset.dropna()

    print("Splitting Attacks and Benigns")

    result = {"attacks": [], "benigns": []}

    result["attacks"].append(dataset[dataset[dataset_label] != 0])

    result["benigns"].append(dataset[dataset[dataset_label] == 0])

    print("Done Splitting The dataset. \n")
    return result


res = prepare_dataset()

all_attacks = res["attacks"][0]
all_benigns = res["benigns"][0]


class IOTEdge:
    def __init__(self):
        # initialize the agent
        model = Sequential()
        model.add(Dense(units=64, input_dim=69, activation="relu"))
        model.add(Dense(units=15, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        self.model = model

    def average_model_weights(self, models_weights, rewards):
        # Calculate the average weights of all models
        average_weights = []
        for j in range(len(models_weights)):
            total_weight = np.sum(
                [
                    rewards[i] * model_weights[j]
                    for i, model_weights in enumerate(models_weights)
                ],
                axis=0,
            )
            average_weight = total_weight / np.sum(rewards)
            average_weights.append(average_weight)
        return average_weights


class IOTDevice:

    def __init__(self, name):
        self.name = name
        self.reward = 0

    def update_model(self, model):
        self.model = model

    def train_model(
        self,
        size_of_data,
        # hyperparameters
        max_steps=150,
        discount_factor=0.95,
        epsilon=1.0,
        min_epsilon=0.01,
    ):

        if max_steps >= size_of_data:
            raise Exception("max_steps should not be greater than the size_of_data")

        # set up the environment
        env = new_env(size_of_data)
        state = env.reset()
        self.reward = 0
        # training loop
        for step in range(max_steps):
            # get the next action
            if np.random.rand() <= epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                state = np.array(state)
                state = state.reshape(1, -1)
                q_values = self.model.predict(state)
                action = np.argmax(q_values[0])

            next_state, reward, done, info = env.step(action)
            self.reward += reward

            state = np.array(state)
            state = state.reshape(1, -1)

            # update the Q-table
            q_values = self.model.predict(state)

            next_state = np.array(next_state)
            next_state = next_state.reshape(1, -1)

            q_values[0][action] = reward + discount_factor * np.max(
                self.model.predict(next_state)[0]
            )
            self.model.fit(state, q_values, verbose=0)

            state = next_state

            # check if the episode has ended
            if done:
                break
            if step % int(max_steps / 100) == 0:
                # update the epsilon value
                epsilon = max(min_epsilon, epsilon * 0.99)

            if step % int(max_steps / 100) == 0:
                # print the reward
                print(
                    "total reward = {0} for total steps = {1}".format(self.reward, step)
                )

    def test_model(self, size_of_data):
        env = new_env(size_of_data)
        total_reward = 0
        state = env.reset()
        episode_reward = 0
        while True:
            state = np.array(state)
            state = state.reshape(1, -1)
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)
            next_state = next_state.reshape(1, -1)
            episode_reward += reward
            if done:
                break
            state = next_state

            total_reward += episode_reward
            average_reward = total_reward / size_of_data
            print(
                "Average reward over {} episodes: {}".format(
                    size_of_data, average_reward
                )
            )

    def return_weights(self):
        return self.model.get_weights()


# number_of_devices = 4
# total_gen_number = 5
# number_of_epochs = 1000
# print_at_every_epochs = 100
# learning_rate = 0.001

# class FederatedLearning():
#     def __init__(self):
#         self.model = []
#         self.model_weights = []
#         self.rewards = []

#     def run(self, total_gen_number=5):
# IOT_devices = []

# for idx in range(number_of_devices):
#     IOT_devices.append(IOTDevice("{0}'s_IOT".format(idx + 1)))

# IOT_device = IOTDevice()
# gen_number = 0
# while gen_number < total_gen_number:

# if(len(self.model_weights) == 0):
#   self.model = IOT_edge.model
# else:
#   self.model = IOT_edge.average_model_weights(self.model_weights, self.rewards)
#   self.model_weights = []
#   self.rewards = []

# for IOT_device in IOT_devices:
# IOT_device.update_model(IOT_edge.model)
# print("Start training")
# IOT_device.train_model(200)
# self.model_weights.append(IOT_device.return_weights())
# self.rewards.append(IOT_device.reward)

# model = IOT_edge.average_model_weights(self.model_weights, self.rewards)
# self.model = IOT_device.model
# self.model.save('./final-model.h5')


# FL = FederatedLearning()
# FL.run(total_gen_number)

IOT_device = IOTDevice()
print("Start training")
IOT_device.train_model(200)
model = IOT_device.model
model.save("./final-model.h5")
