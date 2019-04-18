# TODO: your agent here!
import random
import numpy as np
from collections import deque

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Lambda, Input, Add, Activation, LeakyReLU
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras import backend as K

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        
        self.build_model()
        
    def build_model(self):
        states = Input(shape=(self.state_size,), name='states')
        
        layers = Dense(24, activation='relu', kernel_initializer='random_normal')(states)
        layers = Dense(48, activation='relu', kernel_initializer='random_normal')(layers)
        layers = Dense(48, activation='relu', kernel_initializer='random_normal')(layers)
        layers = Dense(48, activation='relu', kernel_initializer='random_normal')(layers)
        layers = Dense(24, activation='relu', kernel_initializer='random_normal')(layers)
        
        raw_actions = Dense(self.action_size, activation='sigmoid')(layers)
        
        def l(x):
            scaled_x = x * self.action_high + self.action_low
            return K.clip(scaled_x, self.action_low, self.action_high)
        
        actions = Lambda(l)(raw_actions)
        
        self.model = Model(inputs=states, outputs=actions)
        
        gradients = Input(shape=(self.action_size,))
        loss = K.mean(-gradients * actions)
        
        optimizer = Adam()
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_function = K.function(inputs=[self.model.input, gradients, K.learning_phase()], outputs=[], updates=updates)
        
class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
    def build_model(self):
        states = Input(shape=(self.state_size,))
        actions = Input(shape=(self.action_size,))
        
        state_layers = Dense(32, activation='relu')(states)
        state_layers = Dense(48, activation='relu')(state_layers)
        state_layers = Dense(48, activation='relu')(state_layers)
        state_layers = Dense(32, activation='relu')(state_layers)
        
        action_layers = Dense(32, activation='relu')(actions)
        action_layers = Dense(48, activation='relu')(action_layers)
        action_layers = Dense(48, activation='relu')(action_layers)
        action_layers = Dense(32, activation='relu')(action_layers)
        
        layers = Add()([state_layers, action_layers])
        layers = Activation('relu')(layers)
        
        Q = Dense(1)(layers)
        
        self.model = Model(inputs=[states, actions], outputs=Q)
        
        optimizer = Adam()
        
        self.model.compile(loss='mse', optimizer=optimizer)
        
        gradients = K.gradients(Q, actions)
        self.get_action_gradients = K.function(inputs=[*self.model.inputs, K.learning_phase()], outputs=gradients)      
        
import copy
class Noise:
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma  * np.random.randn(len(x))
        
        self.state = x + dx
        
        return self.state
        

class Agent():
    def __init__(self, task, max_memory=100000, batch_size=32):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.memory = deque(maxlen=max_memory)
        self.batch_size = 64
        
        # Episode variables
        self.reset_episode()
        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        #print(self.actor_local.model.summary())
        #print(self.critic_local.model.summary())
        
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        
        self.gamma = 0.99
        self.tau = 0.01
        
        self.best_score = -np.inf
        self.total_reward = 0
        self.count = 0
        self.score = -np.inf
        
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.noise = Noise(self.action_size, self.mu, self.theta, self.sigma)
        
    def reset_episode(self):
        self.total_reward = 0
        state = self.task.reset()
        self.last_state = state
        return state
    
    def step(self, action, reward, next_state, done):
        self.memory.append((self.last_state, action, reward, next_state, done))
        
        self.count += 1
        self.total_reward += reward
        
        if len(self.memory) > self.batch_size:
            self.learn()
            
        self.score = self.total_reward
        if self.score > self.best_score:
            self.best_score = self.score
            
    def act(self, state):
        state = np.reshape(state, (-1, self.state_size))
        self.last_state = state
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
            
    def learn(self):
        experiences = random.sample(self.memory, self.batch_size)
        
        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.vstack([e[2] for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        Q_targets = rewards + self.gamma * Q_next * (1 - dones)
        
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_function([states, action_gradients, 1])
        
        self.soft_update(self.actor_local.model, self.actor_target.model)
        self.soft_update(self.critic_local.model, self.critic_target.model)
        
        
    def soft_update(self, local, target):
        local_weights = np.array(local.get_weights())
        target_weights = np.array(target.get_weights())
        
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        
        target.set_weights(new_weights)
        
        