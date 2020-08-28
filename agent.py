import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *

class Agent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        self.env = env
        # Params
        self.num_states = env.observation_space.shape[0] # number of states
        self.num_actions = env.action_space.shape[0] # number of actions
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions) # neural network of input num_states, hidden layer size hidden_size and output num_actions
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions) # same as above
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions) # neural network of input num_states + num_actions, hidden layer size hidden_size and
                                                                                                # output num_actions
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.replay_buffer = ReplayBuffer(max_memory_size) # inizialize replay_buffer of size max_memory_size
        self.critic_loss  = nn.MSELoss() # function for calculate loss
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate) # optimizer for gradient descent
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Noise
        self.noise = OUNoise(self.env.action_space) # noise to add on actions

    # from a state get a specific action
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)) # unsqueeze = view()
        action = self.actor.forward(state) # call forward fuction that use relu activation on neural network
        action = action.detach().numpy()[0,0]  # take first action in first row first column
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size) # take states, actions, rewards and next_states from a sample of memory ReplayBuffer
        states = torch.FloatTensor(states) # transform a list in a FloatTensor --> a tensor of float
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss --> check notes pag 2 we want to minimize the loss function
        Qvals = self.critic.forward(states, actions) # this is the q value output of critic network
        next_actions = self.actor_target.forward(next_states) # this are the actions from output of actor network
        next_Q = self.critic_target.forward(next_states, next_actions.detach()) # combine actor(next_actions) network and critic network(obs next from replay_buffer line 50 and 46) --> q-val next
        Qprime = rewards + self.gamma * next_Q # calculate Qprime

        critic_loss = self.critic_loss(Qvals, Qprime) # the difference between qvalue - q'

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad() # gradient descent
        policy_loss.backward() # backpropagation
        self.actor_optimizer.step() # step

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # save networks parameters --> TODO
    

    def train(self, max_episode, max_step, batch_size, env):
        rewards = []

        for episode in range(max_episode):
            self.noise.reset()
            state = self.env.reset()
            episode_reward = 0

            for step in range(max_step):
                #env.render()
                action = self.get_action(state)
                action = self.noise.get_action(action, step)
                new_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, new_state, done)

                if len(self.replay_buffer) > batch_size:
                    self.update(batch_size)

                state = new_state
                episode_reward += reward

                if done:
                    print("episode " + str(episode) + ", " + "reward  " + str(episode_reward))
                    break

            rewards.append(episode_reward)

        return rewards
