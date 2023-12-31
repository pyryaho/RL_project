from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        # added
        state_shape = (state_dim, )
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
        # added
        self.buffer_size = self.cfg.buffer_size
        self.d = 2
        self.d_counter = 0
        
        # for the OU-process
        self.prev_noise = 0. * np.ones(self.action_dim)
        self.alpha = 0.
        self.beta = 0.2 
        self.sigma = 0.3
        self.wiener = 0. * np.ones(self.action_dim)
        
        # Initialize one actor and two critics
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))

        self.q1 = Critic(state_dim, self.action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=float(self.lr))
        
        self.q2 = Critic(state_dim, self.action_dim).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))
        
        self.buffer = ReplayBuffer(state_shape, self.action_dim, max_size=int(float(self.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
    

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        #from ex6_DDPG
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transition

        if self.buffer_ptr > self.random_transition: # update once we have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    # method from ex6_DDPG
    def _update(self,):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        # batch contains:
        #    state = batch.state, shape [batch, state_dim]
        #    action = batch.action, shape [batch, action_dim]
        #    next_state = batch.next_state, shape [batch, state_dim]
        #    reward = batch.reward, shape [batch, 1]
        #    not_done = batch.not_done, shape [batch, 1]
        state_batch = batch.state
        next_state_batch = batch.next_state
        action_batch = batch.action
        reward_batch = batch.reward 
        not_done = batch.not_done 
        # TODO:
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using u.soft_update_params() (See the DQN code)
        # compute current q
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q1 = self.q1(state_batch, action_batch)
        q2 = self.q2(state_batch, action_batch)
        
        # print(q.shape)
        # compute target q
        next_state_action_values_1 = self.q1_target(next_state_batch, self.pi_target(next_state_batch))
        next_state_action_values_2 = self.q2_target(next_state_batch, self.pi_target(next_state_batch))
        
        next_state_action_values = torch.min(next_state_action_values_1, next_state_action_values_2)
        q_tar = (next_state_action_values*(not_done)*self.gamma) + reward_batch
        # print(q_tar.shape)
        # compute critic loss
        criterion = torch.nn.MSELoss()
        critic_loss = criterion(q1, q_tar) + criterion(q2, q_tar)

        # optimize the critics
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        critic_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()
        
        # do actor update
        if self.d_counter >= self.d:
            # compute actor loss
            actor_loss = -self.q1(state_batch, self.pi(state_batch))
            actor_loss = actor_loss.mean()

            # optimize the actor
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            # update the target q and target pi using u.soft_update_params() function
            cu.soft_update_params(self.pi, self.pi_target, self.tau)
            cu.soft_update_params(self.q1, self.q1_target, self.tau)
            cu.soft_update_params(self.q2, self.q2_target, self.tau)
            
            # reset the counter
            self.d_counter = 0
        
        # increment the counter
        self.d_counter += 1
        
        ########## Your code ends here. ##########
        
        return {}
    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        
        # from ex6_DDPG
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)

        # The exploration noise added to the action follows a Gaussian distribution (mean is 0 and standard deviation is 0.3)
        # expl_noise = 0.3 * self.max_action # the stddev of the expl_noise if not evaluation
            
        # TODO:
        ########## Your code starts here. ##########
        # Use the policy to calculate the action to execute
        # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
        # Hint: Make sure the returned action's shape is correct.
        if evaluation:
            action = self.pi.forward(x)
        else:
            action = self.pi.forward(x)
             # OU-noise
            x = self.prev_noise
            W = self.wiener + np.array([np.random.randn() for i in range(len(x))])
            # the 
            x_next = x - self.beta * (x-self.alpha) + self.sigma * (W-self.wiener)
            self.wiener = W
            self.prev_noise = x_next    
            action += x_next
            # action += expl_noise*torch.randn_like(action)

        ########## Your code ends here. ##########
        # clip the action
        return action.clip(-1,1), {} # just return a positional value

        
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, _ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        print(filepath)
        
        d = torch.load(filepath)
        self.q1.load_state_dict(d['q1'])
        self.q1_target.load_state_dict(d['q1_target'])
        self.q2.load_state_dict(d['q2'])
        self.q2_target.load_state_dict(d['q2_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q1': self.q1.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2': self.q2.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
       
    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
