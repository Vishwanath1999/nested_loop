import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import random
from torch.distributions import Normal, MultivariateNormal

class ReplayBuffer():
    def __init__(self, input_shape, n_actions, max_size=int(1e6)):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros((self.mem_size,1))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def reallocate(self,max_size=None):
        if max_size is not None:
            self.mem_size=max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*self.input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*self.input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,self.n_actions))
        self.reward_memory = np.zeros((self.mem_size,1))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def add_multiple(self,size,state,action,reward,next_state,done):
        self.state_memory[self.mem_cntr:self.mem_cntr+size] = state
        self.action_memory[self.mem_cntr:self.mem_cntr+size] = action
        self.reward_memory[self.mem_cntr:self.mem_cntr+size] = reward
        self.new_state_memory[self.mem_cntr:self.mem_cntr+size] = next_state
        self.terminal_memory[self.mem_cntr:self.mem_cntr+size] = done
        self.mem_cntr += size

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states,actions,rewards,states_,dones

class Model(nn.Module):
    def __init__(self,input_dims,n_actions,hidden_dims=200,\
        weight_decay=1e-4,name='model_mbpo',chkpt_dir='tmp/model',use_bn=False):
        super(Model, self).__init__()
        
        self.chkpt_file = os.path.join(chkpt_dir,name)
        self.weight_decay = weight_decay
        self.use_bn = use_bn

        self.MAX_LOG_VAR = T.tensor(-2, dtype=T.float32) # 0.5 was -2.
        self.MIN_LOG_VAR = T.tensor(-5., dtype=T.float32) #-10 was -5.

        self.fc1 = nn.Linear(input_dims[0]+n_actions,hidden_dims)        
        self.fc2 = nn.Linear(hidden_dims,hidden_dims)
        self.fc3 = nn.Linear(hidden_dims,hidden_dims)
        self.fc4 = nn.Linear(hidden_dims,hidden_dims)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dims)
            self.bn2 = nn.BatchNorm1d(hidden_dims)
            self.bn3 = nn.BatchNorm1d(hidden_dims)
            self.bn4 = nn.BatchNorm1d(hidden_dims)

        self.state_mu = nn.Linear(hidden_dims,input_dims[0])
        self.state_sigma = nn.Linear(hidden_dims,input_dims[0])

        self.reward_mu = nn.Linear(hidden_dims,1)
        self.reward_sigma = nn.Linear(hidden_dims,1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        state = T.cat([state,action],dim=1)
        x = self.fc1(state)
        if self.use_bn:
            x = self.bn1(x)
        x = F.silu(x)

        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.silu(x)

        x = self.fc3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = F.silu(x)

        x = self.fc4(x)
        if self.use_bn:
            x = self.bn4(x)
        x = F.silu(x)

        state_mu = self.state_mu(x)
        log_var_output = self.state_sigma(x)

        reward_mu = self.reward_mu(x)
        log_var_reward = self.reward_sigma(x)

        log_var_output = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_output)
        log_var_reward = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_reward)

        log_var_output = self.MIN_LOG_VAR + F.softplus(log_var_output - self.MIN_LOG_VAR)
        log_var_reward = self.MIN_LOG_VAR + F.softplus(log_var_reward - self.MIN_LOG_VAR)

        return [state_mu,log_var_output],[reward_mu,log_var_reward]


    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    
class EnsembleModel:
    def __init__(self,alpha,input_dims,n_actions,weight_decay,n_models,n_elites=None,hidden_dims=200,\
        batch_size=256,name='model_env_2',use_bn=False):
        self.alpha = alpha
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.n_models = n_models

        if n_elites is None:
            self.n_elites = n_models
        else:
            self.n_elites = n_elites
        
        self.models = nn.ModuleList([Model(input_dims=input_dims,n_actions=n_actions,hidden_dims=hidden_dims,\
            weight_decay=weight_decay,name=name+'_'+str(idx),use_bn=use_bn) for idx in range(n_models)])
        self.model_optimizer = optim.AdamW(params=self.models.parameters(),lr=alpha,weight_decay=weight_decay,amsgrad=True)
    
    
    def forward(self,state,action):
        model_outs = [model.forward(state,action) for model in self.models]
        o_next_pred, r_pred = (zip(*model_outs))
        o_next_pred = [T.stack(item) for item in zip(*o_next_pred)]
        o_next_pred[0] = T.flatten(state,start_dim=1) + o_next_pred[0]
        r_pred = [T.stack(item) for item in zip(*r_pred)]
        return o_next_pred,r_pred

    def get_traj(self,state,action,reward,next_state,batch_size,elite_idx=None):
        
        idx = [i for i in range(batch_size)]
        if self.n_elites == self.n_models:
            elite_list = np.arange(self.n_models)
        else:
            if elite_idx is not None:
                elite_list = elite_idx
            else:
                elite_list = self.get_elite_idx(state,action,reward,next_state)
        randomlist = random.choices(elite_list, k=batch_size)
        [state_mu,log_state_sigma],[reward_mu,log_reward_sigma] = self.forward(state,action)
        next_state = T.normal(state_mu,T.exp(0.5*log_state_sigma))
        reward = T.normal(reward_mu,T.exp(0.5*log_reward_sigma))
        Next_state = next_state.cpu().detach().numpy()
        Reward = reward.cpu().detach().numpy()
        return Next_state[randomlist, idx, :],Reward[randomlist, idx, :]
    
    def get_elite_idx(self,states,actions,rewards,next_states):
        o_next_pred,r_pred = self.forward(states,actions)
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        l1_eval = T.mean(T.cat((((mu_o - next_states) * (mu_o - next_states)),
                                    ((mu_r - rewards) * (mu_r - rewards))), dim=-1),dim=(-1,-2)).cpu().detach().numpy()
        idxs = np.argsort(l1_eval)
        return idxs[:self.n_elites]

    def update_step(self,states,actions,rewards,next_states):
        
        o_next_pred,r_pred = self.forward(states,actions)
        log_var_o = o_next_pred[1]
        log_var_r = r_pred[1]
        inv_var_o = T.exp(-log_var_o)
        inv_var_r = T.exp(-log_var_r)
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        l1_eval = T.mean(T.cat((((mu_o - next_states) * (mu_o - next_states)),
                                    ((mu_r - rewards) * (mu_r - rewards))), dim=-1),
                        dim=(-1, -2))
        l1_eval_ = l1_eval.cpu().detach().numpy().mean()

        vx = rewards - T.mean(rewards)
        vy = mu_r[0] - T.mean(mu_r[0])      
        R_corr_r = (T.sum(vx * vy) / (T.sqrt(T.sum(vx ** 2)) * T.sqrt(T.sum(vy ** 2)))).cpu().detach().numpy()

        vx = next_states - T.mean(next_states)
        vy = mu_o[0] - T.mean(mu_o[0])
        R_corr_o = (T.sum(vx * vy) / (T.sqrt(T.sum(vx ** 2)) * T.sqrt(T.sum(vy ** 2)))).cpu().detach().numpy()

        l1 = T.sum(T.mean(T.cat(((0.5*(mu_o - next_states) * inv_var_o * (mu_o - next_states)),\
            0.5*((mu_r - rewards) * inv_var_r * (mu_r - rewards))), dim=-1),dim=(-1,-2)))
        l2 = T.sum(T.mean(T.cat((log_var_o, 1*log_var_r), dim=-1),dim=(-1,-2)))

        loss = l1+l2
        l1_= l1.item()
        l2_ = l2.item()
        self.model_optimizer.zero_grad()
        loss.backward()
        # T.nn.utils.clip_grad_norm_(self.models.parameters(),0.5)
        self.model_optimizer.step()
        return l1_*np.ones(1)+l2_*np.ones(1),l1_eval_,R_corr_o,R_corr_r
    
    def save_models(self):
        print(" ..... saving models ..... ")
        # self.model.eval()
        for idx in range(self.n_models):
            self.models[idx].save_checkpoint()
        # self.model.train()
    
    def load_models(self):
        print(" ..... loading models ..... ")
        for idx in range(self.n_models):
            self.models[idx].load_checkpoint()
        # self.model.train()

class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_dims,n_actions,fc1_dims,fc2_dims,name,\
        chkpt_dir='tmp/sac',max_action=1,act_dist='normal'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        name += '.pt'
        self.chkpt_file = os.path.join(chkpt_dir,name)
        self.reparam_noise=1e-6
        self.act_dist = act_dist

        self.fc1 = nn.Linear(*input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims,self.n_actions)      
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.max_action = T.tensor(max_action).to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma,-20,2).exp()
        return mu,sigma
    
    # def sample_normal(self,state,reparameterize=False,deterministic=False):
    #     mu,sigma = self.forward(state)
    #     pi_dist = T.distributions.Normal(mu,sigma)
        
    #     if deterministic==True:
    #         return self.action_scale*T.tanh(mu)

    #     if reparameterize:
    #         actions = pi_dist.rsample() # reparameterizes the policy
    #     else:
    #         actions = pi_dist.sample()
            
    #     action = T.tanh(actions)
    #     log_probs = pi_dist.log_prob(actions)
    #     log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
    #     log_probs = log_probs.sum(1)
    #     return action, log_probs
    
    def sample_normal(self,state,reparameterize=False,deterministic=False):
        mu,sigma = self.forward(state)
        if deterministic is True:
            return T.tanh(mu),None
        if self.act_dist == 'normal':
            pi_dist = Normal(mu,sigma)
            pi_action = pi_dist.rsample() 
            log_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            corr = (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            log_pi -= corr
            pi_action = self.max_action*T.tanh(pi_action)
        elif self.act_dist == 'mv_normal':
            pi_dist = MultivariateNormal(mu,sigma)
            pi_action = pi_dist.rsample() 
            log_pi = pi_dist.log_prob(pi_action)
            pi_action = self.max_action*T.tanh(pi_action)
        return pi_action,log_pi
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
            n_actions,name, chkpt_dir='./tmp/sac'):#
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)        

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q = F.relu(self.fc1(T.cat([state,action],dim=1)))
        q = F.relu(self.fc2(q))
        q = self.q1(q)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class SAC_Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,n_elites,
            env_id, gamma=0.99,n_models=10,fake_ratio = 0.95,rollout_len=1,act_dist='normal',
            n_actions=2, max_size=int(1e6), layer1_size=256,weight_decay=1e-5,no_bad_state=False,
            layer2_size=256, ac_batch_size=256, model_lr=1e-2,model_batch_size=512,use_model=True,use_bn=False):
        self.gamma = gamma
        self.tau = tau
        self.real_memory = ReplayBuffer(input_dims, n_actions, max_size)
        self.fake_memory = ReplayBuffer(input_dims, n_actions, 2000)
        self.ac_batch_size = ac_batch_size
        self.model_lr = model_lr
        self.model_batch_size = model_batch_size
        self.n_actions = n_actions
        self.use_model = use_model
        self.fake_ratio = fake_ratio
        self.env_id = env_id
        self.rollout_len = rollout_len
        self.no_bad_state = no_bad_state

        if self.use_model==True:
            self.models = EnsembleModel(alpha=model_lr,input_dims=input_dims,n_actions=n_actions,\
                weight_decay=weight_decay,n_models=n_models,batch_size=model_batch_size,n_elites=n_elites,use_bn=use_bn)
        else:
            self.models = None

        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=layer1_size,
                                  fc2_dims=layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor', act_dist=act_dist,
                                  max_action=env.action_space.high)
        
        self.critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
       
        self.target_critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name=env_id+'_target_critic_1')

        self.target_critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name=env_id+'_target_critic_2')

        self.update_network_parameters(tau=1)
        self.target_ent_coef = -np.prod(env.action_space.shape)
        self.log_ent_coef = T.log(T.ones(1,device=self.actor.device)).requires_grad_(True)
        self.ent_coef_optim = T.optim.Adam([self.log_ent_coef],lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    
    def check_terminal(self,obs,no_bad_state=False):
        if 'Hopper' in self.env_id:
            if no_bad_state == False:
                done1 = np.array([all(x >= -100 and x <= 100 for x in obs[i,1:]) for i in range(obs.shape[0])]).astype(bool)
                done2 = np.array([obs[i,0]>0.7 for i in range(obs.shape[0])]).astype(bool)
                done3 = np.array([obs[i,1]>=-0.2 and obs[i,1]<=0.2 for i in range(obs.shape[0])]).astype(bool)
                return done1*done2*done3
            else:
                done = np.ones((obs.shape[0])).astype(bool)
                return done
        else:
            return

    def choose_action(self, observation,deterministic=False):

        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, deterministic=deterministic)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.real_memory.store_transition(state, action, reward, new_state, done)
    
    def real_mem_ready(self):
        flag = False
        if self.real_memory.mem_cntr > self.model_batch_size:
            flag = True
        return flag
    
    def fake_mem_ready(self):
        flag = False
        if self.fake_memory.mem_cntr > self.ac_batch_size:
            flag = True
        return flag
    
    def train_model(self):
        device = self.device

        state,action,reward,state_,_= \
                            self.real_memory.sample_buffer(self.model_batch_size)
                            
        states = T.tensor(state,dtype=T.float).to(device)
        actions = T.tensor(action,dtype=T.float).to(device)
        rewards = T.tensor(reward,dtype=T.float).to(device)
        next_states = T.tensor(state_,dtype=T.float).to(device)
        next_states = T.flatten(next_states,start_dim=1)
        loss,l1_eval,r_corr_s,r_corr_r = self.models.update_step(states,actions,rewards,next_states)
        return loss,l1_eval,r_corr_s,r_corr_r
    
    def generate_efficient(self, n_ac_steps=20):
        batch_size = 1000
        self.fake_memory.reallocate(batch_size*n_ac_steps*self.rollout_len)
        for _ in range(n_ac_steps):
            state,act,reward,next_state,done = self.real_memory.sample_buffer(batch_size)
            stateT = T.tensor(state,dtype=T.float).to(self.device)
            rewardT = T.tensor(reward,dtype=T.float).to(self.device)
            next_stateT = T.tensor(next_state,dtype=T.float).to(self.device)
            actT = T.tensor(act,dtype=T.float).to(self.device)
            elite_idx = self.models.get_elite_idx(stateT,actT,rewardT,next_stateT)
            for idx in range(self.rollout_len):
                actionT = self.actor.sample_normal(stateT)[0]
                action = actionT.cpu().detach().numpy()[0]
                next_state_pred,reward_pred = self.models.get_traj(stateT,actionT,rewardT,next_stateT,batch_size,elite_idx)
                if idx == 0:
                    self.fake_memory.add_multiple(batch_size,state,action,reward_pred,\
                            next_state_pred,done)
                else:
                    state = stateT.cpu().detach().numpy()
                    self.fake_memory.add_multiple(batch_size,state,action,reward_pred,next_state_pred,self.check_terminal(state,self.no_bad_state))
                    stateT = T.tensor(next_state_pred,dtype=T.float).to(self.device)

    def train_ac(self):
        if self.use_model == True:
            fake_ratio = int(self.fake_ratio*self.ac_batch_size)
            state, action, reward, new_state, done = \
                    self.fake_memory.sample_buffer(fake_ratio)
            state1,action1,reward1,new_state1,done1 = \
                    self.real_memory.sample_buffer(self.ac_batch_size-fake_ratio)
                    
            state = np.append(state,state1,axis=0)
            action = np.append(action,action1,axis=0)
            reward = np.append(reward,reward1,axis=0)
            new_state = np.append(new_state,new_state1,axis=0)
            done = np.append(done,done1,axis=0)
        else:
            state, action, reward, new_state, done = \
                    self.real_memory.sample_buffer(self.ac_batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
       
        actions_, log_probs_ = self.actor.sample_normal(state_)
        actions, log_probs = self.actor.sample_normal(state)
        
        ent_coef_loss = -(self.log_ent_coef*(log_probs+self.target_ent_coef).detach()).mean()
        ent_coef = T.exp(self.log_ent_coef.detach())
        
        q1_ = self.target_critic_1.forward(state_, actions_)
        q2_ = self.target_critic_2.forward(state_, actions_)

        critic_value_ = T.min(q1_, q2_)
        critic_value_ = critic_value_.view(-1)

        # Critic Optimization
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        target = reward.view(-1) + (1-done.int())*self.gamma*(critic_value_ - ent_coef*log_probs_)
        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1, target)
        critic_2_loss = 0.5*F.mse_loss(q2, target)
        critic_loss = critic_1_loss + critic_2_loss
        cl = critic_loss.item()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state,reparameterize=True)

        q1 = self.critic_1.forward(state, actions)
        q2 = self.critic_2.forward(state, actions)
        critic_value = T.min(q1, q2)
        critic_value = critic_value.view(-1)

        #Actor Optimization
        actor_loss = T.mean(ent_coef*log_probs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optimizer.step()        
        al = actor_loss.item()    
             
        # Entropy Optimization
        self.ent_coef_optim.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optim.step()

        self.update_network_parameters()
        return al,cl,ent_coef_loss.item(),ent_coef.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        
        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_1_params = self.critic_1.named_parameters()

        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)

        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        # self.critic_1.save_checkpoint()
        # self.critic_2.save_checkpoint()
        # self.target_critic_1.save_checkpoint()
        # self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        # self.critic_1.load_checkpoint()
        # self.critic_2.load_checkpoint()
        # self.target_critic_1.load_checkpoint()
        # self.target_critic_2.load_checkpoint()