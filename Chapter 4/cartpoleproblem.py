import numpy as np
import torch
import gym
from matplotlib import pyplot as plt

def nn_model():
    l1 = 4 #A
    l2 = 150
    l3 = 2 #B

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.Softmax(dim=0) #C
    )
    return model

def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards #A
    disc_return /= disc_return.max() #B
    return disc_return

def loss_fn(preds, r): #A
    return -1 * torch.sum(r * torch.log(preds)) #B

def REINFORCE(env, optimizer, model):
    MAX_DUR = 200
    MAX_EPISODES = 500
    gamma = 0.99
    score = [] #A
    expectation = 0.0
    for episode in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False
        transitions = [] #B
        
        for t in range(MAX_DUR): #C
            act_prob = model(torch.from_numpy(curr_state).float()) #D
            action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) #E
            prev_state = curr_state
            curr_state, _, done, info = env.step(action) #F
            transitions.append((prev_state, action, t+1)) #G
            if done: #H
                break

        ep_len = len(transitions) #I
        score.append(ep_len)
        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) #J
        disc_returns = discount_rewards(reward_batch) #K
        state_batch = torch.Tensor([s for (s,a,r) in transitions]) #L
        action_batch = torch.Tensor([a for (s,a,r) in transitions]) #M
        pred_batch = model(state_batch) #N
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() #O
        loss = loss_fn(prob_batch, disc_returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def some_scratch_code(env):
    box_obs = env.observation_space
    actions = env.action_space
    env.reset()
    for i in range(100):
        env.render()
        obs, reward, done, info = env.step(actions.sample())
        print(obs, reward, done, info)

def test_model(env, model):
    MAX_DUR = 200
    score = []
    games = 100
    done = False
    state1 = env.reset()
    for i in range(games):
        t=0
        while not done: #F
            env.render()
            pred = model(torch.from_numpy(state1).float()) #G
            action = np.random.choice(np.array([0,1]), p=pred.data.numpy()) #H
            state2, reward, done, info = env.step(action) #I
            state1 = state2 
            t += 1
            #if t > MAX_DUR: #L
            #    break;
        state1 = env.reset()
        done = False
        score.append(t)
    score = np.array(score)

if __name__=="__main__":
    model = nn_model()
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    env = gym.make("CartPole-v0")
    #some_scratch_code(env)
    REINFORCE(env, optimizer=optimizer, model=model)
    test_model(env, model)
