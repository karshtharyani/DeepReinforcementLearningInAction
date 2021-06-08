# Policy Gradients and the REINFORCE
## Recap
- Till now, we are deciding what action to take based on the Q value
- A Q value is a function of the state and action which returns some scalar.

## Here's a short quiz - 
### How do you test/deploy your reinforcement learning algorithm given your Q network?
- Option A: We use the epsilon greedy strategy which was used to train the network
- Option B: We use the contextual policy similar to the context bandits
- Option C: We use the agrmax action of the Q value for a state 

### Now if we used what this chapter talks about - how would we use our "agent"?

## What's different now and what's the main idea?
- **Don't learn the Q value to learn the optimal policy**
- Think of maximizing your policy distribution over states. 
- Think gradient descent to converge to a degenerate/stochastic policy distribution

## Keywords
- Stochastic policy - applicable if your environment is random
- Deterministic policy - applicable if your environment is stationary
- Degenerate distribution over actions - given a state - something that outputs
a 1 for the optimal action and zero for the rest

## REINFORCE - really boils down to Section 4.2.2
- Action reinforcement is a tricky idea for an RL agent
- You could use gradient descent to update your network in a supervised learning
problem because it was static (no notion of time) by following the label one hot
vector.  [CategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)
- But that was supervised learning and there was no temporal corelation. If I
gave the network an input now versus the next time step, the inputs are not
corelated in time. It doesn't matter which image was shown first during
training/testing. 
- But **we want to capture temporal corelation** because your current action
influences the next and so on. 

### Maximize the distribution or minimize the negative log of it
- If *pi(a|theta)* increases, *log(pi(a|theta))* increases, or its *-log()*
decreases

### Credit assignment
- **Every RL problem is episodic.**
- We are updating our neural net with every step. 
- Think long term - the further down the line better action, the more reliable. 
- G_t := future return cumulative reward from *t* till *T* termination.
- Actions temporally more distant from the received reward should be weighted
less than actions closer.

## Implementation
- See cartpoleproblem.py
- See [Gym Environment for Cart
  Pole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

