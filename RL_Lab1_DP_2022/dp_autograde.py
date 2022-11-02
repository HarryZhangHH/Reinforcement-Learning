import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    delta = np.inf
    while delta >= theta:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            Q = np.zeros(env.nA)
            for a in range(env.nA):
                transition = env.P[s][a][0]
                prob = transition[0]
                next_s = transition[1]
                reward = transition[2]
                is_done = transition[3]
                Q[a] = prob * (reward + discount_factor * V[next_s])
            V[s] = np.sum(policy[s] * Q)
            delta = np.max([delta, np.abs(v - V[s])])
    # raise NotImplementedError
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    policy_stable = False
    while not policy_stable:
        # policy evaluation
        V = policy_eval_v(policy, env)
        # policy improvement
        policy_stable = True
        for s in range(env.nS):
            old_action = policy[s].copy()
            Q = np.zeros(env.nA)
            for a in range(env.nA):
                transition = env.P[s][a][0]
                prob = transition[0]
                next_s = transition[1]
                reward = transition[2]
                Q[a] = prob * (reward + discount_factor * V[next_s])
            policy[s] = 0  # reset policy
            pi_idx = np.where(Q==np.max(Q))[0]
            policy[s][pi_idx] = 1/pi_idx.shape[0]
            if not np.array_equal(old_action, policy[s]):
                policy_stable = False
    # raise NotImplementedError
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    delta = np.inf
    while delta >= theta:
        delta = 0
        for s in range(env.nS):
            for a in range(env.nA):
                q = Q[s, a]
                transition = env.P[s][a][0]
                prob = transition[0]
                next_s = transition[1]
                reward = transition[2]
                Q[s,a] = prob * (reward + discount_factor * np.max(Q[next_s]))
                delta = np.max([delta, np.abs(q - Q[s, a])])
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        pi_idx = np.where(Q[s]==np.max(Q[s]))
        policy[s][pi_idx] = 1/pi_idx[0].shape[0]
    # raise NotImplementedError
    return policy, Q
