import matplotlib.pyplot as plt
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')
#print(env.action_space)
#print(env.observation_space)
def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
    epsilon = 1.0/i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon /env.nA
    policy_s[np.argmax(Q_s)] = 1-epsilon+(epsilon/env.nA)
    return policy_s

#TD Control: sarsa

def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # progress
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}/{num_episodes}')
            sys.stdout.flush()

        score = 0
        state = env.reset()
        policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
        action = np.random.choice(np.arange(env.nA), p=policy_s)
        
        for t_step in np.arange(300):
            next_state, reward, done, info = env.step(action)
            score += reward

            if not done:
                # get epsilon greedy action probilities
                policy_s = epsilon_greedy_probs(env,Q[next_state], i_episode)
                next_action = np.random.choice(np.arange(env.nA), p=policy_s)
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], 
                                            reward, alpha, gamma)
                # S <- S'
                state = next_state
                # A <- A'
                action = next_action
            if done:
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                # append score
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q
                
# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)