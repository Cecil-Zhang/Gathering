---
title:  "Reinforcement Learning"
date:   2022-05-02 21:00:04 -0700
categories: 
 - Artificial Intelligence
toc: true
toc_sticky: true
usemathjax: true
---

## Introduction

- Reinforcement learning addresses the question of how an autonomous agent that senses and acts in its environment can learn to choose optimal actions to achieve its goals.

![RL Environment]({{ site.baseurl }}/assets/images/RL/RL.png)

- Difference from other function approxmation tasks
    - Delayed reward: In RL, training information is not available in form $$<s,\pi(s)>$$. Instead, the trainer provides only a sequence of immediate reward values as the agent executes its sequence of actions. The agent faces the problem of ***temporal credit assignment:*** determining which of the actions in its sequence are to be credited with producing the eventual rewards.
    - Exploration: The learner faces a tradeoff in choosing whether to favor *exploration* of unkown states and actions, or *exploitation* of states and actions that it has already learned.
    - Partially observable states: in many practical situations sensors provide only partial information.
    - Life-long learning.

## Reinforcement Learning Settings

### Terminology

- Markov decision processes (MDP):
    - The agent can perceive a set ***S*** of distinct **states** of its **environment** $$\epsilon$$ and has a set ***A*** of **actions** that it can perform.
    - At each discrete time step *t ,* the agent senses the **current** **state** $$s_t$$***,*** chooses a current **action** $$a_t$$***,*** and performs it.
    - The **environment** responds by giving the agent a **reward** $$r_t=r(s_t,a_t)$$ and by producing the **succeeding state** $$s_{t+1}=\delta(s_t,a_t)$$.
    - The **reward function** and **state transition function** $$\delta, r$$ are part of the **environment** and are not neccessarily known to the agent.
    - The functions $$\delta(s_t,a_t)$$ and $$r(s_t,a_t)$$ depend only on the **current state** and **action**, and not on earlier states or actions.
- ***S*** and ***A*** are finite.

### Goal

- The task of the agent is to learn a ***policy,*** $$\pi: S\to A$$, for selecting its next action $$a_t$$ based on the current observed state $$s_t$$***;*** that is, $$\pi(s_t)=a_t$$.
- We require that the agent learns a policy $$\pi$$ that maximizes **Discounted Cumulative Reward** $$V^\pi (s)$$ for all states *s*.
- Such a policy is called an *optimal policy*, denoted as $$\pi^*$$.
- $$\pi^*\equiv\arg\max V^\pi(s),\;(\forall s)$$
- Simplify the value function $$V^{\pi^*}(s)$$ to $$V^*(s)$$. It gives the maximum discounted cumulative reward that the agent can obtain starting from state s. That is, the discounted cumulative reward obtained by following the optimal policy beginning at state s.

### Environment

- **State transition function**: $$\delta(s,a)\to s'$$
- **Reward function**: assigns a numerical value to each distinct action the agent may take from distinct state
    - **Discounted Cumulative  Reward**: $$V^\pi (s_t)=\sum_{i=0}^\infty r_{t+i}\cdot \gamma^i$$
        - $$0\le \gamma <1$$: a constant that determines the relative value of delayed versus immediate rewards.
        - If $$\gamma=0$$, only the immediate reward is considered.
        - As we set $$\gamma$$ closer to 1, future rewards are given greater emphasis relative to the immediate reward.
        - Why discount future reward: Because in many cases, we prefer to obtain the reward sooner than later.
    - **Finiter Horizon Reward**: $$\sum_{i=0}^h r_{t+i}$$
        - Considers the undiscounted sum of rewards over a finite number *h* of steps.
    - **Average Reward**: $$\lim_{h\to\infty}\sum_{i=0}^h r_{t+i}$$
        - Considers the average reward per time step over the entire lifetime of the agent.

## Q-Learning

### Motivation

- Difficult to learn the function $$\pi^*: S\to A$$ directly, because the available training data doesn’t provide training examples of the form <s,a>.
- A possible learning strategy
    - Choose the optimal action in state s that maximizes the sum of the immdeiate reward plus the $$V^*$$ of the immediate successor state, discounted by $$\gamma$$: $$\pi^*(s)=\arg\max_a [r(s,a)+\gamma V^*(\delta (s,a))]$$
    - Requires the agent to have **perfect domain knowledge** of the immediate reward function *r* and the state transition function $$\delta$$ to calculate $$V^*$$. But it’s impractical in many RL cases.

### Q-function

- Define the evaluation function ***Q(s, a)*** as the reward received immediately upon executing action ***a*** from state ***s***, plus the value (discounted by $$\gamma$$) of following the optimal policy thereafter:
    - $$Q(s,a)\equiv r(s,a)+\gamma V^*(\delta (s,a))$$
- To learn the policy function $$\pi^*$$, $$\pi^*=\arg\max_a Q(s,a)$$
    - It shows that if the agent learns the Q function instead of the *V** function, it will be able to select optimal actions even when it has no knowledge of thefunctions r and $$\delta$$.
    - It need only consider each available action ***a*** in its current state ***s*** and choose the action that maximizes ***Q(s, a)***.
    - Why can one choose globally optimal action sequences by reacting repeatedly to the local values of Q for the current state? Part of the beauty of *Q* learning is that the evaluation function is defined to have precisely this property-the value of *Q* for the current state and action summarizes in a single number all the information needed to determine the discounted cumulative reward that will be gained in the future if action ***a*** is selected in state ***s***

### An algorithm for learning Q

- Learning the *Q* function corresponds to learning the optimal policy.
- The key problem is finding a reliable way to estimate training values for *Q ,* given only a sequence of immediate rewards *r* spread out over time. This can be accomplished through iterative approximation.
- $$V^*(s)=\max_{a'}Q(s,a')\implies Q(s,a)= r(s,a)+\gamma \max_{a'}Q(s,a')$$
    - Q maps history-action pairs to scalar estimates of their Q-value
- This recursive definition of Q provides the basis for algorithms that iteratively approximate Q.
- In this algorithm the learner represents its hypothesis $$\hat{Q}$$ (estimate of the actual ***Q***) by a large table with a separate entry for each state-action pair.
- Update rule: $$\hat{Q}(s,a)\gets r(s,a)+\gamma \max_{a'}\hat{Q}(s,a')$$
- Algorithm Pseudo Code
    
    ![Q-Learning Algo]({{ site.baseurl }}/assets/images/RL/RL1.png)
    

### Training Process

- Training consists of a series of ***episodes.***
    - Each episode starts at a random initial state and ends at the absorbing goal state.
    - With all the $$\hat{Q}$$ values initialized to zero, the agent will make no changes to any $$\hat{Q}$$ table entry until it happens to reach the goal state and receive a nonzero reward.
- Two properties of the algo
    - $$\hat{Q}$$ values never decrease during training.
    - Throughout the training process every Q value will remain in the interval between zero and its true Q value.

### Proof of Convergence

- Three conditions required for the convergence of $$\hat{Q}$$ table to the true Q function.
    - The system is a deterministic Markov-Decision Process.
    - The immediate reward values are bounded.
    - The agent selects actions in such a fashion that it visits every possible state-action pair infinitely often.
    - **Proof**
        
        ![Proof of Convergence]({{ site.baseurl }}/assets/images/RL/RL2.png)
        
- The **key idea** underlying the proof of convergence is that the table entry $$\hat{Q}(s,a)$$ with the largest error must have its error reduced by a factor of $$\gamma$$ whenever it is updated. The reason is that its new value depends only in part on error-prone $$\hat{Q}$$ estimates, with the remainder depending on the error-free observed immediate reward r.

### Experiment Strategy

- Problem (Exploration vs. Exploitation)
    - One obvious strategy would be for the agent in state ***s*** to select the action ***a*** that maximizes $$\hat{Q}(s,a)$$, thereby exploiting its current approximation $$\hat{Q}$$.
    - However, with this strategy the agent runs the risk that it will overcommit to actions that are found during early training to have high Q values, while failing to explore other actions that have even higher values. (Same problem with greedy algo)
    - In this way, the requirement that each state-action transition occur infinitely often required by convergence theorem will not be satisfied, thus failed to converge.
- Solution
    - Use a probabilistic approach to selecting actions.
    - Actions with higher $$\hat{Q}$$ values are assigned higher probabilities, but every action is assigned a **nonzero** probability.
    - $$P(a_i|s)=\frac{k^{\hat Q (s,a_i)}}{\sum_j k^{\hat Q (s,a_j)}}$$ or $$\frac{\hat{Q}(s,a_i)+1}{\sum_j \hat{Q}(s,a_j)+|j|}$$
        - ***k*** > **0** is a constant that determines how strongly the selection favors actions with high Q values.
        - Larger values of k will assign higher probabilities to actions with above average $$\hat{Q}$$, causing the agent to ***exploit*** what it has learned and seek actions it believes will maximize its reward.
        - In contrast, small values of k will allow higher probabilities for other actions, leading the agent to ***explore*** actions that do not currently have high $$\hat{Q}$$ values.
        - In some cases, k is varied with the number of iterations so that the agent favors **exploration** during **early** stages of learning, then gradually shifts toward a strategy of **exploitation**.
            - $$P(a_i|s)=(\frac{\hat{Q}(s,a_i)+1}{\sum_j \hat{Q}(s,a_j)+|j|})^{1+\frac{T}{N}}$$, T is the current episode, N is the max episode

### Updating Sequence

- Normal Process
    - If we begin with all *Q* values initialized to zero, then after the first full episode only
    one entry in the agent's *Q* table will have been changed. If we run repeated identical episodes in this fashion, the frontier of nonzero *Q* values will creep backward from the goal state at the rate of **one new state-action transition per episode.**
    - Slower to converge
- Strategy1
    - Apply the same update rule for each transition considered, but perform these updates in reverse order. ($$(s_n,a_n)\to (s_{n-1},a_{n-1}) \to \cdots (s_1,a_1)$$)
    - Converge in fewer iterations, and requires the agent use more memory to store the entire episode before beginning the training for that episode.
- Strategy2
    - A second strategy for improving the rate of convergence is to store past state-action transitions, along with the immediate reward that was received, and retrain on them periodically.
- A large number of efficient algorithms from the field of dynamic programming can be applied when the functions $$\delta$$ **and *r* are known.

## Nondeterministic Rewards and Actions

<aside>
💡 Use math expectation based on probability distribution instead of simple average.

</aside>

- Nondeterministic: the reward function *r(s ,a)* and action transition function $$\delta$$*(s ,a)* may have probabilistic outcomes.
- Nondeterministic Markov decision process: In such cases, the functions $$\delta$$*(s,a)* and *r(s,a)* can be viewed as first producing a probability distribution over outcomes based on *s* and $$\delta$$*,* and then drawing an outcome at random according to this distribution. When these probability distributions depend solely on *s* and *a* (e.g., they do not depend on previous states or actions), then we call the system a nondeterministic Markov decision process.
- Generalization to nondemeterministic
    - Discounted Cumulative Reward: $$V^{\pi}(s_t)=E[\sum_{i=0}^\infty r_{t+i}\cdot \gamma^i]$$, revise it to expected value
    - $$Q(s,a)\equiv E[r(s,a)+\gamma V^*(\delta (s,a))]=E[r(s,a)]+\gamma E[ V^*(\delta (s,a))]=E[r(s,a)]+\gamma \sum_{s'} P(s'|s,a)V^*(s')$$
    - Training rule
        - The previous rule fails to converge (repeatedly alter the $$\hat{Q}$$ value)
        - $$\hat{Q}_n(s,a)\gets (1-\alpha_n)\hat{Q}_{n-1}+\alpha_n[r+\gamma \max_{a'}\hat{Q}(s,a')],\;\alpha_n=\frac{1}{1+visits_n(s,a)}$$
            - *visits,(s, a)* is the total number of times this state-action pair has been visited up to and including the *nth* iteration.
            - By reducing *a* at an appropriate rate during training, we can achieve convergence
            to the correct *Q* function.

## Deeq Q-Learning

- Problem
    - Most successful RL applications that operate on domains like vision and speech have relied on hand-crafted features combined with linear value functions or policy representations.
    - Clearly, the performance of such systems heavily relies on the quality of the feature representation.
- Solution
    - Deep learning

### DQN

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
    - Use Deep Neural Network (Q-network) to approximate the Q-value function $$Q(s,a)$$
    - Use experience replay buffer to store the last K frames and take random samples from the buffer to train the Q-network