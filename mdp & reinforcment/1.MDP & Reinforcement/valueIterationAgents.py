# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        num_iterations = self.iterations
        for iteration in range(num_iterations):
            values = util.Counter()  # before each iteration, copy values to not work with real one.
            for state in all_states:
                if self.mdp.isTerminal(state):
                    continue
                compute_value = util.Counter()  # make a dictionary for all possible values for each action
                for action in self.mdp.getPossibleActions(state):
                    each_action_value = self.computeQValueFromValues(state, action)
                    compute_value[action] = each_action_value
                values[state] = max(compute_value.values())
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, probability in transitions:
            reward = self.mdp.getReward(state, action, next_state)
            q_value += probability * (reward + (self.discount * self.values[next_state]))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if self.mdp.isTerminal(state):
        #     return None
        actions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()
        if self.mdp.isTerminal(state):
            return
        for action in actions:
            q_values[action] = self.computeQValueFromValues(state, action)
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if self.mdp.isTerminal(state):
                continue
            one_state = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                one_state[action] = self.computeQValueFromValues(state, action)
            self.values[state] = max(one_state.values())


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        priority_queue = util.PriorityQueue()
        predecessor = {}

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            q_values = {}
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in predecessor:
                        predecessor[next_state] = set()
                    predecessor[next_state].add(state)
                q_values[action] = self.getQValue(state, action)
            diff = abs(self.values[state] - max(q_values.values()))
            priority_queue.update(state, -diff)

        for iteration in range(self.iterations):
            if priority_queue.isEmpty():
                break
            state = priority_queue.pop()
            if self.mdp.isTerminal(state):
                continue
            update_values = {}
            for action in self.mdp.getPossibleActions(state):
                update_values[action] = self.getQValue(state, action)
            self.values[state] = max(update_values.values())
            for p in predecessor[state]:
                q_values = {}
                for action in self.mdp.getPossibleActions(p):
                    q_values[action] = self.getQValue(p, action)
                diff = abs(self.values[p] - max(q_values.values()))
                if diff > self.theta:
                    priority_queue.update(p, -diff)
