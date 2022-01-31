# multiAgents.py
# --------------
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


from util import manhattanDistance
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        evf_result = successorGameState.getScore()
        food_distance = [manhattanDistance(newPos, i) for i in newFood.asList()]

        if action == "Stop":
            evf_result -= 150

        for i in range(len(newGhostStates)):
            if (newGhostStates[i].getPosition() == newPos and (newScaredTimes[i] == 0)) or \
                    util.manhattanDistance(newGhostStates[i].getPosition(), newPos) < 2:
                evf_result -= 100
        if len(food_distance) > 0:
            evf_result += 1 / min(food_distance)

        return evf_result


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[0]

    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isLose() or \
                gameState.isWin():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minvalue(gameState, depth)
        else:
            return self.maxvalue(gameState, depth)

    def minvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        min_result = "", float("Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1)
            if result[1] < min_result[1]:
                min_result = (action, result[1])
        return min_result

    def maxvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
        return max_result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, float("-Inf"), float("Inf"))[0]

    def value(self, gameState, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or \
                gameState.isWin() or\
                gameState.isLose():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minvalue(gameState, depth, alpha, beta)
        else:
            return self.maxvalue(gameState, depth, alpha, beta)

    def minvalue(self, gameState, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        min_result = "", float("Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1, alpha, beta)
            if result[1] < min_result[1]:
                min_result = (action, result[1])
            if min_result[1] < alpha:
                return min_result
            beta = min(beta, min_result[1])
        return min_result

    def maxvalue(self, gameState, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1, alpha, beta)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
            if max_result[1] > beta:
                return max_result
            alpha = max(alpha, max_result[1])
        return max_result


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[0]

    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or \
                gameState.isWin() or \
                gameState.isLose():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.expectvalue(gameState, depth)
        else:
            return self.maxvalue(gameState, depth)

    def expectvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        expect_value = 0

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        weight_probability = 1. / len(legal_actions)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1)
            expect_value += result[1] * weight_probability
        return "", expect_value

    def maxvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
        return max_result


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    ghost_position = currentGameState.getGhostPositions()[0]
    scared_timer = currentGameState.getGhostStates()[0].scaredTimer
    ghost_distance = manhattanDistance(ghost_position, pacman_position)
    food_position = currentGameState.getFood().asList()

    food_items = []
    ghost_near = 0

    for food in food_position:
        food_items.append(-1 * manhattanDistance(pacman_position, food))
    if not food_items:
        food_items.append(0)

    if ghost_distance == 0 and scared_timer == 0:
        ghost_near = -150
    elif scared_timer > 0:
        ghost_near = -1 / ghost_distance

    num_capsules = len(currentGameState.getCapsules())

    return currentGameState.getScore() + ghost_near + num_capsules + max(food_items)


# Abbreviation
better = betterEvaluationFunction
