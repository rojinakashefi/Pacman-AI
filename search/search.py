# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
       Search the deepest nodes in the search tree first.

       Your search algorithm needs to return a list of actions that reaches the
       goal. Make sure to implement a graph search algorithm.

       To get started, you might want to try some of these simple commands to
       understand the search problem that is being passed in:

       print("Start:", problem.getStartState())
       print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
       print("Start's successors:", problem.getSuccessors(problem.getStartState()))
       """
    "*** YOUR CODE HERE ***"
    return search_without_cost(problem, fringe=util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return search_without_cost(problem, fringe=util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return search_with_cost(problem)


def nullHeuristic(state, problem=None):
    """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return search_with_cost(problem, heuristic)


def search_without_cost(problem, fringe):
    closed = list()
    # each state has current position and paths
    start_node = (problem.getStartState(), [])
    fringe.push(start_node)
    while not fringe.isEmpty():
        current_state = fringe.pop()
        # first we check if the current state is goal or not
        if problem.isGoalState(current_state[0]):
            return current_state[1]
        # check if we have checked the current state before or not
        if current_state[0] not in closed:
            # if not add it to closed list
            closed.append(current_state[0])
            # we get current position successors
            # get successor return 3 information : next state coordination , action , cost
            current_state_successors = problem.getSuccessors(current_state[0])
            # for all successors available for current state
            # check if it successors state has been checked before or not
            for state in current_state_successors:
                if state[0] not in closed:
                    # if we have a new state we add it stack
                    # fringe contain new state location and the path till that
                    fringe.push((state[0], current_state[1] + [state[1]]))
    return None


def search_with_cost(problem, heuristic=nullHeuristic):
    # we use priority queue for both usc and a star algorithm
    # cause cost is important for both of them
    # in a star in addition of cost , heuristic is important
    fringe = util.PriorityQueue()
    closed = list()
    # each state contain two things current state and the path
    start_node = (problem.getStartState(), [])
    # because fringe is priority queue
    # we should give it its priority
    # which is the cost to that state
    fringe.push(start_node, 0)
    while not fringe.isEmpty():
        current_state = fringe.pop()
        if problem.isGoalState(current_state[0]):
            return current_state[1]
        if current_state[0] not in closed:
            closed.append(current_state[0])
            current_state_successors = problem.getSuccessors(current_state[0])
            for state in current_state_successors:
                if state[0] not in closed:
                    path = current_state[1] + [state[1]]
                    # if heuristic is null which it returns 0
                    # it means it is usc
                    if heuristic == nullHeuristic:
                        fringe.push((state[0], current_state[1] + [state[1]]), problem.getCostOfActions(path))
                    else:
                        # if not beside cost to that state we should consider it heuristic
                        f_calculation = problem.getCostOfActions(path) + heuristic(state[0], problem)
                        fringe.push((state[0], current_state[1] + [state[1]]), f_calculation)
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
