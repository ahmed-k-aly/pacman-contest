# myTeam.py
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


import random
import util

from captureAgents import CaptureAgent
from game import Directions


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='FoodSearchAgent', second='FoodSearchAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)


class FoodSearchAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

        # go thru each state and initialize q-values in each cell.
        actions = gameState.getLegalActions()

        # pacman position.
        # find the matrix
        food_matrix = self.getFood(gameState)
        height = food_matrix.height
        width = food_matrix.width
        half_width = width / 2 - 1
        # list containing places where we deem ourselves safe, (x, y) safety and back on def.
        # (x+1, y) hunting, back to offense
        openings = [(half_width, i) for i in range(height)]
        self.openings = openings
        self.problem = FindOpeningProblem(gameState, self.index, openings)
        self.search = Search(self.problem)


        # # successor = self.getSuccessor(gameState, action)
        # #
        # #     myState = successor.getAgentState(self.index)
        # #     myPos = myState.getPosition()
        # state_values = util.Counter()
        # values = list()
        # for action in actions:
        #     # get successor state, (calculate the nearest food from where we are)
        #     successor = my_state.getSuccessor(gameState, action)
        #     # food list
        #     food_list = self.getFood(gameState).asList()
        #     # get distance to the nearest food.
        #     myPos = successor.getAgentState(self.index).getPosition()
        #     # get pos in food list
        #     minDistance = min([self.getMazeDistance(myPos, food) for food in food_list])
        #     # basically don't consider where there is no food? Ideally we need not to check this case.
        #     if minDistance != 0:
        #         values.append((action, 1 / minDistance))
        #
        # state_values[position] = values
        #
        # # make the action, q-value pair for each state in the board.
        # # down play when it's a corner
        # # in-corporate paths and distances back to home base.
        # # sonar readings as well.
        available_actions = gameState.getLegalActions(self.index)
        actions = self.search.aStarSearch(available_actions)

        self.counter = 0
        self.actions = actions

    def chooseAction(self, gameState):
        action = self.actions[self.counter]
        self.counter += 1
        return action

class FindOpeningProblem:

    def __init__(self, gameState, agentIndex, openings):
        self.gameState = gameState
        self.index = agentIndex
        self.openings = openings

    def getStartState(self):
        return self.gameState.getAgentPosition(self.index)

    def isGoalState(self, state):
        return state in self.openings

    def getSuccessors(self, actions):
        walls = self.gameState.getWalls()
        successors = list() # list of successors
        for action in actions: # for action in actions
            successor_state = self.gameState.generateSuccessor(self.index, action)
            new_pacman_pos = successor_state.getAgentPosition(self.index)
            isPacmanInWall = new_pacman_pos in walls
            print " Action " + str(action) + " will lead us to wall? " + str(isPacmanInWall)
            if not isPacmanInWall:
                successors.append((new_pacman_pos, action, 1))

        return successors


class Search:

    def __init__(self, problem):
        self.problem = problem

    def nullHeuristic(state, position=None):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def aStarSearch(self, actions, heuristic=nullHeuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE ***"

        "Our frontier is the queue containing subsequent problems"
        frontier = util.PriorityQueue()
        "We keep a set of explored nodes"
        explored_nodes = set()
        "Initialize the initial node into the stack"
        start_state = self.problem.getStartState()
        "solution"

        """
         total cost is the sum of initial cost associated with node, which is 0
          added to the cost associated with the heuristic
        """
        total_cost = 0 + heuristic(start_state)
        # a node on the frontier has it's state, directions leading up to it, a cost associated with moving up to that.
        frontier_node = (start_state, [], 0)
        frontier.push(frontier_node, total_cost)

        while not frontier.isEmpty():
            problem_node = frontier.pop()

            cur_state, action_paths, cost = problem_node

            if self.problem.isGoalState(cur_state):
                print action_paths
                return action_paths

            # if current state is not in explored nodes
            if cur_state not in explored_nodes:
                # get successors from the problem using our cur state
                successors = self.problem.getSuccessors(actions)
                # add cur state to explored nodes
                explored_nodes.add(cur_state)
                # for each successor in successors
                # unpack successor to get pos, directions, and path cost
                # copy list of prev actions to new actions
                # add the direction of that successor to the new actions
                # update the path cost
                # push to the frontier.
                for successor in successors:
                    pos, direction, path_cost = successor
                    new_actions = action_paths[:]
                    new_actions.append(direction)
                    # get a cost associated with the node
                    node_cost = cost + path_cost
                    # get the total cost (cost associated with node + heuristic cost) for the frontier priority queue
                    new_total_cost = node_cost + heuristic(pos)
                    # frontier node has just it's node cost (doesn't include cost associated with heuristic)
                    new_frontier_node = (pos, new_actions, node_cost)
                    # in the frontier, cost includes heuristic cost
                    frontier.push(new_frontier_node, new_total_cost)
