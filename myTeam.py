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
from baselineTeam import ReflexCaptureAgent

from captureAgents import CaptureAgent
from game import Directions
from operator import itemgetter
from capture import noisyDistance


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DefensiveReflexAgent'):
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

        # initial agent pos
        self.initialAgentPos = gameState.getInitialAgentPosition(self.index)
        # check if it's on red Team

        # limit of food we're defending
        self.foodEaten = 0

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        '''
                PICK AN ACTION
        '''
        actions = gameState.getLegalActions(self.index)
        # check if we're in home territory
        # get pacman pos
        isOffense = gameState.getAgentState(self.index).isPacman
        # we are in home territory
        if self.foodEaten < 5:
            return self.approachFoodAction(gameState, actions)
        else:
            return self.defendFoodAction(gameState, actions)

    def approachFoodAction(self, gameState, actions):
        # TODO:
        #  1. replace with A-star search
        #  2. avoid ghosts.
        #  3. predict position of ghosts.
        # food action pairs
        action_food_distance_list = []
        # for each action
        for action in actions:
            '''
                SUCCESSOR STATE AND ALL IT HAS TO OFFER US (FOOD LIST, AGENT POSITION)
            '''
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # food list pacman is supposed to choose.
            successor_foodList = self.getFood(successor_state).asList()
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            new_agent_position = agent_state.getPosition()

            '''
                    FOOD DISTANCES TO DETERMINE OUR ACTION
            '''

            # check if we have food nearby
            food_positions = self.getFood(gameState)  # list of game states: self.getFood(gameState).asList()
            food_list = food_positions.asList()
            # EAT FOOD IF NEARBY
            if new_agent_position in food_list:
                self.foodEaten += 1
                return action
            # list storing food distances
            food_distances = []
            # loop to get distance of all the food's in action
            for food in successor_foodList:
                # calculate the distance between food and position of pacman after action and food.
                distance = self.distancer.getDistance(food, new_agent_position)
                # add to list of food distances
                food_distances.append(distance)
            # action and distance to nearest food list
            action_food_distance = (action, min(food_distances))
            # all legal actions (action, food distance) list
            action_food_distance_list.append(action_food_distance)

        if not action_food_distance_list:
            "Oops, We can't take any action"
            return None
        return min(action_food_distance_list, key=itemgetter(1))[0]

    def defendFoodAction(self, gameState, actions):
        # TODO:
        #  1. replace with A-star search
        #  2. avoid ghosts.
        #  3. predict position of ghosts.
        # food action pairs
        action_food_distance_list = []
        # for each action
        for action in actions:
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # food list pacman is supposed to choose.
            successor_foodList = self.getFoodYouAreDefending(successor_state).asList()
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            successor_pos = agent_state.getPosition()
            # check if we have food nearby
            # get all food positions we are to eat.
            # food_positions = self.getFood(gameState)  # list of game states: self.getFood(gameState).asList()
            # food_list = food_positions.asList()
            # if is pacman, reset the food count to 0
            if not agent_state.isPacman:
                self.foodEaten = 0
                # if successor_pos in food_list:
            #     return action
            # threat weight
            threat_weight = self.weighOpponentThreat(successor_state)
            # list storing food distances
            food_distances = []
            # loop to get distance of all the food's in action
            for food in successor_foodList:
                # calculate the distance between food and position of pacman after action and food.
                distance = self.distancer.getDistance(food, successor_pos)
                # add to list of food distances
                food_distances.append(threat_weight * distance)
            # action and distance to nearest food list
            action_food_distance = (action, min(food_distances))
            # all legal actions (action, food distance) list
            action_food_distance_list.append(action_food_distance)

        if not action_food_distance_list:
            "Oops, We can't take any action"
            return None
        return min(action_food_distance_list, key=itemgetter(1))[0]

        # if we are on offense, evaluate danger then take action from the saved actions.

    # method to define how opponent affects us (-ve opponent is no threat and therefore hunt them, +ve opponent is high threat so run)
    def weighOpponentThreat(self, successor_state):
        """
            WEIGH THREAT BASED ON NOISY DISTANCES OF OPPONENTS
        """
        # get agent state so we can work
        new_agent_state = successor_state.getAgentState(self.index)
        # get the new agent position
        new_agent_position = new_agent_state.getPosition()
        # check if the agent is pacman
        is_agent_pacman = new_agent_state.isPacman
        # check if the agent is pacman and we're not a ghost
        is_my_scared_timer_on = not is_agent_pacman and new_agent_state.scaredTimer > 0
        # constant opponent threat
        opponent_threat_weight = 1
        threat_identifier = 1 if is_agent_pacman or is_my_scared_timer_on else -1
        # get opponents
        opponents = self.getOpponents(successor_state)
        # for each opponent
        for oppIndex in opponents:
            opp_state = successor_state.getAgentState(oppIndex)
            # if there scared timer is on, threat is low
            threat_identifier = -1 if opp_state.scaredTimer > 0 else 1
            # get the supposed agent index
            opp_position = opp_state.getPosition()
            # check if our reading returned anything
            if opp_position:
                # get the noisy distance from that position
                noisy_distance = noisyDistance(new_agent_position, opp_position)
                # if noisy distance is less than 2
                if noisy_distance <= 2:
                    # increment the threat weight
                    opponent_threat_weight += 100

        actual_threat = opponent_threat_weight * threat_identifier
        return threat_identifier * opponent_threat_weight


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


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



