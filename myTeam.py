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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from baselineTeam import ReflexCaptureAgent
from captureAgents import CaptureAgent
from game import Directions
from operator import itemgetter
from capture import noisyDistance
import math


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
        # get opponents initial jail pos
        opponents = self.getOpponents(gameState)
        self.opponent_jail_pos = [gameState.getInitialAgentPosition(opponents[i]) for i in range(len(opponents))]
        # counter to initialize probability of being eaten in a state.
        opponent_probability = util.Counter()
        # legal positions our pacman could occupy
        legal_positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        print self.opponent_jail_pos
        self.legal_positions = legal_positions
        # initialize extra for jail positions (places where we have ghosts)
        for pos in self.opponent_jail_pos:
            opponent_probability[pos] += 2
            legal_positions.remove(pos)

        # initialize probability of opponent position uniformly elsewhere
        for pos in legal_positions:
            opponent_probability[pos] += 1
        opponent_probability.normalize()

        # check if it's on red Team
        self.opponent_position_distribution = opponent_probability

        # limit of food we're defending
        self.foodEaten = 0

        # have we started the game
        self._hasStarted = False

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        if not self._hasStarted:
            self._hasStarted = True


        '''
                PICK AN ACTION
        '''
        actions = gameState.getLegalActions(self.index)
        # check if we're in home territory
        # get pacman pos
        isOffense = gameState.getAgentState(self.index).isPacman
        # food list
        food_matrix = self.getFood(gameState)
        food_list = food_matrix.asList()
        remaining_food = [(x, y) for x, y in food_list if food_matrix[x][y]]
        food_ratio = 0.25 * len(remaining_food)
        # print len(remaining_food)
        # we are in home territory
        if self.foodEaten < math.ceil(food_ratio):
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
            # threat_weight = self.weighOpponentThreat(successor_state)
            opponents = self.getOpponents(successor_state)
            # for each opponent
            for oppIndex in opponents:
                opp_state = successor_state.getAgentState(oppIndex)
                # get the supposed agent index
                opp_position = opp_state.getPosition()
                # check if our reading returned anything
                if opp_position:
                    # get the noisy distance from that position
                    noisy_distance = noisyDistance(successor_pos, opp_position)
                    # if noisy distance is less than 2
                    if noisy_distance <= 2:
                        # increment the threat weight
                        print opp_state

            # list storing food distances
            food_distances = []
            # loop to get distance of all the food's in action
            for food in successor_foodList:
                # calculate the distance between food and position of pacman after action and food.
                distance = self.distancer.getDistance(food, successor_pos)
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

        # if we are on offense, evaluate danger then take action from the saved actions.

    def didWeDie(self, gameState):
        if self._hasStarted:
            agent_state = gameState.getAgentState(self.index)
            agent_position = agent_state.getPosition()
            if agent_position == self.initialAgentPos:
                print "Oh shit, here we go again"

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
        # check if the our agent is a ghost and our scared timer is on
        is_my_scared_timer_on = not is_agent_pacman and new_agent_state.scaredTimer > 0
        # constant opponent threat
        opponent_threat_weight = 1
        # get opponents
        opponents = self.getOpponents(successor_state)
        # for each opponent
        for oppIndex in opponents:
            opp_state = successor_state.getAgentState(oppIndex)
            # if there scared timer is on, threat is low
            threat_identifier = 1 if (is_agent_pacman and opp_state.scaredTimer <= 0) or is_my_scared_timer_on else -1
            # get the supposed agent index
            opp_position = opp_state.getPosition()
            # check if our reading returned anything
            if opp_position:
                # get the noisy distance from that position
                noisy_distance = noisyDistance(new_agent_position, opp_position)
                # if noisy distance is less than 2
                if noisy_distance <= 2:
                    # increment the threat weight
                    opponent_threat_weight += (100 * threat_identifier)

        return opponent_threat_weight


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
