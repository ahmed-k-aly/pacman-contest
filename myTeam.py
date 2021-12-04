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
from operator import itemgetter
from baselineTeam import ReflexCaptureAgent
from capture import noisyDistance
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='FoodGreedyDummyAgent', second='HuntGhostDummyAgent'):
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

class Inference:
    def __init__(self, gameState, index, opponentIndex, numParticles=600):
        # TODO: Use info from both my agents to infer
        self.numParticles = numParticles
        self.myIndex = index
        self.opponentIndex = opponentIndex
        self.legalPositions = self.getLegalPositions(gameState)

    def getLegalPositions(self, gameState):
        '''
        Method that returns all legal positions for a given gameState
        #TODO: implement caching since legal positions are the same throughout the game
        '''
        walls = gameState.getWalls()
        legalPositions = []
        for line in walls:
            row = [not a for a in line]
            legalPositions.append(row)

        legalPositionsAsList = []
        for x in range(len(legalPositions)):
            for y in range(len(legalPositions[x])):
                position = (x, y)
                if legalPositions[x][y]:
                    legalPositionsAsList.append(position)
        return legalPositionsAsList

    def initializeUniformly(self):
        numLegalPositions = len(self.legalPositions)
        numParticlesPerPosition = self.numParticles // numLegalPositions
        # for each position, assign numParticles/numLegalPositions particles
        self.particleList = []
        for position in self.legalPositions:
            particle = position
            self.particleList += [particle] * numParticlesPerPosition

    def observe(self, gameState):
        agentIndex = self.opponentIndex
        enemyPos = gameState.getAgentPosition(agentIndex)
        if enemyPos is not None:
            self.particleList = [enemyPos]
            return
        noisyDistance = gameState.getAgentDistances()[agentIndex]
        myPosition = gameState.getAgentPosition(self.myIndex)

        beliefs = self.getBeliefDistribution()  # get particles belief  distribution
        allPossible = util.Counter()
        for particle in self.particleList:  # loop over all legal positions
            trueDistance = util.manhattanDistance(particle, myPosition)
            if gameState.getDistanceProb(trueDistance, noisyDistance) > 0:
                allPossible[particle] = beliefs[particle] * gameState.getDistanceProb(trueDistance,
                                                                                      noisyDistance)  # updates belief value for a position
        allPossible.normalize()
        beliefs = allPossible  # update the new belief state
        if beliefs.totalCount() == 0:  # if all weights are zero, initialize from the prior
            self.initializeUniformly()
            beliefs = self.getBeliefDistribution()

        newParticleList = []
        for _ in range(self.numParticles):
            particle = util.sample(beliefs, self.particleList)  # sample particle from the distribution
            newParticleList.append(particle)
        self.particleList = newParticleList  # update the particle list

    def elapseTime(self):
        newParticleList = []
        for particle in self.particleList:
            newPosDist = self.getPositionDistribution(particle)
            newParticle = util.sample(newPosDist)
            newParticleList.append(newParticle)
        self.particleList = newParticleList

    def getPositionDistribution(self, particle):
        counter = util.Counter()
        possibleFuturePos = self.getSuccessorPositions(particle)
        for pos in possibleFuturePos:
            counter[pos] = 1.0 / len(possibleFuturePos)
        return counter

    def getSuccessorPositions(self, position):
        legalPos = [position]
        x, y = position
        if (x + 1, y) in self.legalPositions:
            legalPos.append(((x + 1), y))
        if (x - 1, y) in self.legalPositions:
            legalPos.append(((x - 1, y)))
        if (x, y + 1) in self.legalPositions:
            legalPos.append(((x, y + 1)))
        if (x, y - 1) in self.legalPositions:
            legalPos.append(((x, y - 1)))
        return legalPos

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        "*** YOUR CODE HERE ***"

        counter = util.Counter()
        for particle in self.particleList:
            counter[particle] += 1
        counter.normalize()
        return counter


class HuntGhostDummyAgent(CaptureAgent):
    '''
        AGENT TO SERIALLY BUST GHOSTS
    '''

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
        self.opponents = opponents
        # distributions of belief of position for each opponent
        self.opp_beliefs = [util.Counter() for _ in range(len(opponents))]

        # calculate the mid pos of the board
        xOpp, YOpp = min(self.opponent_jail_pos)
        xInit, YInit = self.initialAgentPos
        self.mid = abs(xOpp - xInit)

        self.beliefsCounter = util.Counter()

        # legal positions our pacman could occupy
        legal_positions = self.getLegalPositions(gameState)
        for jail in self.opponent_jail_pos:
            # add jail if not already there bc some weird ass bug
            if jail not in legal_positions:
                legal_positions.append(jail)

        self.legal_positions = legal_positions
        num_partices = 400
        self.inference1 = Inference(gameState, self.index, opponents[0], num_partices)
        self.inference1.initializeUniformly()
        self.inference2 = Inference(gameState, self.index, opponents[1], num_partices)
        self.inference2.initializeUniformly()

        self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution()

        # for alpha-beta pruning
        self.depth = 2

    def getLegalPositions(self, gameState):
        '''
        Method that returns all legal positions for a given gameState
        #TODO: implement caching since legal positions are the same throughout the game
        '''
        walls = gameState.getWalls()
        legalPositions = []
        for line in walls:
            row = [not a for a in line]
            legalPositions.append(row)

        legalPositionsAsList = []
        for x in range(len(legalPositions)):
            for y in range(len(legalPositions[x])):
                position = (x, y)
                if legalPositions[x][y]:
                    legalPositionsAsList.append(position)
        return legalPositionsAsList

    def chooseAction(self, gameState):
        '''INFERENCE'''

        self.inference1.observe(gameState)
        self.inference1.elapseTime()
        beliefs1 = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[0]] = beliefs1

        self.inference2.observe(gameState)
        self.inference2.elapseTime()

        beliefs2 = self.inference2.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = beliefs2

        self.agents = [self.index]
        for opp in self.getOpponents(gameState):
            if gameState.getAgentPosition(opp):
                self.agents.append(opp)

        '''
            MINIMAX IT IF WE OUR SCARED TIMER IS ON
        '''
        # get agent state
        agent_state = gameState.getAgentState(self.index)
        # is in danger
        is_in_danger = not agent_state.isPacman and agent_state.scaredTimer > 0
        # if we are ghost and our timer is on, minimax
        if is_in_danger:
            self.turns = len(self.agents)  # number of turns for each depth of a-b-search
            currTurn = 0
            agentIndex = self.agents[currTurn]
            return self.alphaBetaSearch(gameState, 1, agentIndex, currTurn)

        return self.huntGhostAction(gameState)

    def huntGhostAction(self, gameState):
        # self.displayDistributionsOverPositions(beliefsList)

        # is in home territory
        # our scared timer is off
        # minimize the distance between us and ghosts if we can see them, us and region of ghost's if we can see them
        '''
            PROBABILITY OF GHOSTS IN OUR HOME AREA
        '''
        # get the probability distribution of ghosts in our home territory
        home_territory_opp_distribution = util.Counter()
        for opponent in self.opponents:
            # get the belief distribution
            belief_distribution = self.beliefsCounter[opponent]
            for pos in self.legal_positions:
                # if pos is in home territory, add to our distribution
                if self.isInHomeTerritory(pos):
                    home_territory_opp_distribution[pos] += belief_distribution[pos]

        # normalize the belief distribution
        home_territory_opp_distribution.normalize()
        # take action that minimizes our distance to a position
        '''
            TAKING ACTIONS
        '''
        # get actions
        actions = gameState.getLegalActions(self.index)
        action_dist_to_ghost = []

        for action in actions:
            '''
                successor state and all it has to offer
            '''
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            new_agent_position = agent_state.getPosition()
            # distance between our new agent pos and highest prob
            dist_to_ghost = self.distancer.getDistance(new_agent_position, home_territory_opp_distribution.argMax())
            # append action and dist to ghost that remain in home territory
            if not agent_state.isPacman:
                action_dist_to_ghost.append((action, dist_to_ghost))
        return min(action_dist_to_ghost, key=itemgetter(1))[0]

    def isInHomeTerritory(self, pos):
        x, y = pos
        x1, y1 = self.initialAgentPos

        # middle x distance > x distance of pos - x distance of initial pos
        return self.mid > abs(x - x1)

    def evaluationFunction(self, gameState):
        agent_state = gameState.getAgentState(self.index)
        is_agent_pacman = agent_state.isPacman
        pacman_pos = agent_state.getPosition()
        # check if the our agent is a ghost and our scared timer is on
        is_my_scared_timer_on = not is_agent_pacman and agent_state.scaredTimer > 0
        # need a much better evaluation function for this Agent to perform decently
        # TODO: REPLACE ME
        opponents = self.getOpponents(gameState)
        # for each opponent
        totalNoisyDistance = 0
        for oppIndex in opponents:
            opp_state = gameState.getAgentState(oppIndex)
            # get the supposed agent index
            opp_position = opp_state.getPosition()
            # threat identifier is a -1 if a state is bad for us (since it'll be multiplied by -1 at the end) and positive if state is good for us
            threat_identifier = -1 if (is_agent_pacman and opp_state.scaredTimer <= 0) or is_my_scared_timer_on else 1
            # check if our reading returned anything
            if opp_position:
                # get the noisy distance from that position
                noisy_distance = noisyDistance(gameState.getAgentState(self.index).getPosition(), opp_position)
                # if noisy distance is less than 2
                num_walls = 1 if self.num_walls[pacman_pos] == 0 else self.num_walls[pacman_pos]
                num_walls_eval = num_walls * 100
                totalNoisyDistance += (noisy_distance)
        return -totalNoisyDistance

    def maxValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MAX's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - in this implementation, MAX can only be our agent.
        NOTE: If we wanted to include our teammate as the other MAX player,
        this method would need to be slightly changed
        currTurn - In this implementation, currTurn in this method is always 0.
        This is because our agent's turn is always first.
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:  # max depth exceeded
            return self.evaluationFunction(gameState)

        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]  # whose turn is next
            v = max((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v > beta:
                # prune
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MIN's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - player whose turn it is right now. Can be any of MAX's opponents
        currTurn - currTurn + 1 is used to determine who's playing next
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        if currTurn == self.turns - 1:
            # if last Agent of ply, call maxAgent to play
            nextTurn = 0
            nextAgentIndex = self.agents[nextTurn]  # whose turn it is next
            for action in legalActions:
                v = min(v,
                        self.maxValue(gameState.generateSuccessor(agentIndex, action), d + 1, nextAgentIndex, nextTurn,
                                      alpha, beta))
                if v < alpha:
                    # prune
                    return v
                beta = min(beta, v)
            return v

        # else, call another minAgent to play
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]
            v = min((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v < alpha:
                # prune
                return v
            beta = min(beta, v)
        return v

    def alphaBetaSearch(self, gameState, d, agentIndex, currTurn):
        """
        Runs alpha-beta-search algorithm. Starts with MAX player.
        PARAM:
        gameState
        d - current depth, set to 0
        agentIndex - player whose turn it is right now; to start, our agent's index
        currTurn - current Turn, starts with 0
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """

        alpha = float("-inf")
        beta = float("inf")
        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        previousV = float("-inf")  # used for comparisons in determining the bestAction
        bestAction = None
        for action in legalActions:
            if currTurn < len(self.agents) - 1:
                nextAgentIndex = self.agents[currTurn + 1]  # whose turn it is next
                v = max(v,
                        self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta))
                if v > previousV:
                    # compares every action value to return the best Action
                    bestAction = action
                if v >= beta:
                    return bestAction
                alpha = max(alpha, v)
                previousV = v
        return bestAction  # basically, a complicated argmax


class HuntFoodDummyAgent(CaptureAgent):
    '''
        AGENT TO SERIALLY BUST GHOSTS
    '''

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
        # food eaten
        self.foodEaten = 0
        # initial agent pos
        self.initialAgentPos = gameState.getInitialAgentPosition(self.index)
        # get opponents initial jail pos
        opponents = self.getOpponents(gameState)
        self.opponent_jail_pos = [gameState.getInitialAgentPosition(opponents[i]) for i in range(len(opponents))]
        self.opponents = opponents
        # distributions of belief of position for each opponent
        self.opp_beliefs = [util.Counter() for _ in range(len(opponents))]

        # calculate the mid pos of the board
        xOpp, YOpp = min(self.opponent_jail_pos)
        xInit, YInit = self.initialAgentPos
        self.mid = abs(xOpp - xInit)

        self.beliefsCounter = util.Counter()

        # legal positions our pacman could occupy
        legal_positions = self.getLegalPositions(gameState)
        for jail in self.opponent_jail_pos:
            # add jail if not already there bc some weird ass bug
            if jail not in legal_positions:
                legal_positions.append(jail)

        self.legal_positions = legal_positions
        num_partices = 400
        self.inference1 = Inference(gameState, self.index, opponents[0], num_partices)
        self.inference1.initializeUniformly()
        self.inference2 = Inference(gameState, self.index, opponents[1], num_partices)
        self.inference2.initializeUniformly()

        self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution()

        # for alpha-beta pruning
        self.depth = 2

    def getLegalPositions(self, gameState):
        '''
        Method that returns all legal positions for a given gameState
        #TODO: implement caching since legal positions are the same throughout the game
        '''
        walls = gameState.getWalls()
        legalPositions = []
        for line in walls:
            row = [not a for a in line]
            legalPositions.append(row)

        legalPositionsAsList = []
        for x in range(len(legalPositions)):
            for y in range(len(legalPositions[x])):
                position = (x, y)
                if legalPositions[x][y]:
                    legalPositionsAsList.append(position)
        return legalPositionsAsList

    def chooseAction(self, gameState):

        '''INFERENCE'''

        self.inference1.observe(gameState)
        self.inference1.elapseTime()
        beliefs1 = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[0]] = beliefs1

        self.inference2.observe(gameState)
        self.inference2.elapseTime()

        beliefs2 = self.inference2.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = beliefs2
        beliefsList = [beliefs1, beliefs2]

        self.agents = [self.index]
        agent_state = gameState.getAgentState(self.index)
        pacman_pos = agent_state.getPosition()
        min_dist_to_opponent = []
        for opp in self.getOpponents(gameState):
            # get agent state
            opp_state = gameState.getAgentState(opp)
            # we can see opponent, we are pacman, and the scared timer is on
            if gameState.getAgentPosition(opp) and agent_state.isPacman and opp_state.scaredTimer <= 0:
                self.agents.append(opp)
                # append distance
                min_dist_to_opponent.append(self.distancer.getDistance(pacman_pos, opp_state.getPosition()))

        dist_to_opponent = 2 if not min_dist_to_opponent else min(min_dist_to_opponent)

        '''IF WE ARE IN DANGER OR WE'VE COLLECTED HALF THE FOOD, START GOING HOME'''
        food_matrix = self.getFood(gameState)
        food_list = food_matrix.asList()
        remaining_food = [(x, y) for x, y in food_list if food_matrix[x][y]]
        acceptable_food_ratio = 0.25 * len(remaining_food)
        # print len(remaining_food)
        # if we are in home territory, reset food to 0
        if self.isInHomeTerritory(pacman_pos):
            self.foodEaten = 0

        # we are in home territory
        # high threat
        # offense
        # 1. we've eaten enough food and we good, go home
        # 2. if we have food in us, and there's an agent nearby
        # if self.foodEaten >= math.ceil(acceptable_food_ratio) or self.foodEaten >= 0 and len(self.agents) > 1:
        #     # go home, how minimize the distance between (xPacmanPos - xHomePos)
        #     return self.goHome(gameState)

        if self.foodEaten <= math.ceil(acceptable_food_ratio) and len(self.agents) == 1:
            return self.huntFoodAction(gameState)

        #if agent_state.isPacman and len(self.agents) > 1 and dist_to_opponent < 2:
        self.turns = len(self.agents)  # number of turns for each depth of a-b-search
        currTurn = 0
        agentIndex = self.agents[currTurn]
        return self.alphaBetaSearch(gameState, 1, agentIndex, currTurn)

        # return self.alphaBetaSearch(gameState)

        # if agent_state.isPacman and len(self.agents) > 1 and dist_to_opponent < 2:
        #     self.turns = len(self.agents)  # number of turns for each depth of a-b-search
        #     currTurn = 0
        #     agentIndex = self.agents[currTurn]
        #     return self.alphaBetaSearch(gameState, 1, agentIndex, currTurn)
        #
        # # return self.huntFoodAction(gameState)

    def goHome(self, gameState):

        # get actions
        actions = gameState.getLegalActions(self.index)
        action_home_dist = []

        for action in actions:
            '''
                successor state and all it has to offer
            '''
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            new_pos = agent_state.getPosition()
            dist_to_initial_pos = self.distancer.getDistance(new_pos, self.initialAgentPos)
            action_home_dist.append((action, dist_to_initial_pos))

        return min(action_home_dist, key=itemgetter(1))[0]

    def huntFoodAction(self, gameState):
        #self.displayDistributionsOverPositions(beliefsList)

        # is in home territory
        # our scared timer is off
        # minimize the distance between us and ghosts if we can see them, us and region of ghost's if we can see them
        '''
            PROBABILITY OF GHOSTS IN OUR OPPONENT AREA
        '''
        # get the probability distribution of ghosts in our home territory
        opponent_territory_distribution = util.Counter()
        for opponent in self.opponents:
            # get the belief distribution
            belief_distribution = self.beliefsCounter[opponent]
            for pos in self.legal_positions:
                # if pos is in not home territory, add to our distribution
                if not self.isInHomeTerritory(pos):
                    opponent_territory_distribution[pos] += belief_distribution[pos]

        # normalize the belief distribution
        opponent_territory_distribution.normalize()
        # take action that minimizes our distance to a position
        '''
            TAKING ACTIONS
        '''
        # get actions
        actions = gameState.getLegalActions(self.index)
        action_dist_to_safe_food = []

        for action in actions:
            '''
                successor state and all it has to offer
            '''
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            new_agent_position = agent_state.getPosition()
            # food pos with lowest probability of ghost

            # check if we have food nearby
            food_positions = self.getFood(gameState)  # list of game states: self.getFood(gameState).asList()
            food_list = food_positions.asList()
            # EAT FOOD IF NEARBY
            if new_agent_position in food_list:
                self.foodEaten += 1
                return action

            food_ghost_probability = []
            successor_foodList = self.getFood(gameState).asList()
            # loop to get distance of all the food's in action
            for food in successor_foodList:
                # append to our (food, Ghost Prob)
                food_ghost_probability.append((food, opponent_territory_distribution[food]))

            # get food places where the probability of ghosts is 0
            # list storing food distances and action probability
            food_distances = []
            for food, ghost_prob in food_ghost_probability:
                # calculate the distance between food and position of pacman after action and food.
                distance = self.distancer.getDistance(food, new_agent_position)
                # calculate the distance between food and argmax of distribution (most likely food distribution)
                opp_food_dist = self.distancer.getDistance(food, opponent_territory_distribution.argMax())

                # add to list of food distances
                food_distances.append(distance * opp_food_dist)
                # if ghost_prob <= 0.1:

            action_dist_to_safe_food.append((action, min(food_distances)))
        return min(action_dist_to_safe_food, key=itemgetter(1))[0]

    def isInHomeTerritory(self, pos):
        x, y = pos
        x1, y1 = self.initialAgentPos

        # middle x distance > x distance of pos - x distance of initial pos
        return self.mid > abs(x - x1)

    def evaluationFunction(self, gameState):
        agent_state = gameState.getAgentState(self.index)
        pacman_pos = agent_state.getPosition()
        # need a much better evaluation function for this Agent to perform decently
        # TODO: REPLACE ME
        opponents = self.getOpponents(gameState)
        # for each food
        # 1. calculate distance between pacman and the food item
        # 2. calculate distance between ghost and the food item (use belief.argMax()) if we can't see the ghost
        # check if we have food nearby
        food_positions = self.getFood(gameState)  # list of game states: self.getFood(gameState).asList()
        food_list = food_positions.asList()
        # pacman_to_food / min_ghost_to_food
        food_distances_ratio = []
        # calculate distance to food
        for food in food_list:
            # distance from pacman to the food
            food_distance_from_pacman = self.distancer.getDistance(pacman_pos, food)  # distance to food
            # distance from food to ghost
            # for each opponent, calculate distance to ghost (use the min distance)
            food_to_opp_distances = []
            for oppIndex in opponents:
                opp_to_food_distance = 0
                opp_state = gameState.getAgentState(oppIndex)
                # get the supposed agent index
                opp_position = opp_state.getPosition()
                # check if our reading returned anything
                if opp_position and opp_state.scaredTimer > 0:
                    # we can see opponent, calculate distance between opponent and food
                    opp_to_food_distance = self.distancer.getDistance(opp_position, food)
                    food_to_opp_distances.append(opp_to_food_distance)
                else:
                    # we cant see ghost, our highest belief and the food
                    opp_to_food_distance = self.distancer.getDistance(self.beliefsCounter[oppIndex].argMax(), food)
                    food_to_opp_distances.append(opp_to_food_distance)
            min_opp_to_food = min(food_to_opp_distances)
            food_distances_ratio.append(food_distance_from_pacman/min_opp_to_food)

            return max(food_distances_ratio)



    def maxValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MAX's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - in this implementation, MAX can only be our agent.
        NOTE: If we wanted to include our teammate as the other MAX player,
        this method would need to be slightly changed
        currTurn - In this implementation, currTurn in this method is always 0.
        This is because our agent's turn is always first.
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:  # max depth exceeded
            return self.evaluationFunction(gameState)

        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]  # whose turn is next
            v = max((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v > beta:
                # prune
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MIN's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - player whose turn it is right now. Can be any of MAX's opponents
        currTurn - currTurn + 1 is used to determine who's playing next
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        if currTurn == self.turns - 1:
            # if last Agent of ply, call maxAgent to play
            nextTurn = 0
            nextAgentIndex = self.agents[nextTurn]  # whose turn it is next
            for action in legalActions:
                v = min(v,
                        self.maxValue(gameState.generateSuccessor(agentIndex, action), d + 1, nextAgentIndex, nextTurn,
                                      alpha, beta))
                if v < alpha:
                    # prune
                    return v
                beta = min(beta, v)
            return v

        # else, call another minAgent to play
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]
            v = min((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v < alpha:
                # prune
                return v
            beta = min(beta, v)
        return v

    def alphaBetaSearch(self, gameState, d, agentIndex, currTurn):
        """
        Runs alpha-beta-search algorithm. Starts with MAX player.
        PARAM:
        gameState
        d - current depth, set to 0
        agentIndex - player whose turn it is right now; to start, our agent's index
        currTurn - current Turn, starts with 0
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """

        alpha = float("-inf")
        beta = float("inf")
        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        previousV = float("-inf")  # used for comparisons in determining the bestAction
        bestAction = None
        for action in legalActions:
            if currTurn < len(self.agents) - 1:
                nextAgentIndex = self.agents[currTurn + 1]  # whose turn it is next
                v = max(v,
                        self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta))
                if v > previousV:
                    # compares every action value to return the best Action
                    bestAction = action
                if v >= beta:
                    return bestAction
                alpha = max(alpha, v)
                previousV = v
        return bestAction  # basically, a complicated argmax



class FoodGreedyDummyAgent(CaptureAgent):
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
        self.legal_positions = legal_positions
        # initialize extra for jail positions (places where we have ghosts)
        for pos in self.opponent_jail_pos:
            opponent_probability[pos] += 2
            legal_positions.remove(pos)

        # initialize probability of opponent position uniformly elsewhere
        for pos in legal_positions:
            opponent_probability[pos] += 1
        opponent_probability.normalize()


        self.opponent_position_distribution = opponent_probability

        # limit of food we're defending
        self.foodEaten = 0

        # calculate the mid pos of the board
        xOpp, YOpp = min(self.opponent_jail_pos)
        xInit, YInit = self.initialAgentPos
        self.mid = abs(xOpp - xInit)


        self.beliefsCounter = util.Counter()
        self.opponents = self.getOpponents(gameState)

        self.legal_positions = legal_positions
        num_partices = 400
        self.inference1 = Inference(gameState, self.index, opponents[0], num_partices)
        self.inference1.initializeUniformly()
        self.inference2 = Inference(gameState, self.index, opponents[1], num_partices)
        self.inference2.initializeUniformly()

        self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution()

        # for alpha-beta pruning
        self.depth = 2

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        '''INFERENCE'''
        # agents for Minimax
        self.agents = [self.index]
        opponents = self.getOpponents(gameState)

        for oppIndex in opponents:
            self.agents.append(oppIndex)

        for oppIndex in opponents:
            if gameState.getAgentState(oppIndex).getPosition() <= 3:
                return self.alphaBetaSearch(gameState, 3, self.index, 0)
        
        self.inference1.observe(gameState)
        self.inference1.elapseTime()
        beliefs1 = self.inference1.getBeliefDistribution()
        self.beliefsCounter[self.opponents[0]] = beliefs1

        self.inference2.observe(gameState)
        self.inference2.elapseTime()

        beliefs2 = self.inference2.getBeliefDistribution()
        self.beliefsCounter[self.opponents[1]] = beliefs2
        '''
                PICK AN ACTION
        '''
        agent_state = gameState.getAgentState(self.index)
        pacman_pos = agent_state.getPosition()
        num_carrying = agent_state.numCarrying

        actions = gameState.getLegalActions(self.index)
        # check if we're in home territory
        # get pacman pos
        isOffense = gameState.getAgentState(self.index).isPacman
        # food list
        food_matrix = self.getFood(gameState)
        food_list = food_matrix.asList()
        remaining_food = [(x, y) for x, y in food_list if food_matrix[x][y]]
        food_ratio = 0.25 * len(remaining_food)
        #print len(remaining_food)
        # we are in home territory
        if num_carrying < math.ceil(food_ratio):
            return self.huntFoodAction(gameState, actions)
        else:
            return self.goHome(gameState)

    def huntFoodAction(self, gameState, actions):
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


    def goHome(self, gameState):

        # get actions
        actions = gameState.getLegalActions(self.index)
        action_home_dist = []

        for action in actions:
            '''
                successor state and all it has to offer
            '''
            # get the successor state (state after which agent takes an action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # get the agent state (AgentState instance let's us get position of agent)
            agent_state = successor_state.getAgentState(self.index)
            # access the position using agent position
            new_pos = agent_state.getPosition()
            dist_to_initial_pos = self.distancer.getDistance(new_pos, self.initialAgentPos)
            action_home_dist.append((action, dist_to_initial_pos))

        return min(action_home_dist, key=itemgetter(1))[0]

    def isInHomeTerritory(self, pos):
        x, y = pos
        x1, y1 = self.initialAgentPos

        # middle x distance > x distance of pos - x distance of initial pos
        return self.mid > abs(x - x1)

    def maxValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MAX's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - in this implementation, MAX can only be our agent.
        NOTE: If we wanted to include our teammate as the other MAX player,
        this method would need to be slightly changed
        currTurn - In this implementation, currTurn in this method is always 0.
        This is because our agent's turn is always first.
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:  # max depth exceeded
            return self.evaluationFunction(gameState)

        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]  # whose turn is next
            v = max((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v > beta:
                # prune
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, d, agentIndex, currTurn, alpha, beta):
        """
        Represents MIN's turn. Returns the value of the best action
        PARAM:
        gameState
        d - current depth, or "round of play". starts at depth 1, limit is self.depth
        agentIndex - player whose turn it is right now. Can be any of MAX's opponents
        currTurn - currTurn + 1 is used to determine who's playing next
        alpha, beta - for pruning
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """
        if d > self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        if currTurn == self.turns - 1:
            # if last Agent of ply, call maxAgent to play
            nextTurn = 0
            nextAgentIndex = self.agents[nextTurn]  # whose turn it is next
            for action in legalActions:
                v = min(v,
                        self.maxValue(gameState.generateSuccessor(agentIndex, action), d + 1, nextAgentIndex, nextTurn,
                                      alpha, beta))
                if v < alpha:
                    # prune
                    return v
                beta = min(beta, v)
            return v

        # else, call another minAgent to play
        for action in legalActions:
            nextAgentIndex = self.agents[currTurn + 1]
            v = min((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta)))
            if v < alpha:
                # prune
                return v
            beta = min(beta, v)
        return v

    def alphaBetaSearch(self, gameState, d, agentIndex, currTurn):
        """
        Runs alpha-beta-search algorithm. Starts with MAX player.
        PARAM:
        gameState
        d - current depth, set to 0
        agentIndex - player whose turn it is right now; to start, our agent's index
        currTurn - current Turn, starts with 0
        COMBINING
        Method must be copied for a-b-search to work in another agent
        """

        alpha = float("-inf")
        beta = float("inf")
        v = float("-inf")
        legalActions = gameState.getLegalActions(agentIndex)
        previousV = float("-inf")  # used for comparisons in determining the bestAction
        bestAction = None
        for action in legalActions:
            if currTurn < len(self.agents) - 1:
                nextAgentIndex = self.agents[currTurn + 1]  # whose turn it is next
                v = max(v,
                        self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1,
                                      alpha, beta))
                if v > previousV:
                    # compares every action value to return the best Action
                    bestAction = action
                if v >= beta:
                    return bestAction
                alpha = max(alpha, v)
                previousV = v
        return bestAction  # basically, a complicated argmax

    def evaluationFunction(self, gameState):
        opponents = self.getOpponents(gameState)
        for oppIndex in opponents:
            print oppIndex
        myPos = gameState.getAgentState(self.index).getPosition()
        print myPos
        isPacman = gameState.getAgentState(self.index).isPacman
        oppDistances = []
        score = 0
        for oppIndex in opponents:
            oppState = gameState.getAgentState(oppIndex)
            oppPos = oppState.getPosition()
            print oppPos
            score += 3 * oppState.scaredTimer
            oppDistances.append(self.distancer.getDistance(myPos, oppPos))
        if not isPacman:
            score += 100
        score += 5 * sum(oppDistances)
        return score
        

        

        

        

