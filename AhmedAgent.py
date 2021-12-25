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


from capture import noisyDistance
from captureAgents import CaptureAgent
import random, time, util
from game import Actions, Directions
import game
from operator import itemgetter
import math
import json


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='HuntGhostDummyAgent', second='OffensiveQAgent'):
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
    """  # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class BasisAgent(CaptureAgent):
    """
    An agent to serve as the basis of the necessary agent structure. This is the highest agent in
    the hierarchy with most of the helper methods the other agents would need. Other agents should extend
    this agent to serve as a foundation. Essentially, this agent is an extension of captureAgent/
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
        self.opponents = self.getOpponents(gameState)
        self.beliefsCounter = util.Counter()  # data structure to store agents' belief distributions in
        self.legalPositions = self.getAllLegalPositions(gameState)  # gets all legal positions in the game
        self.inference1 = Inference(self.index, self.opponents[0], self.legalPositions,
                                    400)  # Initialize particle filtering for enemy agent 1
        self.inference1.initializeUniformly()
        self.inference2 = Inference(self.index, self.opponents[1], self.legalPositions,
                                    400)  # Initialize particle filtering for enemy agent 2
        self.inference2.initializeUniformly()

        self.beliefsCounter[
            self.opponents[0]] = self.inference1.getBeliefDistribution()  # store agent 1's belief distribution
        self.beliefsCounter[
            self.opponents[1]] = self.inference2.getBeliefDistribution()  # store agent 2's belief distribution'
        self.maxFood = len(self.getFood(gameState).asList())  # maximum food in the game
        self.start = gameState.getAgentPosition(self.index)  # initial position
        self.flag = True  # a flag
        self.pastActions = []  # list of past actions taken by agent (used in learning)

        # calculate the mid pos of the board
        # initial agent pos
        self.initialAgentPos = gameState.getInitialAgentPosition(self.index)
        # get opponents initial jail pos
        opponents = self.getOpponents(gameState)
        # starting positions of opponent
        self.opponent_jail_pos = [gameState.getInitialAgentPosition(opponents[i]) for i in range(len(opponents))]
        xOpp, YOpp = min(self.opponent_jail_pos)
        xInit, YInit = self.initialAgentPos
        self.mid = abs(xOpp - xInit)

        # for alpha-beta pruning
        self.depth = 2

    def getAllLegalPositions(self, gameState):
        '''
        Method that returns all legal positions in a given gameState
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

    def agentState(self, gameState, agentIndex):
        '''
        Returns if the passed in Agent is ghost or PACMAN
        '''
        agentState = str(gameState.getAgentState(agentIndex))  # convert state to string
        return agentState.split(':')[0]  # return the first word

    def distanceToExtremeFood(self, gameState, food, isClosest=True):
        '''
        Takes a food list and a boolean isClosest. If isClosest is True, returns
        the distance to the closest food from the list, else returns the
        distance to the furthest food. PASS FOOD AS LIST NOT GRID
        '''
        myPos = gameState.getAgentPosition(self.index)  # my position
        foodDistances = []
        if len(food) <= 0:
            return 0
        for foodPos in food:
            foodDistances.append(self.getMazeDistance(foodPos, myPos))  # append distances
        if isClosest:
            return min(foodDistances)  # return closest food
        else:
            return max(foodDistances)  # return furthest food

    def getEnemyDistance(self, gameState, enemyIndex):
        '''
        Returns Enemy distance if within our manhattan distance range. Else, returns the most probable
        position out of the belief distribution
        '''
        myPos = gameState.getAgentPosition(self.index)
        enemyPos = self.beliefsCounter[enemyIndex].argMax()
        distance = self.getMazeDistance(myPos, enemyPos)
        return distance

    def goHome(self, gameState, actions):
        '''
        Method that returns an action that gets us closer to home base
        '''
        # get actions
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
            dist_to_initial_pos = self.distancer.getDistance(new_pos, self.start)
            action_home_dist.append((action, dist_to_initial_pos))

        return min(action_home_dist, key=itemgetter(1))[0]

    def distanceToBases(self, gameState, steps):
        '''
        Gets closest distance to base using BFS
        '''
        legalActions = gameState.getLegalActions(self.index)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        if self.agentState(gameState, self.index) == "Ghost":
            return steps
        if gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index):
            # if I wake up in Pacman heaven (starting position), i'm dead
            return 9999
        if steps > 40:
            # if we're too far in
            return 1000
        for action in legalActions:
            successorState = gameState.generateSuccessor(self.index, action)
            steps += 1
            steps = self.distanceToBases(successorState, steps)
        return steps

    def infer(self, gameState):
        '''
        Method that infers the possible positions of the opponents
        '''
        self.inference1.observe(gameState)  # observe location of opponent 1
        self.inference1.elapseTime()
        self.beliefsCounter[
            self.opponents[0]] = self.inference1.getBeliefDistribution()  # updates belief distribution for opponent 1

        self.inference2.observe(gameState)  # observe location of opponent 2
        self.inference2.elapseTime()
        self.beliefsCounter[
            self.opponents[1]] = self.inference2.getBeliefDistribution()  # updates belief distribution for opponent 2

    def getSuccessorPositions(self, position):
        '''
        Method that gets successor positions
        '''
        legalPos = []
        x, y = position
        if (x + 1, y) in self.legal_positions:
            legalPos.append(((x + 1), y))
        if (x - 1, y) in self.legal_positions:
            legalPos.append(((x - 1, y)))
        if (x, y + 1) in self.legal_positions:
            legalPos.append(((x, y + 1)))
        if (x, y - 1) in self.legal_positions:
            legalPos.append(((x, y - 1)))
        return legalPos

    def died(self, newState):
        '''
        Returns true if action makes us die
        '''
        if newState.getAgentPosition(self.index) == self.start:
            if len(self.observationHistory) > 3.0:  # if game already under way means we were not at base
                return True
        return False

    def isInHomeTerritory(self, pos):
        x, y = pos
        x1, y1 = self.initialAgentPos

        # middle x distance > x distance of pos - x distance of initial pos
        return self.mid > abs(x - x1)


class Inference():
    '''
    Independent module used for inference. The inference method being implemented here is particle filtering
    '''

    def __init__(self, index, opponentIndex, legalPositions, numParticles=600):
        self.numParticles = numParticles
        self.myIndex = index
        self.opponentIndex = opponentIndex
        self.legalPositions = legalPositions  # gets the legal positions

    def initializeUniformly(self):
        # for each position, assign numParticles/numLegalPositions particles
        self.particleList = []
        i = 0
        while i <= self.numParticles:
            self.particleList += self.legalPositions
            i += len(self.legalPositions)

    def observe(self, gameState):
        '''
        Observing method that uses probability distribution to observe where the
        agent that's being tracked might be. Generates a new particles list that contains
        particles for the positions the agent might be.
        '''
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
        '''
        Elapses time once. Result is a particle list to the positions that the
        enemy can go to in one move.
        '''
        if len(self.particleList) == 1: return
        newParticleList = []
        for particle in self.particleList:
            newPosDist = self.getPositionDistribution(particle)
            newParticle = util.sample(newPosDist)
            newParticleList.append(newParticle)
        self.particleList = newParticleList

    def getPositionDistribution(self, particle):
        '''
        Returns the position distribution of the particle list to where the agent can move to
        '''
        counter = util.Counter()
        possibleFuturePos = self.getSuccessorPositions(particle)
        for pos in possibleFuturePos:
            counter[pos] = 1.0 / len(possibleFuturePos)
        return counter

    def getSuccessorPositions(self, position):
        '''
        Gets possible positions each particle can transition to (called in elapse time)
        '''
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
        counter = util.Counter()
        for particle in self.particleList:
            counter[particle] += 1
        counter.normalize()
        return counter


class ApproximateQAgent(BasisAgent):
    '''
    Module used for an approximateQAgent with features. This class implements
    Approximate Q Learning. Extend it to implement specific agents. Only override
    getFeatures(), rewardFunction(), final(), and getWeights() Functions.
    '''

    def registerInitialState(self, gameState):
        BasisAgent.registerInitialState(self, gameState)
        self.weights = util.Counter(self.getWeights())
        # Turn off learning parameters by changing discount and epsilon to be zero
        self.alpha = 0.7
        self.epsilon = 0.0
        self.discount = 0.0

    def getQValue(self, gameState, action):
        '''
        Takes a gamestate and an action and returns the Q value for the state-action pair.
        '''
        features = self.getFeatures(gameState, action)  # get features
        weights = self.weights
        QValue = 0
        for feature in features:  # loop through features
            if features[feature] == None:
                continue
            try:
                QValue += features[feature] * weights[feature]  # implement equation
            except:
                print("Error with feature: {}\n".format(feature))
                continue
        return QValue

    def update(self, gameState, action, nextGameState, reward):
        """
        updates weights based on transition (used for training)
        """
        QValueOldState = self.getQValue(gameState, action)  # current state QValue
        QValueNextState = self.getValue(nextGameState)  # next state value
        difference = (reward + (self.discount * QValueNextState)) - QValueOldState
        features = self.getFeatures(gameState, action)
        for feature in features:  # loop through features
            self.weights[feature] += (self.alpha * difference * features[feature])  # implement equations

    def chooseAction(self, gameState):
        try:
            self.infer(gameState)
            legalActionsList = gameState.getLegalActions(self.index)

            if Directions.STOP in legalActionsList:
                legalActionsList.remove(Directions.STOP)  # remove Stopping from legal actions list

            if self.flag:  # choose a random action for the first move since there's no PreviousObservation
                action = random.choice(legalActionsList)
                self.pastActions.append(action)
                self.flag = False  # sets flag to false to never go through this block again
                return action

            carryingThresh = 0.25 * len(self.getFood(gameState).asList())  # threshold for numFood carrying
            if 2 < len(self.getFood(
                    gameState).asList()) < 5:  # threshold exception if we're carrying between 2 and 5 foods
                carryingThresh = len(self.getFood(gameState).asList()) - 2
            if gameState.getAgentState(
                    self.index).numCarrying > carryingThresh:  # if we're carrying more than the threshold
                return self.goHome(gameState, legalActionsList)  # return home

            if util.flipCoin(self.epsilon):  # Flip Coin vs probability epsilon
                action = random.choice(legalActionsList)  # if true, choose a random action
            else:
                action = self.getPolicy(gameState)  # if not, follow policy
            self.pastActions.append(action)
        except:
            return random.choice(gameState.getLegalActions(self.index))
        return action  # return action

    def getPolicy(self, gameState):
        '''
        Returns the optimal action according to the policy.
        '''
        return self.computeActionFromQValues(gameState)

    def computeActionFromQValues(self, gameState):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        possibleActions = gameState.getLegalActions(self.index)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        QValues = util.Counter()
        for action in possibleActions:  # loop through possible actions
            QValue = self.getQValue(gameState, action)
            QValues[action] = QValue
        return QValues.argMax()

    def computeValueFromQValues(self, gameState):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        compareList = []  # Put all possible action combinations for one state and get max
        possibleActions = gameState.getLegalActions(self.index)
        if len(possibleActions) == 0:  # If no possible actions, return 0
            return 0.0
        for action in possibleActions:  # Loop through legal actions
            QValue = self.getQValue(gameState, action)
            compareList.append(QValue)  # Add QValues of all legal actions for that state in compareList
        return max(compareList)  # Return the top

    def getValue(self, gameState):
        '''
        Returns value of a state
        '''
        return self.computeValueFromQValues(gameState)

    def getWeights(self):
        '''
        Gets the weights for features. Override in specific agents
        '''
        pass

    def getFeatures(self, gameState, action):
        '''
        Generate features for agent. Override in specific agents.
        '''
        pass

    def rewardFunction(self, gameState, successorState):
        '''
        Function used to generate reward. Override in specific agents
        '''
        pass

    def final(self, gameState):
        '''
        Called after game concludes (used for learning). Override in later classes
        '''
        pass


class OffensiveQAgent(ApproximateQAgent):
    '''
    Offensive Approximate Q Agent with mainly offensive features
    '''

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)

    def getWeights(self):
        '''
        Gets the weights for features. Weights are hardcoded in after training
        '''
        # transform json dict to counter object
        weights = {"trapped": 13.122370684591138, "enemy2scared": 22.968380471615937, "bias": 264.4039735582454,
                   "distanceToEnemy2Ghost": 6.400730669328083, "enemy1scared": 31.57088332866122,
                   "eats-food": 60.59532291717952, "distanceToEnemy1Ghost": 5.4450168219363455,
                   "closest-food": -71.63420575843236, "meScared": 1.3421265214059215,
                   "closestCapsule": -18.368584106364622, "distanceToEnemy1Pacman": 0.35962756114955263,
                   "distanceToScared1Ghost": -32.54014536057823, "#-of-ghosts-1-step-away": -54.29555975955324,
                   "foodCarry": -35.80272093504407, "depositFood": 169.36302262122584, "distanceToEnemy2Pacman": 0.0,
                   "distanceToScared2Ghost": -3.16976239353985817}
        return weights

    def getFeatures(self, gameState, action):
        '''
        Gets features for the offensive agent
        '''

        successorState = gameState.generateSuccessor(self.index, action)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        ghostPositions = []
        for i in range(len(self.opponents)):
            ghostPositions.append(self.beliefsCounter[self.opponents[i]].argMax())
        ghosts = ghostPositions

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features['enemy1scared'] = 1.0 if gameState.getAgentState(
            self.opponents[0]).scaredTimer > 1.0 else 0  # on if enemy 1 is scared feature
        features['enemy2scared'] = 1.0 if gameState.getAgentState(
            self.opponents[1]).scaredTimer > 1.0 else 0  # on if enemy 2 is scared feature
        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        if features['enemy1scared'] == 1.0:  # if enemy1 is scared
            features['distanceToScared1Ghost'] = 1.0 + self.getEnemyDistance(gameState, self.opponents[0]) / float(
                walls.width * walls.height) if self.agentState(gameState, self.opponents[0]) == "Ghost" else 0
            features["#-of-ghosts-1-step-away"] = 0  # no fear of ghosts (assumes other ghost's effect is negligible)
        if features['enemy2scared'] == 1.0:  # if enemy 2 is scared
            features['distanceToScared2Ghost'] = 1.0 + self.getEnemyDistance(gameState, self.opponents[1]) / float(
                walls.width * walls.height) if self.agentState(gameState, self.opponents[1]) == "Ghost" else 0
            features["#-of-ghosts-1-step-away"] = 0  # no fear of ghosts (assumes other ghost's effect is negligible)
        if not features["#-of-ghosts-1-step-away"] and food[next_x][
            next_y]:  # if there is no danger of ghosts then eat food
            features["eats-food"] = 1.0
        if features['enemy1scared'] == 1.0 and (next_x, next_y) in ghosts:  # if enemy 1 scared
            features["eats-ghost1"] = 1.0  # eat enemy 1
        if features['enemy2scared'] == 1.0 and (next_x, next_y) in ghosts:  # if enemy 2 scared
            features["eats-ghost2"] = 1.0  # eat enemy 2
        numFoodCarrying = float(gameState.getAgentState(self.index).numCarrying)
        if numFoodCarrying > 0:  # if carrying more than one food
            features['foodCarry'] = (numFoodCarrying) / (self.distanceToBases(gameState,
                                                                              0))  # feature of keep carrying food or not based on how much food agent is carrying vs distance from home
        features["depositFood"] = 1.0 if successorState.getScore() - gameState.getScore() > 0 else 0

        features["closestCapsule"] = self.distanceToExtremeFood(gameState, self.getCapsules(gameState)) / float(
            walls.width * walls.height)  # distance to closest capsule

        newFood = self.getFood(successorState).asList()
        dist = self.distanceToExtremeFood(successorState, newFood)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        features.divideAll(10.0)

        return features

    def rewardFunction(self, gameState, successorState):
        # Rewards for minimizing distance to food
        reward = 0.0
        if (self.distanceToExtremeFood(successorState,
                                       self.getFood(successorState).asList()) - self.distanceToExtremeFood(gameState,
                                                                                                           self.getFood(
                                                                                                               gameState).asList())) < 0:
            # if we get closer to food
            reward += 5.2 + 3.0 / (
                    1 + self.distanceToExtremeFood(successorState, self.getFood(successorState).asList()))
        else:
            reward += -2.1

        if self.died(successorState):
            # if died
            reward += -30  # punish 30 points

        newCapsules = self.getCapsules(successorState)
        oldCapsules = self.getCapsules(gameState)
        if len(newCapsules) - len(oldCapsules) < 0:
            reward += 0.01
        for i in self.opponents:
            oldEnemyDistance = self.getEnemyDistance(gameState, i)
            newEnemyDistance = self.getEnemyDistance(successorState, i)
            if oldEnemyDistance <= 1.0 and newEnemyDistance > 5 and gameState.getAgentState(i).scaredTimer > 0:
                # Ate Pacman
                print("Ate Pacman")
                reward += 20
            elif self.getEnemyDistance(gameState, i) < 5:
                enemyPos = self.beliefsCounter[i].argMax()
                if self.distancer.getDistance(enemyPos, gameState.getInitialAgentPosition(i)) < 8:
                    reward += 20

        if (self.getScore(successorState) - self.getScore(gameState)) > 0:
            # if deposited food
            reward += 21 + self.getScore(successorState)  # reward 21 points + how much food added

        if (len(self.getFood(successorState).asList()) - len(self.getFood(gameState).asList())) < 0:
            # if ate food
            reward += 10.0
        return reward


class HuntGhostDummyAgent(BasisAgent):
    '''
        AGENT TO SERIALLY BUST GHOSTS
    '''


    def chooseAction(self, gameState):
        '''INFERENCE'''
        self.infer(gameState)

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
            for pos in self.legalPositions:
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
                # num_walls = 1 if self.num_walls[pacman_pos] == 0 else self.num_walls[pacman_pos]
                # num_walls_eval = num_walls * 100
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
        bestAction = Directions.STOP
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
