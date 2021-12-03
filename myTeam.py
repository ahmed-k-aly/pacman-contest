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


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
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
        # count walls per state

        # initial agent pos
        self.initialAgentPos = gameState.getInitialAgentPosition(self.index)
        # get opponents initial jail pos
        opponents = self.getOpponents(gameState)
        self.opponent_jail_pos = [gameState.getInitialAgentPosition(opponents[i]) for i in range(len(opponents))]
        self.opponents = opponents
        # distributions of belief of position for each opponent
        self.opp_beliefs = [util.Counter() for _ in range(len(opponents))]

        # walls in the game
        game_walls = gameState.getWalls()
        # legal positions our pacman could occupy
        legal_positions = [pos for pos in game_walls.asList(False) if pos[1] > 1]
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

    def chooseAction(self, gameState):
        """

        """

        self.inference1.observe(gameState)
        self.inference1.elapseTime()
        beliefs1 = self.inference1.getBeliefDistribution()

        self.inference2.observe(gameState)
        self.inference2.elapseTime()
        beliefs2 = self.inference2.getBeliefDistribution()
        beliefsList = [beliefs1, beliefs2]
        self.displayDistributionsOverPositions(beliefsList)
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        agent_state = gameState.getAgentState(self.index)
        # if we are pacman, and the agent is nearby, then threat is high (so negative) (we'll multiply by it later)
        is_pacman = agent_state.isPacman
        surroundingState = util.Counter()
        for action in actions:
            newGameState = gameState.generateSuccessor(self.index, action)
            newPacManPos = newGameState.getAgentPosition(self.index)
            distances_to_ghosts = []
            for belief in beliefsList:
                max_prob = belief.argMax()
                # get the distance to max probability
                distance = self.distancer.getDistance(newPacManPos, max_prob)
                # get the probability of the ghost being nearby
                ghost_prob = belief[max_prob]

                # if the threat is high, this will be true
                # situations of high threat:
                #    1. we are pacman and ghost is close to us (as well as probability is high)
                #    2. we are a ghost, but our scared timer is high, and the probability of a ghost being somewhere is extremely high
                high_threat = (is_pacman and distance < 3 and ghost_prob > 0.7) \
                              or (not is_pacman and agent_state.scaredTimer > 0 and distance < 3 and ghost_prob > 0.7)

                threat_idenitifier = -1 if high_threat else 1
                # mutliply by the threat identifier to make the distance low.
                surroundingState[action] = -distance * threat_idenitifier
                distances_to_ghosts.append((action, distance))
        return surroundingState.argMax()
        # minimax (go back home if we are carrying food) when threat is high (?)

        # hunt food/ghosts when threat is low (?)

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

    
        


