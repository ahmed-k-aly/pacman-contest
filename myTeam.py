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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HuntGhostDummyAgent', second = 'OffensiveQAgent'):
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

class DummyAgent(CaptureAgent):
  def registerInitialState(self, gameState):
        
    CaptureAgent.registerInitialState(self, gameState)
  
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)  
    
class OffensiveAgent(CaptureAgent):
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
    self.opponents = self.getOpponents(gameState)
    self.beliefsCounter = util.Counter()
    self.inference1 = Inference(gameState, self.index, self.opponents[0], 400)
    self.inference1.initializeUniformly()
    self.inference2 = Inference(gameState, self.index, self.opponents[1], 400)
    self.inference2.initializeUniformly()
    
    self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution() 
    self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution()
    self.maxFood = len(self.getFood(gameState).asList())
    self.numFoodCarrying = 0
    self.furthestFoodDistance = self.distanceToExtremeFood(gameState, self.getFood(gameState).asList(), False)
    self.start = gameState.getAgentPosition(self.index)
    self.flag = True
    self.pastActions=[]
    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    
    """
    
    self.inference1.observe(gameState) # observe location of opponent 1
    self.inference1.elapseTime() 
    self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution() # updates belief distribution for opponent 1

    self.inference2.observe(gameState) # observe location of opponent 2
    self.inference2.elapseTime()
    self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution() # updates belief distribution for opponent 2

    beliefs = [self.beliefsCounter[self.opponents[0]], self.beliefsCounter[self.opponents[1]]]
    #self.displayDistributionsOverPositions(beliefs)
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    surroundingState = util.Counter()
    for action in actions:
      newGameState = gameState.generateSuccessor(self.index, action)
      
      stateValue = self.computeStateValue(newGameState)
      if self.agentState(gameState,self.index) == "Pacman" and (self.getEnemyDistance(gameState,self.opponents[0]) == 1 or self.getEnemyDistance(gameState,self.opponents[1]) == 1) and self.agentState(newGameState,self.index) == "Ghost":
        continue
      print("Action, StateValue = {}, {}".format(action, stateValue))
      surroundingState[action] = stateValue
    action = surroundingState.argMax() # action to return
    
    gameState = gameState.generateSuccessor(self.index, action) # the new game state
    currFood = self.getFood(gameState) # new food
    previousGameState = self.getPreviousObservation() # the previousGameState
    if previousGameState == None: 
      return action
    prevFood = self.getFood(previousGameState) # the old food
    x,y = gameState.getAgentPosition(self.index) # my curr location
    if prevFood[x][y] and not currFood[x][y]:
      self.numFoodCarrying += 1
    myState = self.agentState(gameState,self.index)
    if myState == "Ghost":
      self.numFoodCarrying = 0
    return action


  def computeStateValue(self, gameState):
      '''  
      An Evaluation Function. Override it in other Agents. 
      '''
      stateValue = 0
      numFoodLeft = len(self.getFood(gameState).asList())
      # Linear combination of distance to food, an enemy function, and num foods left with the opponent
      stateValue += (1*self.ghostFunction(gameState, False)) 
      stateValue += 1*self.distanceToFoodFeature(gameState)
      #stateValue += 1*self.numWallsNearbyFeature(gameState)
      #stateValue += 10.0*self.numFoodLeftToAttackFeature(gameState)
      if self.numFoodCarrying > 5:
        stateValue += 10000000 * self.distanceToMyClosestFoodFeature(gameState)
      
      
      return stateValue


  def distanceToMyClosestFoodFeature(self, gameState):
    distance = self.distanceToExtremeFood(gameState, self.getFoodYouAreDefending(gameState).asList())
    return 1.0/distance
  
  
  def numFoodCarryingFeature(self, gameState):
    return self.numFoodCarrying
  
  def numFoodLeftToDefendFeature(self, gameState):
    return len(self.getFood(gameState).asList()) / float(self.maxFood)
  
  
  def numWallsNearbyFeature(self, gameState):
    ''' Returns a value between 0 and 1 relative
        to how many walls are nearby
    '''
    x,y = gameState.getAgentPosition(self.index)
    northWall = gameState.getWalls()[x][y+1]
    southWall = gameState.getWalls()[x][y-1]
    westWall = gameState.getWalls()[x+1][y]
    eastWall = gameState.getWalls()[x-1][y]
    lengthWalls = int(northWall) + int(southWall) + int(eastWall) + int(westWall)
    return 1.0/ (lengthWalls+1)
  
  
  def numFoodLeftToAttackFeature(self,gameState):
    ''' Returns a value between 0-1 of how many foods left to attack'''
    return 1.0 / pow(10,len(self.getFood(gameState).asList()))
  
  def distanceToFoodFeature(self, gameState):
    '''  
    Returns how good the current state is relative to
    the distance of the food
    '''
    currFood = self.getFood(gameState)
    previousGameState = self.getPreviousObservation()
    if previousGameState is not None:
      prevFood = self.getFood(previousGameState)
      x,y = gameState.getAgentPosition(self.index)
      if prevFood[x][y] and not currFood[x][y]:
        self.numFoodCarrying += 1
        return 1.0
    return 100.0/math.log((1+self.distanceToExtremeFood(gameState,currFood.asList())))


  def ghostFunction(self, gameState, isDefensive):
    ''' 
    Evaluates the state relative to the agents of the opponents.
    Returns a value from 0-1 based on how good the current state is
    relative only to ghosts
    '''
    #TODO: Method for power pellets logic
    #TODO: Make logic for when we're scared ghost and opponent is PACMAN
    #TODO: Make logic for when we're PACMAN and opponent is scared ghost
    
    enemyIndices = self.getOpponents(gameState)
    enemyDistances = util.Counter()
    enemiesStates = util.Counter()
    utilityEstimate = {} # how scared am I from each opponent
    myState = self.agentState(gameState, self.index)
    for i in enemyIndices:
      utilityEstimate[i] = 0
      enemyDistances[i] = self.getEnemyDistance(gameState, i)
      enemiesStates[i] = self.agentState(gameState, i)
    for i in enemyIndices:
      if enemiesStates[i] == "Ghost":
        utilityEstimate[i] = 3*math.log(1+enemyDistances[i])
      # TODO: add power pellet logic
    return min(utilityEstimate.values())


  def ghostsFunction(self, gameState, isDefensive):
    ''' 
    Evaluates the state relative to the agents of the opponents.
    Returns a value from based on how good the current state is
    relative only to ghosts
    '''
    #TODO: Method for power pellets logic
    #TODO: Make logic for when we're scared ghost and opponent is PACMAN
    #TODO: Make logic for when we're PACMAN and opponent is scared ghost
    
    enemyIndices = self.getOpponents(gameState)
    enemyDistances = util.Counter()
    enemiesStates = util.Counter()
    utilityEstimate = {} # how scared am I from each opponent
    myState = self.agentState(gameState, self.index)
    for i in enemyIndices:
      utilityEstimate[i] = 0
      enemyDistances[i] = self.getEnemyDistance(gameState, i)
      enemiesStates[i] = self.agentState(gameState, i)
    for i in enemyIndices:
      # TODO: add power pellet logic
      if enemiesStates[i] == "Ghost" and myState == "Pacman":
        # Try and get away from the enemy
          if enemyDistances[i] != None:
            utilityEstimate[i] = enemyDistances[i]
          elif enemyDistances[i] == None:
            utilityEstimate[i] = 1
          elif enemyDistances[i] <= 2 : 
            utilityEstimate[i] = float('-inf') # worst case
      elif enemiesStates[i] == "Pacman" and myState == "Ghost":
        # Try and approach the enemy
          if enemyDistances[i] != None:
            utilityEstimate[i] = -enemyDistances[i] # negative distance to the ith enemy
          elif enemyDistances[i] == None: 
            #TODO: ADD prob distribution logic
            # Hunt Pacman
            utilityEstimate[i] = 0
    return sum(utilityEstimate.values())


  def agentState(self,gameState, agentIndex):
    ''' 
    Returns if the passed in Agent is ghost or PACMAN
    '''
    agentState = str(gameState.getAgentState(agentIndex)) # convert state to string
    return agentState.split(':')[0] # return the first word

  def distanceToExtremeFood(self, gameState, food, isClosest = True):
      '''
      Takes a food list and a boolean isClosest. If isClosest is True, returns 
      the distance to the closest food from the list, else returns the
      distance to the furthest food
      '''
      myPos = gameState.getAgentPosition(self.index) 
      foodDistances = []
      if len(food) <= 0:
        return 0
      for foodPos in food:
        foodDistances.append(self.getMazeDistance(foodPos, myPos))
      if isClosest:
        return min(foodDistances)
      else:
        return max(foodDistances)
  
  def getEnemyDistance(self, gameState, enemyIndex):
    ''' 
    Returns Enemy distance if within our range. Else, return None
    '''
    myPos = gameState.getAgentPosition(self.index)
    enemyPos = self.beliefsCounter[enemyIndex].argMax()
    distance = self.getMazeDistance(myPos,enemyPos)
    return distance
    
class DefensiveAgent(OffensiveAgent):


  def registerInitialState(self, gameState):
    OffensiveAgent.registerInitialState(self, gameState)

  def computeStateValue(self, gameState):
      # Linear combination of distance to food, an enemy function, and num foods left with the opponent
      stateValue = -self.distanceToExtremeFood(gameState,self.getFoodYouAreDefending(gameState).asList(), False) + (100*self.ghostFunction(gameState, True))
      return stateValue

  def computessStateValue(self, gameState):
      '''  
      An Evaluation Function. Override it in other Agents. 
      '''
      numFoodLeft = len(self.getFood(gameState).asList())
      # Linear combination of distance to food, an enemy function, and num foods left with the opponent
      stateValue = -100*self.distanceToExtremeFood(gameState,self.getFood(gameState).asList()) + (10*self.ghostFunction(gameState, True)) - 100*numFoodLeft 
      
      return stateValue

        
class Inference():
  def __init__(self, gameState, index, opponentIndex, numParticles = 600):
    #TODO: Use info from both my agents to infer 
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
          position = (x,y)
          if legalPositions[x][y]:
            legalPositionsAsList.append(position)
    return legalPositionsAsList
  
  
  def initializeUniformly(self):
      numLegalPositions = len(self.legalPositions)
      numParticlesPerPosition = self.numParticles//numLegalPositions
      # for each position, assign numParticles/numLegalPositions particles
      self.particleList = []
      i = 0
      while i <= self.numParticles:
        self.particleList += self.legalPositions
        i += len(self.legalPositions)
      
      # for position in self.legalPositions:
      #     particle = position
      #     self.particleList += [particle] * numParticlesPerPosition

  def observe(self, gameState):
        agentIndex = self.opponentIndex
        enemyPos = gameState.getAgentPosition(agentIndex)
        if enemyPos is not None:
            self.particleList = [enemyPos]
            return
        noisyDistance = gameState.getAgentDistances()[agentIndex]
        myPosition = gameState.getAgentPosition(self.myIndex)

        beliefs = self.getBeliefDistribution() # get particles belief  distribution
        allPossible = util.Counter() 
        for particle in self.particleList: # loop over all legal positions
            trueDistance = util.manhattanDistance(particle, myPosition)
            if gameState.getDistanceProb(trueDistance, noisyDistance) > 0:
              allPossible[particle] = beliefs[particle] * gameState.getDistanceProb(trueDistance, noisyDistance) #  updates belief value for a position
        allPossible.normalize()
        beliefs = allPossible # update the new belief state
        if beliefs.totalCount() == 0: # if all weights are zero, initialize from the prior
            self.initializeUniformly()
            beliefs = self.getBeliefDistribution()
        
        newParticleList = []
        for _ in range(self.numParticles):
            particle = util.sample(beliefs, self.particleList) # sample particle from the distribution
            newParticleList.append(particle)
        self.particleList = newParticleList # update the particle list

  def elapseTime(self):
    if len(self.particleList) ==1: return
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
        counter[pos] = 1.0/len(possibleFuturePos)
    return counter
    
  def getSuccessorPositions(self, position):
    legalPos = [position]
    x,y = position
    if (x+1, y) in self.legalPositions:
      legalPos.append(((x+1), y))
    if (x-1, y) in self.legalPositions:
        legalPos.append(((x-1, y)))
    if (x, y+1) in self.legalPositions:
        legalPos.append(((x, y+1)))
    if (x, y-1) in self.legalPositions:
        legalPos.append(((x, y-1)))
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











class JointParticleFilter(Inference):
  
  def __init__(self, opponents):
    self.opponents = opponents
    
  def initializeParticles(self):
    numLegalPositions = (len(self.legalPositions))^(self.numGhosts)
    numParticlesPerPosition = self.numParticles//numLegalPositions
    # for each position, assign numParticles/numLegalPositions particles
    self.particles = []


    # import the product function from itertools module
    from itertools import product
    combinationOfPositions = list(product(self.legalPositions, repeat = self.numGhosts))
    random.shuffle(combinationOfPositions)
    i = 0
    while i <= self.numParticles:
        self.particles += combinationOfPositions
        i += len(combinationOfPositions)

  
  def observeState(self, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getAgentDistances()

    "*** YOUR CODE HERE ***"
    beliefs = self.getBeliefDistribution() # get particles belief distribution
    # import the product function from itertools module
    allPositions = util.Counter() 
    for particle in self.particles: # loop over particles
        allPositions[particle] = beliefs[particle]
        for i in self.opponents: # we multiply the emissionModels for each ghost seperately
            enemyPos = gameState.getAgentPosition(self.opponents[i])
            if enemyPos is not None:
                temp = beliefs[particle]
                allPositions[particle] = 0
                particle = enemyPos
                allPositions[particle] = temp
                continue
            trueDistance = util.manhattanDistance(particle[i], pacmanPosition)
            allPositions[particle] *=gameState.getDistanceProb(trueDistance, noisyDistances[i]) #  updates belief value for a position
    allPositions.normalize()
    beliefs = allPositions # update the new belief state

    if beliefs.totalCount() == 0: # if all weights are zero, initialize from the prior
        self.initializeParticles() 
        beliefs = self.getBeliefDistribution()

    newParticleList = []
    for _ in range(self.numParticles):
        '''
        items = sorted(beliefs.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
        '''
        particle = util.sample(beliefs, []) # sample particle from the distribution
        newParticleList.append(particle)
    self.particles = newParticleList # update the particle list

  def elapseTime(self, gameState):
    newParticles = []
    for oldParticle in self.particles:
        newParticle = list(oldParticle) # A list of ghost positions
        # now loop through and update each entry in newParticle...
        "*** YOUR CODE HERE ***"
        for i in range(self.numGhosts): # Loop over all ghosts

            newPosDist= getPositionDistributionForGhost(setGhostPositions(gameState, newParticle), i, self.ghostAgents[i])
            newPosi = util.sample(newPosDist) # sample a new position from the position distribution
            newParticle[i] = newPosi # the newPosition is our new position component for the ghost indexed by i in the particle
        "*** END YOUR CODE HERE ***"
        newParticles.append(tuple(newParticle))
    self.particles = newParticles


class ApproximateQAgent(OffensiveAgent):

    def registerInitialState(self, gameState):
        OffensiveAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        self.weights = { "trapped": 13.122370684591138, "enemy2scared": 22.968380471615937, "bias": 264.4039735582454, "distanceToEnemy2Ghost": 6.400730669328083, "enemy1scared": 31.57088332866122, "eats-food": 60.59532291717952, "distanceToEnemy1Ghost": 5.4450168219363455, "closest-food": -71.63420575843236, "meScared": 1.3421265214059215, "closestCapsule": -18.368584106364622, "distanceToEnemy1Pacman": 0.35962756114955263, "distanceToScared1Ghost": -32.54014536057823, "#-of-ghosts-1-step-away": -54.29555975955324, "foodCarry": -35.80272093504407, "depositFood": 169.36302262122584, "distanceToEnemy2Pacman": 0.0, "distanceToScared2Ghost": -3.16976239353985817}
        
          # Turn off learning parameters
        self.alpha = 0.0
        self.epsilon = 0
        self.discount = 0.85


    def getAllFeatures(self, gameState):
        features = util.Counter()
        features['bias'] = 1.0
        features['closestFood']  = self.distanceToExtremeFood(gameState, self.getFood(gameState).asList())
        features['numFoodLeftToAttack'] = len(self.getFood(gameState).asList())
        features['numFoodLeftToDefend'] = len(self.getFoodYouAreDefending(gameState).asList())
        x, y = gameState.getAgentPosition(self.index)
        northWall = gameState.getWalls()[x][y+1]
        southWall = gameState.getWalls()[x][y-1]
        westWall = gameState.getWalls()[x+1][y]
        eastWall = gameState.getWalls()[x-1][y]
        features['northWall'] = int(northWall)
        features['southWall'] = int(southWall)
        features['westWall'] = int(westWall)
        features['eastWall'] = int(eastWall)
        lengthWalls = int(northWall) + int(southWall) + int(eastWall) + int(westWall)
        features['numWallsNearby'] = lengthWalls
        
        features['numCapsulesLeft'] = len(self.getCapsules(gameState))
        features['defendingCapsules'] = len(self.getCapsulesYouAreDefending(gameState))
        features['score'] = self.getScore(gameState)
        enemyIndices = self.getOpponents(gameState) 
        features['distanceToEnemy1'] = self.getEnemyDistance(gameState, enemyIndices[0])
        features['distanceToEnemy1'] = self.getEnemyDistance(gameState, enemyIndices[1])
        myTeam = self.getTeam(gameState)
        myTeam.remove(self.index)
        teammate = myTeam[0]
        teammatePosition = gameState.getAgentPosition(teammate)
        getPos = gameState.getAgentPosition(self.index)
        features['distanceToTeammate'] = self.getMazeDistance(getPos, teammatePosition)
        for feature in features:
            if features[feature] == None: 
                features[feature] = 0
        return features

    def getWeights(self):
        # transform json dict to counter object
        return self.weights
    

    def getQValue(self, gameState, action):
        successorState = gameState.generateSuccessor(self.index, action)
        features = self.getFeatures(gameState, action)
        weights = self.weights
        QValue = 0
        
        
        for feature in features: # loop through features
            if features[feature] == None:
                continue      
            try:
              QValue += features[feature] * weights[feature] # implement equation
            except:
              continue
        #print("QValue: {} Action: {}".format(QValue, action))
        return QValue

    def update(self, gameState, action, nextGameState, reward):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        QValueOldState = self.getQValue(gameState, action) # current state QValue
        QValueNextState = self.getValue(nextGameState) # next state value
        difference = (reward + (self.discount*QValueNextState)) - QValueOldState
        features = self.getFeatures(gameState, action)
        for feature in features: # loop through features
            self.weights[feature] += (self.alpha * difference * features[feature])  # implement equations
            #print("Feature: {}, weight: {}".format(feature, self.weights[feature]))
        #self.weights.normalize()
        


    def goHome(self, gameState, actions):
    
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

    def chooseAction(self, gameState):
      try:
        self.inference1.observe(gameState) # observe location of opponent 1
        self.inference1.elapseTime() 
        self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution() # updates belief distribution for opponent 1

        self.inference2.observe(gameState) # observe location of opponent 2
        self.inference2.elapseTime()
        self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution() # updates belief distribution for opponent 2
        legalActionsList = gameState.getLegalActions(self.index) 
        
        if Directions.STOP in legalActionsList:
          legalActionsList.remove(Directions.STOP)
        previousGameState = self.getPreviousObservation()
        

        
        if self.flag: 
          action = random.choice(legalActionsList) 
          self.pastActions.append(action)
          self.flag = False
          return action
        carryingThresh = 0.25 * len(self.getFood(gameState).asList())
        if 2 < len(self.getFood(gameState).asList()) <5:
            carryingThresh = len(self.getFood(gameState).asList()) -2
        if gameState.getAgentState(self.index).numCarrying > carryingThresh:
          return self.goHome(gameState, legalActionsList)
        
        if util.flipCoin(self.epsilon): # Flip Coin vs probability epsilon
            action = random.choice(legalActionsList) # if true, choose a random action
        else:
            action =  self.getPolicy(gameState) # if not, follow policy
        #successorState = gameState.generateSuccessor(self.index, action)
        #reward = self.rewardFunction(gameState, successorState)
        #print(self.getPolicy(gameState)) 
        #self.update(gameState,action, successorState, reward)
        self.pastActions.append(action)
      except:
        
          return random.choice(gameState.getLegalActions(self.index)) 

      return action
    
    def rewardFunction(self, gameState, successorState):
        reward = 0.0
        reward += 10e3*(self.getScore(successorState) - self.getScore(gameState))
      
        for i in self.opponents:
          if gameState.getAgentPosition(i) != None:
            newEnemyDistance = self.getEnemyDistance(successorState,i) 
            oldEnemyDistance = self.getEnemyDistance(gameState,i)
            deltax = newEnemyDistance - oldEnemyDistance 
            if oldEnemyDistance <= 2 and deltax >5:
                  reward += 10e6
                  
              
        myPos2 = successorState.getAgentPosition(self.index)
        if myPos2 == self.start:
            reward += -10e5
        
        reward += 10*(len(self.getFood(successorState).asList()) - len(self.getFood(gameState).asList())) 
        
        reward += 100*(self.distanceToExtremeFood(successorState,self.getFood(successorState).asList()) - self.distanceToExtremeFood(gameState,self.getFood(gameState).asList())) 
        return reward

    def getPolicy(self, gameState):
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
        highestStateValue = self.computeValueFromQValues(gameState) # Gets the highest value of a state 
        QValues = util.Counter()
    
        for action in possibleActions: # loop through possible actions
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
        "*** YOUR CODE HERE ***"
        compareList = [] # Put all possible action combinations for one state and get max
        possibleActions = gameState.getLegalActions(self.index)
        if len(possibleActions) == 0: # If no possible actions, return 0
            return 0.0
        for action in possibleActions: # Loop through legal actions
            QValue = self.getQValue(gameState, action)
            compareList.append(QValue) # Add QValues of all legal actions for that state in compareList
        return max(compareList) # Return the top

    def getFeatures(self, gameState, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        ghostPositions =[]
        for i in range(len(self.opponents)):
            ghostPositions.append(self.beliefsCounter[self.opponents[i]].argMax())
        ghosts = ghostPositions

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        
        
        features['numFoodLeftToAttack'] = len(self.getFood(gameState).asList()) /float(self.maxFood-2)
        features['numFoodLeftToDefend'] = len(self.getFoodYouAreDefending(gameState).asList())/ float(self.maxFood-2)
        northWall = gameState.getWalls()[x][y+1]
        southWall = gameState.getWalls()[x][y-1]
        westWall = gameState.getWalls()[x+1][y]
        eastWall = gameState.getWalls()[x-1][y]
        features['northWall'] = int(northWall)
        features['southWall'] = int(southWall)
        features['westWall'] = int(westWall)
        features['eastWall'] = int(eastWall)
        lengthWalls = int(northWall) + int(southWall) + int(eastWall) + int(westWall)
        features['numWallsNearby'] = lengthWalls / 4.0
        numPacman = 0
        if self.agentState(gameState, self.opponents[0]) == "Pacman":
          features['isEnemy1Pacman'] = 1
          features['isEnemy1Ghost'] = 0
        else:
          features['isEnemy1Ghost'] = 1
          features['isEnemy1Pacman'] = 0
          
        if self.agentState(gameState,self.opponents[1]) == "Pacman":
          features['isEnemy2Pacman'] = 1
          features['isEnemy2Ghost'] = 0
        else:
          features['isEnemy2Ghost'] = 1
          features['isEnemy2Pacman'] = 0
        features['numEnemyPacman'] = features['isEnemy1Pacman'] + features['isEnemy2Pacman']
        features['numEnemyGhost'] = (2-features['numEnemyPacman'])/2.0
        features['numCapsulesLeft'] = len(self.getCapsules(gameState)) /2.0
        #TODO: Add feature distance to nearest capsule
        features['defendingCapsules'] = len(self.getCapsulesYouAreDefending(gameState)) /2.0
        features['score'] = self.getScore(gameState)/ self.maxFood
        enemyIndices = self.getOpponents(gameState) 
        #TODO: Add feature FOR SCARED GHOSTS
        
        features['distanceToEnemy1Pacman'] = self.getEnemyDistance(gameState, enemyIndices[0])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[0]) == "Pacman" else 0
        features['distanceToEnemy1Ghost'] = self.getEnemyDistance(gameState, enemyIndices[0])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[0]) == "Ghost" else 0
        features['distanceToEnemy2Pacman'] = self.getEnemyDistance(gameState, enemyIndices[1])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[1]) == "Pacman" else 0
        features['distanceToEnemy2Ghost'] = self.getEnemyDistance(gameState, enemyIndices[1])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[1]) == "Ghost" else 0

        myTeam = self.getTeam(gameState)
        myTeam.remove(self.index)
        teammate = myTeam[0]
        teammatePosition = gameState.getAgentPosition(teammate)
        getPos = gameState.getAgentPosition(self.index)
        features['distanceToTeammate'] = self.getMazeDistance(getPos, teammatePosition) / float(walls.width * walls.height)
        features['numFoodCarrying'] = self.numFoodCarrying/float(self.maxFood)
        #features['distanceToBase'] = self.getMazeDistance((x,y), self.start) / float(walls.width * walls.height)
        for feature in features:
            if features[feature] == None: 
                features[feature] = 0
        features.divideAll(10.0)
        return features

    def distanceToBases(self, gameState,steps):
      '''  
      Gets closest distance to base 
      '''
      legalActions = gameState.getLegalActions(self.index)
      if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
      if self.agentState(gameState, self.index) == "Ghost":
        return steps 
      if gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index):
            # if I wake up in Pacman heaven, i'm dead
            return 9999
      if steps > 40:
        # if we're too far in
        return 1000
      for action in legalActions:
        successorState = gameState.generateSuccessor(self.index, action)
        steps+=1
        steps = self.distanceToBases(successorState, steps)
      return steps
        
    def getValue(self, gameState):
        return self.computeValueFromQValues(gameState)

    def final(self, gameState):
      return 0

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None
  
  
class OffensiveQAgent(ApproximateQAgent):
  def registerInitialState(self, gameState):
    ApproximateQAgent.registerInitialState(self, gameState)
  
  def getFeatures(self, gameState, action):
    successorState = gameState.generateSuccessor(self.index, action)
    food = self.getFood(gameState)
    walls = gameState.getWalls()
    ghostPositions =[]
    for i in range(len(self.opponents)):
        ghostPositions.append(self.beliefsCounter[self.opponents[i]].argMax())
    ghosts = ghostPositions

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
    if features['enemy1scared'] == 1.0:
        features['distanceToScared1Ghost'] = 1.0 + self.getEnemyDistance(gameState, self.opponents[0])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[0]) == "Ghost" else 0
        features["#-of-ghosts-1-step-away"] = 0
    if features['enemy2scared'] == 1.0:
        features['distanceToScared2Ghost'] = 1.0 + self.getEnemyDistance(gameState, self.opponents[1])/ float(walls.width * walls.height) if self.agentState(gameState,self.opponents[1]) == "Ghost" else 0
        features["#-of-ghosts-1-step-away"] = 0
    # if there is no danger of ghosts then add other features
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        features["eats-food"] = 1.0
    if features['enemy1scared'] == 1.0 and (next_x, next_y) in ghosts:
        features["eats-ghost1"] = 1.0
    if features['enemy2scared'] == 1.0 and (next_x, next_y) in ghosts:
        features["eats-ghost2"] = 1.0
    numFoodCarrying = float(gameState.getAgentState(self.index).numCarrying)
    if numFoodCarrying > 0:
      features['foodCarry'] = (numFoodCarrying)/(self.distanceToBases(gameState, 0)) # numCarrying
    features["depositFood"] = 1.0 if successorState.getScore() - gameState.getScore() > 0 else 0
    features['enemy1scared'] = 1.0 if gameState.getAgentState(self.opponents[0]).scaredTimer > 1.0 else 0
    features['enemy2scared'] = 1.0 if gameState.getAgentState(self.opponents[1]).scaredTimer > 1.0 else 0
    
    
    features["closestCapsule"] = self.distanceToExtremeFood(gameState, self.getCapsules(gameState)) / float(walls.width * walls.height)
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
        # make the distance a number less than one otherwise the update
        # will diverge wildly
        features["closest-food"] = float(dist) / (walls.width * walls.height)
    
    
    features.divideAll(10.0)

    return features


  def died(self, oldState, newState):
    if newState.getAgentPosition(self.index) == self.start:
      if len(self.observationHistory) >3.0: # if game already under way means we were not at base
          return True
    return False
  
  def rewardFunction(self, gameState, successorState):
    # Rewards for minimizing distance to food
    reward = 0.0
    if (self.distanceToExtremeFood(successorState,self.getFood(successorState).asList()) - self.distanceToExtremeFood(gameState,self.getFood(gameState).asList())) < 0:
      # if we get closer to food
      reward += 5.2 + 3.0/(1+self.distanceToExtremeFood(successorState,self.getFood(successorState).asList()))
    else:
        reward += -2.1 

    if self.died(gameState, successorState):
      # if died
      reward+= -30
    
    newCapsules = self.getCapsules(successorState)
    oldCapsules = self.getCapsules(gameState)
    if len(newCapsules) - len(oldCapsules) <0:
        reward += 0.01
    for i in self.opponents:
      oldEnemyDistance = self.getEnemyDistance(gameState, i)
      newEnemyDistance = self.getEnemyDistance(successorState,i)
      if oldEnemyDistance <=1.0 and newEnemyDistance >5 and gameState.getAgentState(i).scaredTimer >0:
        #Ate Pacman
        reward +=20
        
    if (self.getScore(successorState) - self.getScore(gameState))> 0:
      # if deposited food
      reward += 21 + self.getScore(successorState) 

    if (len(self.getFood(successorState).asList()) - len(self.getFood(gameState).asList())) < 0:
      # if ate food
      reward += 10.0

    return reward
  
  
  
  
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

        self.esc_dist = self.getEscapingChance(gameState)

        #esc_beliefs = [self.esc_dist]
        #self.displayDistributionsOverPositions(esc_beliefs)

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

    def getEscapingChance(self, gameState):
        '''
            EVALUATE THE ESCAPING CHANCES OF A GHOST
            HIGHER VALUES: MORE CHANCES TO ESCAPE
            LOWER VALUES : LESS CHANCES TO ESCAPE
        '''
        legal_positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        escaping_chance_distribution = util.Counter()
        # for each legal position, initialize number of escaping routes
        for pos in legal_positions:
            # get possible number of escape routes
            possible_escape_routes = self.getSuccessorPositions(pos)
            # assumption is survival chance is 100% if there are 4 routes, else 1 / num_escape routes
            # survival_chance = 1 if (4 - len(possible_escape_routes)) == 0 else 1 / (4 - len(possible_escape_routes))
            # (assuming 4 routes: 100%; 3 routes : 75%; 2 routes - 50%; 1 route : 25%
            survival_chance = 0.25 * len(possible_escape_routes)
            # get the possible escape routes
            escaping_chance_distribution[pos] = survival_chance
        # normalize escape route chances
        escaping_chance_distribution.normalize()

        for pos in legal_positions:
            # old_belief
            old_belief = escaping_chance_distribution[pos]
            # get possible number of escape routes
            possible_escape_routes = self.getSuccessorPositions(pos)
            # assumption is survival chance is 100% if there are 4 routes, else 1 / num_escape routes
            # survival_chance = 1 if (4 - len(possible_escape_routes)) == 0 else 1 / (4 - len(possible_escape_routes))
            # (assuming 4 routes: 100%; 3 routes : 75%; 2 routes - 50%; 1 route : 25%
            survival_chance = 0.25 * len(possible_escape_routes)
            # for each escape route,
            # calculate the survival chance thru that route
            sum = 0
            for escape_route in possible_escape_routes:
                # sum is belief of escape route * survival chance
                sum += escaping_chance_distribution[escape_route] * survival_chance
            new_belief = old_belief if (old_belief - sum) < 0 else (old_belief - sum)
            # print new_belief
            # get the possible escape routes
            escaping_chance_distribution[pos] = new_belief
        escaping_chance_distribution.normalize()
        return escaping_chance_distribution
    
    def getSuccessorPositions(self, position):
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