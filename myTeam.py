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
from game import Actions, Directions
import game
from operator import itemgetter
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'OffensiveAgent'):
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
    self.inference1 = Inference(gameState, self.index, self.opponents[0], 50)
    self.inference1.initializeUniformly()
    self.inference2 = Inference(gameState, self.index, self.opponents[1],50)
    self.inference2.initializeUniformly()
    
    self.beliefsCounter[self.opponents[0]] = self.inference1.getBeliefDistribution() 
    self.beliefsCounter[self.opponents[1]] = self.inference2.getBeliefDistribution()
    self.maxFood = len(self.getFood(gameState).asList())
    self.numFoodCarrying = 0
    self.furthestFoodDistance = self.distanceToExtremeFood(gameState, self.getFood(gameState).asList(), False)

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
    CaptureAgent.registerInitialState(self, gameState)
    self.weights = util.Counter()
    self.alpha = 0.5
    self.epsilon = 0.0
    self.discount = 1.0


  def getFeatures(self, gameState):
    features = util.Counter()
    features['bias'] = 1.0
    features['numFoodLeftToAttack'] = len(self.getFood(gameState).asList())
    features['numFoodLeftToDefend'] = len(self.getFoodYouAreDefending(gameState).asList())
    features['closestFood']  = self.distanceToExtremeFood(gameState, self.getFood(gameState).asList())
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
    features['GhostFunction'] = self.ghostFunction(gameState)
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
    return self.weights
    

  def getQValue(self, gameState, action):
    successorState = gameState.generateSuccessor(self.index, action)
    features = self.getFeatures(successorState)
    weights = self.getWeights()
    QValue = 0
    for feature in features: # loop through features
      if features[feature] == None:
        continue      
      QValue += features[feature] * weights[feature] # implement equation
    return QValue

  def update(self, gameState, action, nextGameState, reward):
    """
    Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    QValueOldState = self.getQValue(gameState, action) # current state QValue
    QValueNextState = self.getValue(nextGameState) # next state value
    difference = (reward + (self.discount*QValueNextState)) - QValueOldState
    features = self.getFeatures(gameState)
    for feature in features: # loop through features
        self.weights[feature] += (self.alpha * difference * features[feature]) # implement equations


  def chooseAction(self, gameState):
    if util.flipCoin(self.epsilon): # Flip Coin vs probability epsilon
      legalActionsList = gameState.getLegalActions(self.index) 
      return random.choice(legalActionsList) # if true, choose a random action
    action =  self.getPolicy(gameState) # if not, follow policy
    successorState = gameState.generateSuccessor(self.index, action)
    reward = self.rewardFunction(gameState, successorState)
    self.update(gameState,action, successorState, reward)
    return action
  
  def rewardFunction(self, gameState, successorState):
    reward = 0.0
    reward += 10*(self.getScore(successorState) - self.getScore(gameState))
    ghostFunc1 = self.ghostFunction(gameState)
    ghostFunc2 = self.ghostFunction(successorState)
    reward += (ghostFunc2 - ghostFunc1)  

    myPos2 = successorState.getAgentPosition(self.index)
    if ghostFunc1 == float("-inf") and myPos2 == self.start:
        reward += -100
    
    reward += -10*(len(self.getFood(successorState).asList()) - len(self.getFood(gameState).asList())) 

    reward += -10000*(self.distanceToExtremeFood(successorState,self.getFood(successorState).asList()) - self.distanceToExtremeFood(gameState,self.getFood(gameState).asList())) 
    
    print(reward)
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
      highestStateValue = self.computeValueFromQValues(gameState) # Gets the highest value of a state 
      QValues = util.Counter()
      for action in possibleActions: # loop through possible actions
          QValue = self.getQValue(gameState, action) 
          QValues[action] = QValue
          if QValue == highestStateValue:
            return action
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


  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)