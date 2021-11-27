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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
    surroundingState = util.Counter()
    for action in actions:
      newGameState = gameState.generateSuccessor(self.index, action)
      numFoodLeft = len(self.getFood(newGameState).asList())
      stateValue = -self.distanceToExtremeFood(newGameState,self.getFood(newGameState).asList()) + (100*self.ghostFunction(newGameState)) - 100*numFoodLeft
      print("Action, StateValue = {}, {}".format(action, stateValue))
      surroundingState[action] = stateValue
      
    return surroundingState.argMax()

  def ghostFunction(self, gameState):
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
      # TODO: add power pellet logic
      if enemiesStates[i] == "Ghost" and myState == "Pacman":
        # Try and get away from the enemy
          if enemyDistances[i] != None:
            utilityEstimate[i] = enemyDistances[i]
          elif enemyDistances[i] == None:
            utilityEstimate[i] = 1
          elif enemyDistances[i] == 2 or enemyDistances[i] == 1: 
            print("enemyDistance: {}".format(enemyDistances[i]))
            utilityEstimate[i] = float('-inf')
      elif enemiesStates[i] == "Pacman" and myState == "Ghost":
        # Try and approach the enemy
          if enemyDistances[i] != None:
            utilityEstimate[i] = -enemyDistances[i]
          elif enemyDistances[i] == None: 
            #TODO: ADD prob distribution logic
            # Hunt Pacman
            utilityEstimate[i] = 0
    return min(utilityEstimate.values())


  def agentState(self,gameState, agentIndex):
    ''' 
    Returns if the passed in Agent is ghost or PACMAN
    '''
    agentState = str(gameState.getAgentState(agentIndex))
    return agentState.split(':')[0]

  def distanceToExtremeFood(self, gameState, food, isClosest = True):
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
    enemyPos =  gameState.getAgentPosition(enemyIndex)
    if myPos and enemyPos != None:
      distance = self.getMazeDistance(myPos,enemyPos)
      return distance
    return None
  
  def closestFood(self, pos, food, walls):
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