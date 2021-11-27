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

class AlphaBetaAgent(CaptureAgent):
  """
    A crappy agent that runs alpha-beta-search
    Agent is yet to be thoroughly tested because its current evaluation function
    is not very good and it chooses a random action if the opponent is not visible.
    As such, it behaves poorly
    To see it in proper action, combine with a more robust agent and a good evaluation function
  """
  def registerInitialState(self, gameState):
    """
    Same as ReflexCaptureAgent, but initializes max depth
    """
    self.start = gameState.getAgentPosition(self.index)
    self.depth = 3 # a depth of 4 considerably slows down
    CaptureAgent.registerInitialState(self, gameState)

  def evaluationFunction(self, gameState):
    # need a much better evaluation function for this Agent to perform decently
    return self.getScore(gameState)

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
    """
    if d > self.depth: # max depth exceeded
      return self.evaluationFunction(gameState)

    v = float("-inf")
    legalActions = gameState.getLegalActions()
    for action in legalActions:
      nextAgentIndex = self.agents[currTurn + 1] # whose turn is next
      v = max((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1, alpha, beta)))
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
    """
    if d > self.depth:
      return self.evaluationFunction(gameState)
    
    v = float("inf")
    legalActions = gameState.getLegalActions(agentIndex)
    if currTurn == self.turns - 1:
        # if last Agent of ply, call maxAgent to play
        nextTurn = 0
        nextAgentIndex = self.agents[nextTurn] # whose turn it is next
        for action in legalActions:
          v = min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), d + 1, nextAgentIndex, nextTurn, alpha, beta))
          if v < alpha:
            # prune
            return v
          beta = min(beta, v)
        return v
    
    # else, call another minAgent to play
    for action in legalActions:
      nextAgentIndex = self.agents[currTurn + 1]
      v = min((v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1, alpha, beta)))
      if v < alpha:
        # prune
        return v
      beta = min(beta,v)
    return v
  

  def alphaBetaSearch(self, gameState, d, agentIndex, currTurn):
    """
    Runs alpha-beta-search algorithm. Starts with MAX player.
    PARAM:
    gameState
    d - current depth, set to 0
    agentIndex - player whose turn it is right now; to start, our agent's index
    currTurn - current Turn, starts with 0
    """
    
    alpha = float("-inf")
    beta = float("inf")
    v = float("-inf")
    legalActions = gameState.getLegalActions(agentIndex)
    previousV = float("-inf") # used for comparisons in determining the bestAction

    for action in legalActions:
      nextAgentIndex = self.agents[currTurn + 1] # whose turn it is next
      v = max(v, self.minValue(gameState.generateSuccessor(agentIndex, action), d, nextAgentIndex, currTurn + 1, alpha, beta))
      if v > previousV:
        # compares every action value to return the best Action
        bestAction = action
      if v >= beta:
        return bestAction
      alpha = max(alpha, v)
      previousV = v
    return bestAction # basically, a complicated argmax

  def chooseAction(self, gameState):
      """
        For our minimax agent, if any opponents are visible, runs alpha-beta-search, 
        with an evaluation function based on score.
        Otherwise, chooses a random action

        All functionality of this method can be added to a more robust version of this method
      """

      # a list of all agents who are considered in our alpha-beta-search algorithm.
      # at the moment, only considers our agent and all visible opponents; our teammate is not considered.
      self.agents = [self.index]
      for opp in self.getOpponents(gameState):
        if gameState.getAgentPosition(opp): # if the opponent is visible
          self.agents.append(opp)

      if len(self.agents) == 1: # if no opponents are visible
        return random.choice(gameState.getLegalActions(self.index))

      # else, initialize variables required for a-b-search
      self.turns = len(self.agents) # number of turns for each depth of a-b-search
      currTurn = 0
      agentIndex = self.agents[currTurn]
      return self.alphaBetaSearch(gameState, 1, agentIndex, currTurn)



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

