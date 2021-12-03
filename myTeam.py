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
        # initial agent pos
        self.initialAgentPos = gameState.getInitialAgentPosition(self.index)
        # get opponents initial jail pos
        opponents = self.getOpponents(gameState)
        self.opponent_jail_pos = [gameState.getInitialAgentPosition(opponents[i]) for i in range(len(opponents))]
        self.opponents = opponents
        # counter to initialize probability of being eaten in a state.
        opponent_probability = util.Counter()
        # legal positions our pacman could occupy
        legal_positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        for jail in self.opponent_jail_pos:
          # add jail if not already there bc some weird ass bug
          if jail not in legal_positions:
            legal_positions.append(jail)
                
        self.legal_positions = legal_positions

        # initialize legal positions as well
        for pos in legal_positions:
            opponent_probability[pos] += 1
            if pos in self.opponent_jail_pos:
                opponent_probability[pos] += 1

        opponent_probability.normalize()

        # check if it's on red Team
        self.opponent_position_distribution = opponent_probability

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        # pacman position now
        pacman_pos = gameState.getAgentState(self.index).getPosition()
        new_belief = util.Counter()
        # get agent distances
        noisy_distances = gameState.getAgentDistances()
        # get noisy distance for each opponent
        for opp in self.opponents:
            # get noisy distance of the opponent
            noisy_dist = noisy_distances[opp]
            # update the belief state
            for pos in self.legal_positions:
                # get the true distance
                true_distance = self.getMazeDistance(pacman_pos, pos)
                prob = gameState.getDistanceProb(true_distance, noisy_dist)
                # new probability * the old probability
                new_belief[pos] = prob 
        new_belief.normalize()

        actions = gameState.getLegalActions(self.index)
        action_ghost_prob_pairs = []
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
            # distance to max probability
            dist = self.getMazeDistance(new_agent_position, new_belief.argMax())
            action_ghost_prob_pairs.append((action, dist))

        # update the belief state
        self.opponent_position_distribution = new_belief
        return max(action_ghost_prob_pairs, key=itemgetter(1))[0]
