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
# from game import Directions
import game 

from game import Directions, Actions

from agents.sample.baselineTeam import PositionSearchProblem 

#################
# Team creation #
#################

Is_train = True

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyAgent1', second = 'MyAgent2'):
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

MAX_CAPACITY = 5
agents_current_target = {}

class QLeaning(CaptureAgent):
  
    def registerInitialState(self, gameState):
        self.epsilon = 0.05
        self.alpha = 0.2 
        self.discount = 0.9
      #   self.weight = {'closet-food':-3.106, 'bias':4.163, 
      #                  '#1-step-away-from-ghost':-22.342,
      #                  'eat-food':25.793}
        self.weight = {'closest-food': -3.106, 'bias': 4.163, '#1-step-away-from-ghost': -22.342, 'eat-food': 25.793}
        self.start = gameState.getAgentPosition(self.index)
        self.featureExtractor = FeaturesExtractor(self)
        self.nextPos = None
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
       legalActions = gameState.getLegalActions(self.index)

       if len(legalActions) ==0:
          return None
       
       foodRemaining = len(self.getFood(gameState).asList())
      #  print(foodRemaining)
       # winState : <=2, back to boundary
       if foodRemaining<=2:
          bestDist = 999999
          for action in legalActions:
             successor = gameState.generateSuccessor(self.index, action)
             pos = successor.getAgentPosition(self.index)
             dist = self.getMazeDistance(self.start, pos)   
             if dist < bestDist:
                bestDist = dist
                bestAction = action
                bestPos = pos
             self.nextPos= bestPos
          return bestAction
       
       if foodRemaining>2:
          bestDist = 999999
          for action in legalActions:
             successor = gameState.generateSuccessor(self.index, action)
             pos = successor.getAgentPosition(self.index)
             closest_food = self.getClosestDist(gameState, self.getFood(gameState).asList())
             dist = self.getMazeDistance(pos, closest_food)   
            #  print("dist:",dist)
             if dist < bestDist:
                bestDist = dist
                bestAction = action
                bestPos = pos
             self.nextPos= bestPos
         #  return bestAction
       
       action = None

       #epsilon greedy 
       #true(0.05) exploit, false(0.05) exploit
       if Is_train:
          if util.flipCoin(self.epsilon):
             action = random.choice(legalActions)
          else:
             action = self.policyExtract(gameState)
          self.updateWeights(gameState, action)

       else:
          action = self.policyExtract(gameState)

       return action
    
    def getWeight(self):
       return self.weight
    
    def getQ(self, gameState, action):
       #return Q(state, action) = w * featureVector
       features = self.featureExtractor.getFeatures(gameState, action)
       print("features:",features)
       return features * self.weight
    
    def updateWeight(self, gameState, action, nextState, reward):
       features = self.featureExtractor.getFeatures(gameState, action)
       value = self.getQ(gameState, action)
       updatedValue = self.maxQValue(nextState)
       diff = (reward + self.discount * updatedValue) - value
       
       for k, w in self.weight.items():
          self.weight[k] = w + self.alpha * diff * features[k]

       print(self.weight)

    def updateWeights(self, gameState, action):
       nextState = gameState.generateSuccessor(self.index, action)
       reward = self.getReward(gameState, nextState)
       print("\nreward:", reward, "\n")
       self.updateWeight(gameState, action, nextState, reward)

    def getReward(self, gameState, nextState):
       
        """
        Compute the immediate reward for transitioning from current state to next state.
        This can be customized based on what you deem as a 'reward' in the game.
        """
        # if eats food, get reward
        # currFoodList = self.getFood(gameState).asList()
        # nextFoodList = self.getFood(nextState).asList()
        # if len(currFoodList) > len(nextFoodList):
        #     return 1.0  # ate food, positive reward
        # # if gets eaten by ghost, get negative reward
        # if nextState.getAgentPosition(self.index) == self.start:
        #     return -1.0
        # # neutral
        # return 0.0
        reward = 0
        agentPosition = gameState.agentPosition = gameState.getAgentPosition(self.index)
        if self.getScore(nextState) > self.getScore(gameState):
           diff = self.getScore(nextState) - self.getScore(gameState)
           reward = diff * 10
        
        Food_defense = self.getFood(gameState).asList()
        food_dist = min([self.getMazeDistance(agentPosition, food)
                         for food in Food_defense])
    
        if food_dist == 1:
            nextFood = self.getFood(nextState).asList()
            if len(Food_defense) - len(nextFood) == 1:
               reward = 10
        opponents = [gameState.getAgentState(i)
                     for i in self.getOpponents(gameState)]
        ghosts = [a for a in opponents if not a.isPacman and 
                  a.getPosition()!= None]
        
        if len(ghosts) > 0:
            min_g_dist = min([self.getMazeDistance(agentPosition, g.getPosition())
                             for g in ghosts])
        
            if min_g_dist == 1:
              next_pos = nextState.getAgentState(self.index).getPosition()
              if next_pos == self.start:
                 reward -=100
         
        food = self.getFood(gameState)
        dist = self.closest_food_dist(agentPosition, food) 
        if self.nextPos != None:
         dist_next_move = self.closest_food_dist(self.nextPos, food)            
         reward += -10*(dist_next_move-dist)


         currentFeatures = self.getFeatures(gameState, Directions.STOP)
         nextFeatures = self.getFeatures(nextState, Directions.STOP)
          # If agent gets closer to food
        if nextFeatures['distanceToFood'] < currentFeatures['distanceToFood']:
            reward += currentFeatures['distanceToFood'] - nextFeatures['distanceToFood']

         # If agent gets too close to a ghost
        if nextFeatures['distanceToGhost'] <= 1:
            reward -= 50

         # If agent is eaten by an enemy while carrying food
        myNextState = nextState.getAgentState(self.index)
        myCurrentState = gameState.getAgentState(self.index)
        if myCurrentState.isPacman and not myNextState.isPacman and myNextState.numCarrying > 0:
            reward -= 20 + 10 * myNextState.numCarrying

         # If we killed an enemy (this assumes there's a way to detect this)
        enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        if any(enemy.isPacman and enemy.getPosition() == None for enemy in enemies):
            reward += 20 + 10 * myCurrentState.numCarrying

        return reward
        
    def final(self, state):
       CaptureAgent.final(self, state)
       print(self.weight)

    def maxQValue(self, gameState):
         """
         Get the maximum Q value for any action in the given state.
         """
         validActions = gameState.getLegalActions(self.index)
         if len(validActions) == 0:
            return 0.0
        #  qValues = [self.getQ(gameState, a) for a in validActions]
        #  return max(qValues)
         best = self.policyExtract(gameState)
         return  self.getQ(gameState, best)
    
    def policyExtract(self, gameState):
        """
        Get the best action to perform in a given state based on Q values.
        """
        validActions = gameState.getLegalActions(self.index)
        if len(validActions) == 0:
            return None
        
        qValues = {a: self.getQ(gameState, a) for a in validActions}

        best_q = max(qValues.values())

        # best_action = [a for a, q_val, in qValues.items() 
        #                if q_val == best_q]

        print("best_q:", best_q)
        best_action = []
        for a, q_val in qValues.items():
            # print("q_val:",q_val)
            if q_val == best_q:
                best_action.append(a)
        print(best_action)
        print(max(qValues, key=qValues.get))
        return max(qValues, key=qValues.get)
        # random break tie
      #   choice = random.choice(best_action)
      #   print("choice:", choice)
      #   return choice
    
    def getClosestDist(self,gameState,pos_list):
      min_length = 9999
      min_pos = None
      my_pos = gameState.getAgentPosition(self.index)
      # my_pos = my_local_state.getPosition()
      for pos in pos_list:
         temp_length = self.getMazeDistance(my_pos,pos)
         if temp_length < min_length:
            min_length = temp_length
            min_pos = pos
      return min_pos
    
    def closest_food_dist(self, pos, food):
        min_length = 9999
        min_pos = None
        my_pos = pos
        food = food.asList()
      #   print("food:", food)
        for f_pos in food:
            # print("my_pos:",my_pos, "f_pos:", f_pos)
            temp_length = self.getMazeDistance(my_pos, f_pos)
            
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos
      #   print(min_length)
        return min_length
    def getFeatures(self, gameState, action):
      # Extract features, like current foods, walls, opponent positions
      foods = self.getFood(gameState).asList()
      walls = gameState.getWalls()
      opponents = [gameState.getAgentState(i) 
                  for i in self.getOpponents(gameState)]
      ghosts = [opp.getPosition() for opp in opponents if not opp.isPacman and opp.getPosition() != None]
      
      features = util.Counter()

      # Initialize the bias
      features["bias"] = 1.0

      agentPosition = gameState.getAgentPosition(self.index)
      x, y = agentPosition
      dx, dy = Actions.directionToVector(action)
      nextPos = (int(x + dx), int(y + dy))

      # Distance to the closest food
      if len(foods) > 0:  # This should always be True, but better safe than sorry
         minDistance = min([self.getMazeDistance(nextPos, food) for food in foods])
         features["closest-food"] = float(minDistance) / (walls.width * walls.height)

      # One step away from ghost
      features["#1-step-away-from-ghost"] = 0
      if len(ghosts) > 0:
         minDistToGhost = min([self.getMazeDistance(nextPos, ghost) for ghost in ghosts])
         if minDistToGhost == 1:
               features["#1-step-away-from-ghost"] = 1

      # Eating food
      features["eat-food"] = 0
      if nextPos in foods:
         features["eat-food"] = 1

      return features
    
class FeaturesExtractor(CaptureAgent):
    def __init__(self, agentInstance):
      self.agentInstance = agentInstance

    def getFeatures(self, gameState, action):
      # Extract features, like current foods, walls, opponent positions
      foods = self.agentInstance.getFood(gameState).asList()
      walls = gameState.getWalls()
      opponents = [gameState.getAgentState(i) 
                  for i in self.agentInstance.getOpponents(gameState)]
      ghosts = [opp.getPosition() for opp in opponents if not opp.isPacman and opp.getPosition() != None]
      
      features = util.Counter()

      # Initialize the bias
      features["bias"] = 1.0

      agentPosition = gameState.getAgentPosition(self.agentInstance.index)
      x, y = agentPosition
      dx, dy = Actions.directionToVector(action)
      nextPos = (int(x + dx), int(y + dy))

      # Distance to the closest food
      if len(foods) > 0:  # This should always be True, but better safe than sorry
         minDistance = min([self.agentInstance.getMazeDistance(nextPos, food) for food in foods])
         features["closest-food"] = float(minDistance) / (walls.width * walls.height)

      # One step away from ghost
      features["#1-step-away-from-ghost"] = 0
      if len(ghosts) > 0:
         minDistToGhost = min([self.agentInstance.getMazeDistance(nextPos, ghost) for ghost in ghosts])
         if minDistToGhost == 1:
               features["#1-step-away-from-ghost"] = 1

      # Eating food
      features["eat-food"] = 0
      if nextPos in foods:
         features["eat-food"] = 1

      return features
#     def getFeatures(self, gameState, action):
#        # extract features, like current foods, walls, opponent pos
#        foods = self.agentInstance.getFood(gameState)
#        walls = gameState.getWalls()
#        opponent = [gameState.getAgentState(i) 
#                    for i in self.agentInstance.getOpponents(gameState)]
#        ghost = [opp.getPosition()
#                  for opp in opponent if not opp.isPacman 
#                  and opp.getPosition()!= None]
#        features = util.Counter()

#        #intilaize the bias
#        features["bias"] = 1.0

#        agentPosition = gameState.getAgentPosition(self.agentInstance.index)
#        x, y = agentPosition
#        dx, dy = Actions.directionToVector(action) 
#        next_x, next_y = int(x + dx), int(y + dy)

#        features["#1-step-away-from-ghost"] = sum((next_x, next_y)
#            in Actions.getLegalNeighbors(g,walls) for g in ghost)
       
#        if not features["#1-step-away-from-ghost"] and foods[next_x][next_y]:
#           features["eat-food"] = 1.0
      
#       #  print("dist:", foods[next_x][next_y])
#        dist = self.closest_food((next_x, next_y), foods, walls)
#       #  print("closest_food_dist:",dist)
#        if dist is not None:
#           features["closest-food"] = float(dist)/ (
#                         walls.width * walls.height)
#       #  print('features["closest-food"]:',features["closest-food"])
#        features.divideAll(10.0)


#        foodList = self.agentInstance.getFood(gameState).asList()
#        minFoodDistance = self.closest_food((next_x, next_y), foods, walls)
#        features["distance-to-food"] = minFoodDistance

#        # Check if the action leads to a ghost

#        features["leads-to-ghost"] = 0
#        for ghostPos in ghost:
#          if (next_x, next_y) == ghostPos:
#             features["leads-to-ghost"] = 1

#        # Check if action leads to food pellet
#        if (next_x, next_y) in foodList:
#           features["leads-to-food-pellet"] = 1
#        else:
#           features["leads-to-food-pellet"] = 0
#           #print(features)
#        return features
    
    def closest_food(self, pos, food, walls):
        min_length = 9999
        min_pos = None
        my_pos = pos
        food = food.asList()
      #   print("food:", food)
        for f_pos in food:
            # print("my_pos:",my_pos, "f_pos:", f_pos)
            temp_length = self.agentInstance.getMazeDistance(my_pos, f_pos)
            
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos
      #   print(min_length)
        return min_length

    def getClosestDist(self,gameState,pos_list):
      min_length = 9999
      min_pos = None
      my_local_state = gameState.getAgentState(self.index)
      my_pos = my_local_state.getPosition()
      for pos in pos_list:
         temp_length = self.getMazeDistance(my_pos,pos)
         if temp_length < min_length:
            min_length = temp_length
            min_pos = pos
      return min_length
    
class MyAgent1(QLeaning):
   pass

class MyAgent2(QLeaning):
   pass
                  