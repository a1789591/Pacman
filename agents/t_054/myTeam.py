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
agents_current_target = {}
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
        self.epsilon = 0.1
        self.alpha = 0.2 
        self.discount = 0.9
      #   self.weight = {'closest-food':-3.106, 'bias':4.163, 
      #                  '#1-step-away-from-ghost':-22.342,
      #                  'eat-food':25.793}
        self.weight = {'closest-food': -3.106, 'bias': 4.163, '#1-step-away-from-ghost': -22.342, 'eat-food': 25.793,
                       'own-food-remaining':4.0, 'opponent-food-remaining':4.0, 'carrying-food':0.0, 'closest-wall':1.0}
        self.start = gameState.getAgentPosition(self.index)
        self.featureExtractor = FeaturesExtractor(self)
        self.nextPos = None

        self.carrying = 0
        self.current_target = None
        self.teammate_target = None
        self.boundary = self.getBoundary(gameState)
        # print(self.boundary)
        self.max_capacity = 5
        self.enemy_pos = self.getOpponents(gameState)
        self.capusule_left = 0
        # for cap in self.getCapsules:
        #   self.capusule_left+=1

        self.current_target = None
        self.teammate_target = None

        # get teammate index
        for x in self.getTeam(gameState):
         if x!=(self.index):
           pos = x
        self.get_teammate_index = pos
        CaptureAgent.registerInitialState(self, gameState)

    def get_teammate_pos(self, gameState):
      pos = gameState.getAgentPosition(self.get_teammate_index)
      return pos
    
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
       
      #  if foodRemaining>2:
      #     bestDist = 999999
      #     for action in legalActions:
      #        successor = gameState.generateSuccessor(self.index, action)
      #        pos = successor.getAgentPosition(self.index)
      #        closest_food = self.getClosestDist(gameState, self.getFood(gameState).asList())
      #        dist = self.getMazeDistance(pos, closest_food)   
      #       #  print("dist:",dist)
      #        if dist < bestDist:
      #           bestDist = dist
      #           bestAction = action
      #           bestPos = pos
      #        self.nextPos= bestPos
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

       dx,dy = game.Actions.directionToVector(action)
       x,y = gameState.getAgentState(self.index).getPosition()
       new_x,new_y = int(x+dx),int(y+dy)
       if (new_x,new_y) == self.current_target:
         self.current_target = None
       if self.getFood(gameState)[new_x][new_y]:
         self.carrying +=1
       elif (new_x,new_y) in self.boundary:
         self.carrying = 0

       return action
    
    def getWeight(self):
       return self.weight
    
    def getBoundary(self,gameState):
      boundary_location = []
      height = gameState.data.layout.height
      width = gameState.data.layout.width
      for i in range(height):
         if self.red:
            j = int(width/2)-1
         else:
            j = int(width/2)
         if not gameState.hasWall(j,i):
            boundary_location.append((j,i))
      return boundary_location
    
    def getQ(self, gameState, action):
       #return Q(state, action) = w * featureVector
       features = self.featureExtractor.getFeatures(gameState, action)
      #  print("features:",features)
       return features * self.weight
    
    def updateWeight(self, gameState, action, nextState, reward):
       features = self.featureExtractor.getFeatures(gameState, action)
       value = self.getQ(gameState, action)
       updatedValue = self.maxQValue(nextState)
       diff = (reward + self.discount * updatedValue) - value
       
       self.weight = {k:w + self.alpha * diff * features[k] for k,w
                      in self.weight.items()}
      #  for k, w in self.weight.items():
      #     self.weight[k] = w + self.alpha * diff * features[k]

    def updateWeights(self, gameState, action):
       nextState = gameState.generateSuccessor(self.index, action)
       reward = self.getReward(gameState, nextState)
      #  print("\nreward:", reward, "\n")
       print("reward:", reward)
       self.updateWeight(gameState, action, nextState, reward)

    def getReward(self, gameState, nextState):
       
        """
        Compute the immediate reward for transitioning from current state to next state.
        This can be customized based on what you deem as a 'reward' in the game.
        """

        reward = 0

        # score obtained/lost
        agentPosition = gameState.agentPosition = gameState.getAgentPosition(self.index)
        if self.getScore(nextState) > self.getScore(gameState):
           diff = self.getScore(nextState) - self.getScore(gameState)
           reward += diff * 30
        
        Food_offense = self.getFood(gameState).asList()
        Food_next = self.getFood(nextState).asList()

        # getting closer to food
        current_food = []
        for pos in Food_offense:
        # print(agents_current_target.values())
           if pos not in agents_current_target.values():
               current_food.append(pos)

        next_food = []
        for pos in Food_next:
        # print(agents_current_target.values())
           if pos not in agents_current_target.values():
               next_food.append(pos)
        
        agentPosition_next = gameState.agentPosition = nextState.getAgentPosition(self.index)
        food_dist = min([self.getMazeDistance(agentPosition, food)
                         for food in current_food])
        
        food_dist_next = min([self.getMazeDistance(agentPosition_next, food)
                         for food in next_food])
        if food_dist !=1:
           reward += -(food_dist_next - food_dist ) * 10

        # food eaten
        if food_dist == 1:
            nextFood = self.getFood(nextState).asList()
            if len(Food_offense) - len(nextFood) == 1:
               reward += 20

        # get eaten (my pacman eaten by enemy ghost)
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
                 reward -=80
        
        # get eaten(my scared ghost get eaten by enemy pacman)
        opponents = [gameState.getAgentState(i)
                     for i in self.getOpponents(gameState)]
        pacman = [a for a in opponents if a.isPacman and 
                  a.getPosition()!= None]
        
        if len(pacman) > 0:
            min_p_dist = min([self.getMazeDistance(agentPosition, p.getPosition())
                             for p in pacman])
            
            if min_p_dist == 1:
              next_pos = nextState.getAgentState(self.index).getPosition()
              if next_pos == self.start:
                 reward -=50
 
        # return to boundary if carrying over 5 food
        boundary = self.getBoundary(gameState)
        boundary_dist = min([self.getMazeDistance(agentPosition, b)
                         for b in boundary])
        
        boundary_next = min([self.getMazeDistance(agentPosition_next, b)
                         for b in boundary])
        
        if self.carrying >=5:
            reward += -(boundary_next - boundary_dist ) * 10
         
        
        # getting closer to closest capsule if being chased / teammate being chased


        # eat a capsule while being chased / teammate being chased
        current_capsule = self.getCapsules(gameState)
        next_capsules = self.getCapsulesYouAreDefending(nextState)
      #   print(self.get_teammate_index, self.index)
        team_state = gameState.getAgentState(self.get_teammate_index)

        if self.chasedByGohst(gameState) == True:
           if len(current_capsule) - len(next_capsules) == 1:
               reward +=30

        # get rid of chasing
        if self.chasedByGohst(gameState) == True and self.chasedByGohst(nextState) == False:
           reward+=10



        # waste a capsule while not being chase
        



        # getting closer while chasing enemies
        



        # own food eaten by enemy pacman
      #   Food_defened = []
      #   Food_defened_next = []
      #   enemies_1_this_state = gameState.getAgentState(self.getOpponents(gameState)[0])
      #   enemies_2_this_state = gameState.getAgentState(self.getOpponents(gameState)[1])

      #   enemies_1_next_state = nextState.getAgentState(self.getOpponents(nextState)[0])
      #   enemies_2_next_state = nextState.getAgentState(self.getOpponents(nextState)[1])

      #   Food_defened = enemies_1_this_state.getFood().asList()
      #   Food_defened_next = enemies_1_next_state.getFood().asList()

      # #   print(len(Food_defened),len(Food_defened_next))
      # #   print(self.getCurrentObservation().getBlueFood().asList())
      #   if len(Food_defened_next)>0 and len(Food_defened) > len(Food_defened_next):
      #      reward -= (len(Food_defened)-len(Food_defened_next))*20


        # If agent is eaten by an enemy while carrying food
        myNextState = nextState.getAgentState(self.index)
        myCurrentState = gameState.getAgentState(self.index)

        this_state_pos = gameState.getAgentState(self.index).getPosition()
        next_state_pos = nextState.getAgentState(self.index).getPosition()


        if myCurrentState.isPacman and next_state_pos == self.start:
            if myCurrentState.numCarrying <3:
               reward -= (20 * myCurrentState.numCarrying)
            else:
               reward -= (50 * myCurrentState.numCarrying)
      #   print("      carrying:      ", myCurrentState.numCarrying, self.carrying)

        # If we killed an enemy
        enemies_1_this_state = gameState.getAgentState(self.getOpponents(gameState)[0])
        enemies_2_this_state = gameState.getAgentState(self.getOpponents(gameState)[1])

        enemies_1_next_state = nextState.getAgentState(self.getOpponents(nextState)[0])
        enemies_2_next_state = nextState.getAgentState(self.getOpponents(nextState)[1])

        if enemies_1_this_state.getPosition() != None:
           if self.getMazeDistance(agentPosition, enemies_1_this_state.getPosition()) == 1 and enemies_1_next_state.getPosition() == None: 
            reward += 50 + 20 * enemies_1_this_state.numCarrying

        if enemies_2_this_state.getPosition() != None:
           if self.getMazeDistance(agentPosition, enemies_2_this_state.getPosition()) == 1 and enemies_2_next_state.getPosition() == None: 
            reward += 50 + 20 * enemies_2_this_state.numCarrying

        return reward
        
    def getFeatures(self, state, action):
      features = util.Counter()
      successor = self.getSuccessor(state, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      
      # Distance to food feature
      foodList = self.getFood(successor).asList()
      if len(foodList) > 0:
         minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
         features['distanceToFood'] = minDistance

      # Distance to ghost feature
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      if len(ghosts) > 0:
         ghostDistances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
         features['distanceToGhost'] = min(ghostDistances)
      
      return features
    
    def getBoundary(self,gameState):
      boundary_location = []
      height = gameState.data.layout.height
      width = gameState.data.layout.width
      for i in range(height):
         if self.red:
            j = int(width/2)-1
         else:
            j = int(width/2)
         if not gameState.hasWall(j,i):
            boundary_location.append((j,i))
      return boundary_location
    
    def final(self, state):
       CaptureAgent.final(self, state)
       print("self.weight",self.weight)

    def maxQValue(self, gameState):
         """
         Get the maximum Q value for any action in the given state.
         """
         validActions = gameState.getLegalActions(self.index)
         if len(validActions) == 0:
            return 0.0
         # qValues = [self.getQ(gameState, a) for a in validActions]
         # return max(qValues)
         bestAction = self.policyExtract(gameState)
         return  self.getQ(gameState, bestAction)
    
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

      #   print("best_q:", best_q)
        best_action = []
      #   for a, q_val in qValues.items():
      #       # print("q_val:",q_val)
      #       if q_val == best_q:
      #           best_action.append(a)
        best_action = [a for a, q_val in qValues.items() if q_val == best_q]
      #   print("best_action",best_action)
      #   print("max(qValues, key=qValues.get)",max(qValues, key=qValues.get))
      #   return max(qValues, key=qValues.get)
        # random break tie
        choice = random.choice(best_action)
      #   print("choice:", choice)
        return choice
    
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
               #  min_pos = pos
      #   print(min_length)
        return min_length
    
    def chasedByGohst(self, gameState):
      
      # get opponent's scared timer
      ghost_state = gameState.getAgentState(self.enemy_pos[0])
      scaredTimer = ghost_state.scaredTimer

      # get myself's scared timer
      ghost_state_m = gameState.getAgentState(self.index)
      scaredTimer_m = ghost_state_m.scaredTimer

      dis_from_op = self.get_op_maze_distance(gameState)
      if dis_from_op[0]!=None:
      #   dis_from_op = self.getMazeDistance(agent_my_pos, agent_op_pos)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op)
        if dis_from_op[0] <= 3 and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[0] <= 3 and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        
      if dis_from_op[1]!=None:
      #   dis_from_op1 = self.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
        if dis_from_op[1]<= 3 and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[1]<= 3 and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True     
      return False
    
    def get_op_maze_distance(self, gameState):
      agent_my_pos = gameState.getAgentPosition(self.index)
      agent_op_pos = gameState.getAgentPosition(self.enemy_pos[0])
      agent_op_pos1 = gameState.getAgentPosition(self.enemy_pos[1])
      dis_from_op = None
      dis_from_op1 = None

      if agent_op_pos!=None:
        dis_from_op = self.getMazeDistance(agent_my_pos, agent_op_pos)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op)
        
      if agent_op_pos1!=None:
        dis_from_op1 = self.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
      
      return (dis_from_op, dis_from_op1)
    
class FeaturesExtractor(CaptureAgent):
    def __init__(self, agentInstance):
      self.agentInstance = agentInstance

   #  def getFeatures(self, gameState, action):
   #    # Extract features, like current foods, walls, opponent positions
   #    foods = self.agentInstance.getFood(gameState).asList()
   #    walls = gameState.getWalls()
   #    opponents = [gameState.getAgentState(i) 
   #                for i in self.agentInstance.getOpponents(gameState)]
   #    ghosts = [opp.getPosition() for opp in opponents if not opp.isPacman and opp.getPosition() != None]
      
   #    features = util.Counter()

   #    # Initialize the bias
   #    features["bias"] = 1.0

   #    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
   #    x, y = agentPosition
   #    dx, dy = Actions.directionToVector(action)
   #    nextPos = (int(x + dx), int(y + dy))

   #    # Distance to the closest food
   #    if len(foods) > 0:  # This should always be True, but better safe than sorry
   #       minDistance = min([self.agentInstance.getMazeDistance(nextPos, food) for food in foods])
   #       features["closest-food"] = float(minDistance) / (walls.width * walls.height)

   #    # One step away from ghost
   #    features["#1-step-away-from-ghost"] = 0
   #    if len(ghosts) > 0:
   #       minDistToGhost = min([self.agentInstance.getMazeDistance(nextPos, ghost) for ghost in ghosts])
   #       if minDistToGhost == 1:
   #             features["#1-step-away-from-ghost"] = 1

   #    # Eating food
   #    features["eat-food"] = 0
   #    if nextPos in foods:
   #       features["eat-food"] = 1

   #    return features

    def getFeatures(self, gameState, action):
       
       # extract features, like current foods, walls, opponent pos
       foods = self.agentInstance.getFood(gameState)

       current_food = []
       for pos in foods.asList():
        # print(agents_current_target.values())
        if pos not in agents_current_target.values():
            current_food.append(pos)

       walls = gameState.getWalls()
       opponent = [gameState.getAgentState(i)
                   for i in self.agentInstance.getOpponents(gameState)]
       ghost = [opp.getPosition()
                 for opp in opponent if not opp.isPacman 
                 and opp.getPosition()!= None]
       features = util.Counter()

       #intilaize the bias
       features["bias"] = 1.0

       agentPosition = gameState.getAgentPosition(self.agentInstance.index)
       x, y = agentPosition
       dx, dy = Actions.directionToVector(action) 
       next_x, next_y = int(x + dx), int(y + dy)

       features["#1-step-away-from-ghost"] = sum((next_x, next_y)
           in Actions.getLegalNeighbors(g,walls) for g in ghost)
       
       if not features["#1-step-away-from-ghost"] and foods[next_x][next_y]:
          features["eat-food"] = 1.0
      
      #  print("dist:", foods[next_x][next_y])

      # extract food feature
       dist = self.closest_food((next_x, next_y), current_food, walls)
      #  print("closest_food_dist:",dist)
       if dist is not None:
          features["closest-food"] = float(dist)/ \
          (walls.width * walls.height)
      #  print('features["closest-food"]:',features["closest-food"])


       # extract wall feature
      #  print(walls.asList())
       boundray = self.getBoundary(gameState)
       dist_wall = self.closest_food((next_x, next_y), boundray, current_food)
       if dist_wall is not None:
          features["closest-wall"] = float(dist)/ \
          (walls.width * walls.height)
       
       # Own food remaining
       myFood = self.agentInstance.getFood(gameState).asList()
       features['own-food-remaining'] = sum(len(food) for food in myFood)

       # Opponent's food remaining
       opponentIndices = self.agentInstance.getOpponents(gameState)
       opponentFoods = self.agentInstance.getFoodYouAreDefending(gameState).asList()
       opponentTotalFood = sum([len(food) for food in opponentFoods])
       features['opponent-food-remaining'] = opponentTotalFood

       # Number of food the pacman is carrying
       features['carrying-food'] = self.agentInstance.carrying

       # Distance between pacman and ghosts
       dists = []
       myPos = gameState.getAgentState(self.agentInstance.index).getPosition()
       enemies = [gameState.getAgentState(i).getPosition() for i in opponentIndices]
       
       dists = [self.agentInstance.getMazeDistance(myPos, a) for a in enemies if self.chasedByGohst(gameState) == True and a is not None]
          
       if len(dists)>0:
          features['pac-ghost-distance'] = min(dists)/ (walls.width * walls.height)
       else:
          features['pac-ghost-distance'] = 0

        # Time remaining (comparing scores)

       myScore = gameState.getScore()

       if myScore > 0:
          features['time-remaining-more-points'] = gameState.data.timeleft
       else:
          features['time-remaining-less-points'] = gameState.data.timeleft


       # distance from capsule 
       current_capsules = self.agentInstance.getCapsules(gameState)
       dist_capsule = self.closest_food((next_x, next_y), current_capsules, walls)
      #  print("closest_capsule_dist:",dist_capsule)
       if dist_capsule is not None:
          features["closest-capsule"] = float(dist)/ \
          (walls.width * walls.height)
       else:
          features["closest-capsule"] = 0.0


       # distance from ememy capsule
       current_enemy_capsules =  self.agentInstance.getCapsulesYouAreDefending(gameState)
       dist_enemy_capsule = self.closest_food((next_x, next_y), current_enemy_capsules, walls)
       if dist_enemy_capsule is not None:
          features["closest-enmey-capsule"] = float(dist)/ \
          (walls.width * walls.height)
       else:
          features["closest-enmey-capsule"] = 0.0

       # distance between own ghost and enemy pacman if being chased and scared





       features.divideAll(10.0)
      #  print("features",features)
       return features
    
    def getFoodYouAreDefending(self, gameState):
       
       return super().getFoodYouAreDefending(gameState)

    def closest_food(self, pos, food, walls):
        min_length = 9999
        min_pos = None
        my_pos = pos
      #   food = food.asList()

      #   print("food:", food)
        for f_pos in food:
            # print("my_pos:",my_pos, "f_pos:", f_pos)
            temp_length = self.agentInstance.getMazeDistance(my_pos, f_pos)
            
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos
      #   print(min_length)
        self.agentInstance.current_target =  min_pos
        agents_current_target[self.agentInstance.index] = min_pos
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
    
    def chasingPacman(self, gameState):
      
      # get myself's scared timer
      ghost_state = gameState.getAgentState(self.agentInstance.index)
      scaredTimer = ghost_state.scaredTimer

      # get opponent's scared timer
      ghost_state_op = gameState.getAgentState(self.agentInstance.enemy_pos[0])
      scaredTimer_op = ghost_state_op.scaredTimer
      dis_from_op = self.get_op_maze_distance(gameState)
      
      if dis_from_op[0]!=None:
        if dis_from_op[0] <= 6  and scaredTimer == 0 and gameState.getAgentState(self.agentInstance.enemy_pos[0]).isPacman and\
                    not gameState.getAgentState(self.agentInstance.index).isPacman:
         #  print("chasing op")
          return True
        elif dis_from_op[0] <= 6  and scaredTimer_op >0 and gameState.getAgentState(self.agentInstance.index).isPacman and\
                     not gameState.getAgentState(self.agentInstance.enemy_pos[0]).isPacman:
         #  print("chasing op")
          return True
        
      if dis_from_op[1]!=None:
        if dis_from_op[1]<= 6 and scaredTimer == 0 and gameState.getAgentState(self.agentInstance.enemy_pos[1]).isPacman and \
                    not gameState.getAgentState(self.agentInstance.index).isPacman:
         #  print("chasing op")
          return True
        elif dis_from_op[1]<= 6 and scaredTimer_op > 0 and gameState.getAgentState(self.agentInstance.index).isPacman and \
                    not gameState.getAgentState(self.agentInstance.enemy_pos[0]).isPacman:
         #  print("chasing op")
            # not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          return True
      
      return False
    
    def isPacman(self, gameState):
     agent_my_pos = gameState.getAgentPosition(self.agentInstance.index)
   #   print(agent_my_pos)
     width = gameState.data.layout.width
     if agent_my_pos[0]<=(int(width/2)-1) and not self.agentInstance.red:
       return True
     elif agent_my_pos[0]>=int(width/2) and self.agentInstance.red:
       return True
     return False
    
    def get_op_maze_distance(self, gameState):
      agent_my_pos = gameState.getAgentPosition(self.agentInstance.index)
      agent_op_pos = gameState.getAgentPosition(self.agentInstance.enemy_pos[0])
      agent_op_pos1 = gameState.getAgentPosition(self.agentInstance.enemy_pos[1])
      dis_from_op = None
      dis_from_op1 = None

      if agent_op_pos!=None:
        dis_from_op = self.agentInstance.getMazeDistance(agent_my_pos, agent_op_pos)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op)
        
      if agent_op_pos1!=None:
        dis_from_op1 = self.agentInstance.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
      
      return (dis_from_op, dis_from_op1)
    
    def chasedByGohst(self, gameState):
      
      # get opponent's scared timer
      ghost_state = gameState.getAgentState(self.agentInstance.enemy_pos[0])
      scaredTimer = ghost_state.scaredTimer

      # get myself's scared timer
      ghost_state_m = gameState.getAgentState(self.agentInstance.index)
      scaredTimer_m = ghost_state_m.scaredTimer

      dis_from_op = self.get_op_maze_distance(gameState)
      if dis_from_op[0]!=None:
      #   dis_from_op = self.getMazeDistance(agent_my_pos, agent_op_pos)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op)
        if dis_from_op[0] <= 6 and scaredTimer == 0 and gameState.getAgentState(self.agentInstance.index).isPacman \
            and not gameState.getAgentState(self.agentInstance.enemy_pos[0]).isPacman:
         #  print("chased by op")
          return True
        elif dis_from_op[0] <= 6 and scaredTimer_m > 0 and not gameState.getAgentState(self.agentInstance.index).isPacman \
            and gameState.getAgentState(self.agentInstance.enemy_pos[0]).isPacman:
         #  print("chased by op")
          return True
        
      if dis_from_op[1]!=None:
      #   dis_from_op1 = self.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
        if dis_from_op[1]<= 6 and scaredTimer == 0 and gameState.getAgentState(self.agentInstance.index).isPacman \
            and not gameState.getAgentState(self.agentInstance.enemy_pos[1]).isPacman:
         #  print("chased by op")
          return True
        elif dis_from_op[1]<= 6 and scaredTimer_m > 0 and not gameState.getAgentState(self.agentInstance.index).isPacman \
            and gameState.getAgentState(self.agentInstance.enemy_pos[1]).isPacman:
         #  print("chased by op")
          return True     
      return False
    
    def getBoundary(self,gameState):
      boundary_location = []
      height = gameState.data.layout.height
      width = gameState.data.layout.width
      for i in range(height):
         if self.agentInstance.red:
            j = int(width/2)-1
         else:
            j = int(width/2)
         if not gameState.hasWall(j,i):
            boundary_location.append((j,i))
      return boundary_location
    
class MyAgent1(QLeaning):
    def registerInitialState(self, gameState):
        self.epsilon = 0.05
        self.alpha = 0.2 
        self.discount = 0.9
      #   self.weight = {'closest-food':-3.106, 'bias':4.163, 
      #                  '#1-step-away-from-ghost':-22.342,
      #                  'eat-food':25.793}
      #   self.weight = {'closest-food': -3.106, 'bias': 4.163, '#1-step-away-from-ghost': -22.342, 'eat-food': 25.793,
      #                  'own-food-remaining':4.0, 'opponent-food-remaining':4.0, 'carrying-food':0.0}
        self.weight = {'closest-food': 8.181939865259111e+189, 'bias': 1.0315168819098802e+191, '#1-step-away-from-ghost': -45.41902183237016, 'eat-food': 7.582582234478367, 'own-food-remaining': 1.2584505959300532e+193, 'opponent-food-remaining': -1.436472565519563e+186, 'carrying-food': 18.818357278589076, 'closest-wall': 8.181939865259111e+189, 'pac-ghost-distance': 3.0, 'closest-capsule': 8.181939865259111e+189, 'closest-enmey-capsule': -8.181939865259111e+189}
        self.start = gameState.getAgentPosition(self.index)
        self.featureExtractor = FeaturesExtractor(self)
        self.nextPos = None
        self.current_target = None
        self.teammate_target = None
        self.carrying = 0
        self.current_target = None
        self.teammate_target = None
        self.boundary = self.getBoundary(gameState)
        # print(self.boundary)
        self.max_capacity = 5
        self.enemy_pos = self.getOpponents(gameState)
        self.capusule_left = 0
        # for cap in self.getCapsules:
        #   self.capusule_left+=1

        # get teammate index
        for x in self.getTeam(gameState):
         if x!=(self.index):
           pos = x
        self.get_teammate_index = pos
        CaptureAgent.registerInitialState(self, gameState)

    pass

class MyAgent2(QLeaning):
    def registerInitialState(self, gameState):
        self.epsilon = 0.05
        self.alpha = 0.2 
        self.discount = 0.9
      #   self.weight = {'closest-food':-3.106, 'bias':4.163, 
      #                  '#1-step-away-from-ghost':-22.342,
      #                  'eat-food':25.793}
      #   self.weight = {'closest-food': -3.106, 'bias': 4.163, '#1-step-away-from-ghost': -22.342, 'eat-food': 25.793,
      #                  'own-food-remaining':4.0, 'opponent-food-remaining':4.0, 'carrying-food':0.0}

        self.weight = {'closest-food': 1.6232850580857832e+188, 'bias': 8.362480065497366e+190, '#1-step-away-from-ghost': -39.763059681475475, 'eat-food': -5.299048795647561e+170, 'own-food-remaining': 1.0202225679906786e+193, 'opponent-food-remaining': -1.1458055780473983e+186, 'carrying-food': 2.50874401964921e+191, 'closest-wall': 1.6232850580857832e+188, 'pac-ghost-distance': 3.0, 'closest-capsule': 1.6232850580857832e+188, 'closest-enmey-capsule': -1.6232850580857832e+188}
        self.start = gameState.getAgentPosition(self.index)
        self.featureExtractor = FeaturesExtractor(self)
        self.nextPos = None

        self.carrying = 0
        self.current_target = None
        self.teammate_target = None
        self.boundary = self.getBoundary(gameState)
        # print(self.boundary)
        self.max_capacity = 5
        self.enemy_pos = self.getOpponents(gameState)
        self.capusule_left = 0
        self.current_target = None
        self.teammate_target = None
         # get teammate index
        for x in self.getTeam(gameState):
         if x!=(self.index):
           pos = x
        self.get_teammate_index = pos
        CaptureAgent.registerInitialState(self, gameState)

    pass
                  