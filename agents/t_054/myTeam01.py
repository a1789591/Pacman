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

from agents.sample.baselineTeam import PositionSearchProblem 

#################
# Team creation #
#################

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

class MyAgent(CaptureAgent):
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

    # get teammate position
    for x in self.getTeam(gameState):
      if x!=(self.index):
        pos = x
    self.get_teammate_pos = pos
    # print(self.get_teammate_pos)


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    if not self.current_target == None:
      # if agent already have a goal
      pass

    elif self.carrying == self.max_capacity or len(self.getFood(gameState).asList())<=2:
      # if agent got all the food it needed
      # it will reach to the closest boundary with A* search (manhattanDistance as heuristic)
      self.current_target = self.getClosestPos(gameState,self.boundary)

      # # store the target information to the map "agents_current_target"
      agents_current_target[self.index] = self.current_target

    else:
      # if agent have more capacity to carry
      # it will find the next closest food
      foodGrid = self.getFood(gameState)
      # print(foodGrid.asList())
      
      # get the current food other than targetted by teammates
      current_food = []
      for pos in foodGrid.asList():
        # print(agents_current_target.values())
        if pos not in agents_current_target.values():
            current_food.append(pos)
      self.current_target = self.getClosestPos(gameState,current_food)

      # store the target information to the map "agents_current_target"
      agents_current_target[self.index] = self.current_target

    problem = PositionSearchProblem(gameState,self.current_target,self.index)
    path  = self.aStarSearch(problem, gameState)

    if path == []:
      actions = gameState.getLegalActions(self.index)
      return random.choice(actions)
    
    else:
      action = path[0]
      dx,dy = game.Actions.directionToVector(action)
      x,y = gameState.getAgentState(self.index).getPosition()
      new_x,new_y = int(x+dx),int(y+dy)
      if (new_x,new_y) == self.current_target:
        self.current_target = None
      if self.getFood(gameState)[new_x][new_y]:
        self.carrying +=1
      elif (new_x,new_y) in self.boundary:
        self.carrying = 0
      return path[0]
  
  def getClosestPos(self,gameState,pos_list):
    min_length = 9999
    min_pos = None
    my_local_state = gameState.getAgentState(self.index)
    my_pos = my_local_state.getPosition()
    for pos in pos_list:
      temp_length = self.getMazeDistance(my_pos,pos)
      if temp_length < min_length:
        min_length = temp_length
        min_pos = pos
    return min_pos

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

  def aStarSearch(self, problem, gameState):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    from util import PriorityQueue
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    # print(f"start states {startState}")
    startNode = (startState, '',0, [])
    heuristic = problem._manhattanDistance
    heuristic_chased_dist = 0
    dist = heuristic(startState)

    if self.chasedByGohst_sensitive(gameState, 5):
      chasing_dis = self.get_op_maze_distance(gameState)
      if chasing_dis[0]!= None and chasing_dis[1]!=None:
        if chasing_dis[0]<=chasing_dis[1]:
          heuristic_chased_dist = chasing_dis[0]
        else:
          heuristic_chased_dist = chasing_dis[1]
      else:
        if chasing_dis[0]!= None:
          heuristic_chased_dist = chasing_dis[0]
        if chasing_dis[1]!=None:
          heuristic_chased_dist = chasing_dis[1]
      # print("heuristic_chased_dist:",dist)
      # print(f"start states {startState}")
      # print("heuristic(startState):",heuristic(startState))
      # print()
    # myPQ.push(startNode,heuristic(startState))

    if heuristic_chased_dist<5:
      dist = heuristic(startState)+100-20*heuristic_chased_dist
    else:
      dist = heuristic(startState)+100-20*heuristic_chased_dist

    myPQ.push(startNode,dist)

    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, action, cost, path = node
        # print(cost)
        # print(f"visited list is {visited}")
        # print(f"best_g list is {best_g}")
        if (not state in visited) or cost < best_g.get(str(state)):
            visited.add(state)
            best_g[str(state)]=cost
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                # print(actions)
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(node, action)])
                # heuristic function
                myPQ.push(newNode,heuristic(succState)+cost+succCost)

                heuristic_chased_dist = 0
                if self.chasedByGohst_sensitive(gameState, 5):
                  chasing_dis = self.get_op_maze_distance(gameState)
                  if chasing_dis[0]!= None and chasing_dis[1]!=None:
                    if chasing_dis[0]<=chasing_dis[1]:
                      heuristic_chased_dist = chasing_dis[0]
                    else:
                      heuristic_chased_dist = chasing_dis[1]
                  else:
                    if chasing_dis[0]!= None:
                      heuristic_chased_dist = chasing_dis[0]
                    if chasing_dis[1]!=None:
                      heuristic_chased_dist = chasing_dis[1]
                
                if heuristic_chased_dist == 0:
                  myPQ.push(newNode,heuristic(succState)-heuristic_chased_dist+cost+succCost)
                  # print("chased = 0",heuristic(succState)-heuristic_chased_dist+cost+succCost)
                elif heuristic_chased_dist<5:
                  myPQ.push(newNode,heuristic(succState)+100+cost+succCost-20*heuristic_chased_dist)
                  # print("chased <3:",heuristic(succState)+100+cost+succCost-20*heuristic_chased_dist)
                else:
                  myPQ.push(newNode,heuristic(succState)+100+cost+succCost-20*heuristic_chased_dist) 
                  # print("chased >=3:",heuristic(succState)+100+cost+succCost-20*heuristic_chased_dist)
    return []
  
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
        if dis_from_op[0] <= 2 and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[0] <= 2 and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        
      if dis_from_op[1]!=None:
      #   dis_from_op1 = self.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
        if dis_from_op[1]<= 2 and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[1]<= 2 and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True     
      return False
  
  def chasedByGohst_sensitive(self, gameState, sensitive):
      
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
        if dis_from_op[0] <= sensitive and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[0] <= sensitive and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chased by op")
          return True
        
      if dis_from_op[1]!=None:
      #   dis_from_op1 = self.getMazeDistance(agent_my_pos, agent_op_pos1)
        # print("agent_op_pos:", agent_op_pos, "dis_from_op", dis_from_op1)
        if dis_from_op[1]<= sensitive and scaredTimer == 0 and gameState.getAgentState(self.index).isPacman \
            and not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True
        elif dis_from_op[1]<= sensitive and scaredTimer_m > 0 and not gameState.getAgentState(self.index).isPacman \
            and gameState.getAgentState(self.enemy_pos[1]).isPacman:
          # print("chased by op")
          return True     
      return False

  def chasingPacman(self, gameState):
      
      # get myself's scared timer
      ghost_state = gameState.getAgentState(self.index)
      scaredTimer = ghost_state.scaredTimer

      # get opponent's scared timer
      ghost_state_op = gameState.getAgentState(self.enemy_pos[0])
      scaredTimer_op = ghost_state_op.scaredTimer
      dis_from_op = self.get_op_maze_distance(gameState)
      
      if dis_from_op[0]!=None:
        if dis_from_op[0] <= 4  and scaredTimer == 0 and gameState.getAgentState(self.enemy_pos[0]).isPacman and\
                    not gameState.getAgentState(self.index).isPacman:
          # print("chasing op")
          return True
        elif dis_from_op[0] <= 4  and scaredTimer_op >0 and gameState.getAgentState(self.index).isPacman and\
                     not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chasing op")
          return True
        
      if dis_from_op[1]!=None:
        if dis_from_op[1]<= 4 and scaredTimer == 0 and gameState.getAgentState(self.enemy_pos[1]).isPacman and \
                    not gameState.getAgentState(self.index).isPacman:
          # print("chasing op")
          return True
        elif dis_from_op[1]<= 4 and scaredTimer_op > 0 and gameState.getAgentState(self.index).isPacman and \
                    not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chasing op")
            # not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          return True
      
      return False
  
  def chasingPacman_sensitive(self, gameState, sensitive):
      
      # get myself's scared timer
      ghost_state = gameState.getAgentState(self.index)
      scaredTimer = ghost_state.scaredTimer

      # get opponent's scared timer
      ghost_state_op = gameState.getAgentState(self.enemy_pos[0])
      scaredTimer_op = ghost_state_op.scaredTimer
      dis_from_op = self.get_op_maze_distance(gameState)
      
      if dis_from_op[0]!=None:
        if dis_from_op[0] <= sensitive  and scaredTimer == 0 and gameState.getAgentState(self.enemy_pos[0]).isPacman and\
                    not gameState.getAgentState(self.index).isPacman:
          # print("chasing op")
          return True
        elif dis_from_op[0] <= sensitive  and scaredTimer_op >0 and gameState.getAgentState(self.index).isPacman and\
                     not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chasing op")
          return True
        
      if dis_from_op[1]!=None:
        if dis_from_op[1]<= sensitive and scaredTimer == 0 and gameState.getAgentState(self.enemy_pos[1]).isPacman and \
                    not gameState.getAgentState(self.index).isPacman:
          # print("chasing op")
          return True
        elif dis_from_op[1]<= sensitive and scaredTimer_op > 0 and gameState.getAgentState(self.index).isPacman and \
                    not gameState.getAgentState(self.enemy_pos[0]).isPacman:
          # print("chasing op")
            # not gameState.getAgentState(self.enemy_pos[1]).isPacman:
          return True
      
      return False
  
  def isPacman(self, gameState):
    agent_my_pos = gameState.getAgentPosition(self.index)
    print(agent_my_pos)
    width = gameState.data.layout.width
    if agent_my_pos[0]<=(int(width/2)-1) and not self.red:
      return True
    elif agent_my_pos[0]>=int(width/2) and self.red:
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

class MyAgent1(MyAgent):
    # def registerInitialState(self, gameState):
    #   CaptureAgent.registerInitialState(self, gameState)
    #   self.carrying = 0
    #   self.current_target = None
    #   self.teammate_target = None
    #   self.boundary = self.getBoundary(gameState)
    #   # print(self.boundary)
    #   self.max_capacity = 5

    def chooseAction(self, gameState):
      
      if self.chasingPacman(gameState):
        pacmans= []
        for op_pos in self.enemy_pos:
          pos = gameState.getAgentPosition(op_pos)
          if pos!=None:
            pacmans.append(pos)
        self.current_target = self.getClosestPos(gameState, pacmans)

      if self.chasedByGohst(gameState) == True:
          dists = []
          # op_dist = self.get_op_maze_distance(gameState)[0]
          # op1_dist = self.get_op_maze_distance(gameState)[1]
          if gameState.getAgentPosition(self.enemy_pos[0])!=None:
            dists.append(gameState.getAgentPosition(self.enemy_pos[0]))
          if gameState.getAgentPosition(self.enemy_pos[1])!=None:
            dists.append(gameState.getAgentPosition(self.enemy_pos[1]))
          caps = self.getCapsules(gameState)
          for cap in caps:
            if cap != None:
              dists.append(cap)
          print(dists)
          closest = self.getClosestPos(gameState,dists)

          if closest in caps:
            self.current_target = self.getClosestPos(gameState,caps)
          else: 
            self.current_target = self.getClosestPos(gameState,self.boundary)
          agents_current_target[self.index] = self.current_target

      if not self.current_target == None:
        # if agent already have a goal
        pass

      elif self.carrying == self.max_capacity or len(self.getFood(gameState).asList())<=2:
        # if agent got all the food it needed
        # it will reach to the closest boundary with A* search (manhattanDistance as heuristic)
        self.current_target = self.getClosestPos(gameState,self.boundary)

        # # store the target information to the map "agents_current_target"
        agents_current_target[self.index] = self.current_target

      else:
        # if agent have more capacity to carry
        # it will find the next closest food
        foodGrid = self.getFood(gameState)
        # print(foodGrid.asList())
        
        # get the current food other than targetted by teammates
        current_food = []
        for pos in foodGrid.asList():
          # print(agents_current_target.values())
          if pos not in agents_current_target.values():
              current_food.append(pos)

        self.current_target = self.getClosestPos(gameState,current_food)

        # store the target information to the map "agents_current_target"
        agents_current_target[self.index] = self.current_target

      problem = PositionSearchProblem(gameState,self.current_target,self.index)
      path = self.aStarSearch(problem, gameState)

      if path == []:
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
      
      else:
        action = path[0]
        dx,dy = game.Actions.directionToVector(action)
        x,y = gameState.getAgentState(self.index).getPosition()
        new_x,new_y = int(x+dx),int(y+dy)
        if (new_x,new_y) == self.current_target:
          self.current_target = None
        if self.getFood(gameState)[new_x][new_y]:
          self.carrying +=1
        elif (new_x,new_y) in self.boundary:
          self.carrying = 0
        return path[0]
    pass

class MyAgent2(MyAgent):
    # def registerInitialState(self, gameState):
    #   CaptureAgent.registerInitialState(self, gameState)
    #   self.carrying = 0
    #   self.current_target = None
    #   self.teammate_target = None
    #   self.boundary = self.getBoundary(gameState)
    #   # print(self.boundary)
    #   self.max_capacity = 5

    def chooseAction(self, gameState):
      
      if self.chasingPacman(gameState):
        pacmans= []
        for op_pos in self.enemy_pos:
          pos = gameState.getAgentPosition(op_pos)
          if pos!=None:
            pacmans.append(pos)
        self.current_target = self.getClosestPos(gameState, pacmans)

      if self.chasedByGohst(gameState) == True:
          dists = []
          # op_dist = self.get_op_maze_distance(gameState)[0]
          # op1_dist = self.get_op_maze_distance(gameState)[1]
          if gameState.getAgentPosition(self.enemy_pos[0])!=None:
            dists.append(gameState.getAgentPosition(self.enemy_pos[0]))
          if gameState.getAgentPosition(self.enemy_pos[1])!=None:
            dists.append(gameState.getAgentPosition(self.enemy_pos[1]))
          caps = self.getCapsules(gameState)
          for cap in caps:
            if cap != None:
              dists.append(cap)
          print(dists)
          closest = self.getClosestPos(gameState,dists)

          if closest in caps:
            self.current_target = self.getClosestPos(gameState,caps)
          else: 
            self.current_target = self.getClosestPos(gameState,self.boundary)
          agents_current_target[self.index] = self.current_target

      if not self.current_target == None:
        # if agent already have a goal
        pass

      elif self.carrying == self.max_capacity or len(self.getFood(gameState).asList())<=2:
        # if agent got all the food it needed
        # it will reach to the closest boundary with A* search (manhattanDistance as heuristic)
        self.current_target = self.getClosestPos(gameState,self.boundary)

        # # store the target information to the map "agents_current_target"
        agents_current_target[self.index] = self.current_target

      else:
        # if agent have more capacity to carry
        # it will find the next closest food
        foodGrid = self.getFood(gameState)
        # print(foodGrid.asList())
        
        # get the current food other than targetted by teammates
        current_food = []
        for pos in foodGrid.asList():
          # print(agents_current_target.values())
          if pos not in agents_current_target.values():
              current_food.append(pos)

        self.current_target = self.getClosestPos(gameState,current_food)

        # store the target information to the map "agents_current_target"
        agents_current_target[self.index] = self.current_target

      problem = PositionSearchProblem(gameState,self.current_target,self.index)
      path = self.aStarSearch(problem, gameState)

      if path == []:
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
      
      else:
        action = path[0]
        dx,dy = game.Actions.directionToVector(action)
        x,y = gameState.getAgentState(self.index).getPosition()
        new_x,new_y = int(x+dx),int(y+dy)
        if (new_x,new_y) == self.current_target:
          self.current_target = None
        if self.getFood(gameState)[new_x][new_y]:
          self.carrying +=1
        elif (new_x,new_y) in self.boundary:
          self.carrying = 0
        return path[0]
    pass
