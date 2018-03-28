# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  
      

        newFoodList = newFood.asList()
        score = successorGameState.getScore() 
        distSum = 0
        for food in newFoodList:
          distSum += manhattanDistance(newPos,food)
        if len(newFoodList) > 0:
          avgDist = float(distSum) / len(newFoodList)
          score += ( 1.0 / avgDist ) + ( 1.0 / len(newFoodList) )
        ghostDist = 0
        for ghost in newGhostStates:
          ghostDist += manhattanDistance(newPos,ghost.getPosition())
        if len(newGhostStates) > 0:
          avgghostDist = float(ghostDist) / len(newGhostStates)
          avgScaredTime = float(sum(newScaredTimes)) / len(newGhostStates)
          if avgScaredTime > 0:
            score += ( 1.0 / avgghostDist )
          else:
            if avgghostDist < 5.0:
              score += avgghostDist

        "*** YOUR CODE HERE ***"
        #score = successorGameState.getScore() + 
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
      #print "Number of Agents : ",gameState.getNumAgents()
      bestScore, bestMove = self.maxF(gameState,self.depth)
      return bestMove


    def  maxF(self,gameState,depth):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), "noMove"
      moves = gameState.getLegalActions()
      scores = []
      for move in moves:
        successor = gameState.generateSuccessor(self.index,move)
        score = self.minF(successor, 1 , depth)
        scores.append(score[0])
      #print "Max : ",scores
      bestScore = max(scores)
      #print "Best Score : ",bestScore
      bestMoveIndex = 0
      for i in range(len(scores)):
        if scores[i] == bestScore:
          bestMoveIndex = i
          break
      bestMove = moves[bestMoveIndex]
      return bestScore,bestMove

    def minF(self,gameState,index,depth):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState),"noMove"
      moves = gameState.getLegalActions(index)
      scores = []
      for move in moves:
        successor = gameState.generateSuccessor(index,move)
        if index < gameState.getNumAgents() - 1:
          score = self.minF(successor,index+1,depth)
          scores.append(score[0])
        else:
          score = self.maxF(successor,depth-1)
          scores.append(score[0])
      #print "Min : ",scores
      bestScore = min(scores)
      #print "Best Score : ",bestScore
      bestMoveIndex = 0
      for i in range(len(scores)):
        if scores[i] == bestScore:
          bestMoveIndex = i
          break
      bestMove = moves[bestMoveIndex]
      return bestScore,bestMove






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestMove = self.maxF(gameState,self.depth,float("-inf"),float("inf"))
        return bestMove

    def maxF(self,gameState,depth,alpha,beta):
      import math
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), "noMove"
      moves = gameState.getLegalActions()
      bestMove = None
      score = float("-inf")
      for move in moves:
        successor = gameState.generateSuccessor(self.index,move)
        tempScore = self.minF(successor, 1 , depth,alpha,beta)
        if tempScore[0] > score:
          score = tempScore[0]
          bestMove = move
        if score > beta:
          return score,bestMove
        alpha = max(score,alpha)
      return score,bestMove 

        
      return bestScore,bestMove

    def minF(self,gameState,index,depth,alpha,beta):
      import math
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState),"noMove"
      moves = gameState.getLegalActions(index)
      score = float("inf")
      bestMove = None
      for move in moves:
        successor = gameState.generateSuccessor(index,move)
        if index < gameState.getNumAgents() - 1:
          tempScore = self.minF(successor,index+1,depth,alpha,beta)
        else:
          tempScore = self.maxF(successor,depth-1,alpha,beta)
        if tempScore[0] < score:
          score = tempScore[0]
          bestMove = move
        if score < alpha:
          return score,bestMove
        beta = min(beta,score)
      return score,bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestMove = self.expectimax(gameState,self.depth,0)
        return bestMove
        #return self.getActionHelper(gameState, self.depth, 0)[1]


    def expectimax(self,gameState,depth, agentIndex):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState),"noMove"
      else:
        if agentIndex == 0:
          score = -float("inf")
        else:
          score = 0
        moves = gameState.getLegalActions(agentIndex)
        bestMove = None
        for move in moves:
          if agentIndex == gameState.getNumAgents() - 1:
            result = self.expectimax(gameState.generateSuccessor(agentIndex,move) , depth-1 ,0 )
          else:
            result = self.expectimax(gameState.generateSuccessor(agentIndex,move) , depth , agentIndex + 1)
          if agentIndex == 0:
            if result[0] > score:
              score = result[0]
              bestMove = move
          else:
            score += 1.0 / len(moves) * result[0]
            bestMove = move
        return score,bestMove

    '''
    def maxF(self,gameState,depth):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState) , "noMove"
      moves = gameState.getLegalActions()
      score = float("-inf")
      bestMove = None
      for move in moves:
        successor = gameState.generateSuccessor(self.index,move)
        tempScore = self.chance(successor,1,depth)
        if tempScore[0] > score:
          score = tempScore[0]
          bestMove = move
      return score,bestMove

    def chance(self,gameState,index,depth):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState) , "noMove"
      moves = gameState.getLegalActions(index)
      score = 0
      for move in moves:
        successor = gameState.generateSuccessor(index,move)
        if index < gameState.getNumAgents() - 1:
          tempScore = self.chance(gameState,index+1,depth)
        else:
          tempScore = self.maxF(gameState,depth-1)
        score += 1.0 / len(moves) * tempScore[0]
        bestMove = move
      # import random
      # bestMove = random.choice(moves)
      #print "Returning Score : ",score,"  Index : ",index
      return score,bestMove

      '''

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The state which has lower food left with less distance with
       the food and some good distance from the ghosts can  be defined as the good
       state. So first we are checking how bad the state is then negating the value
       to check how good the state is. If ghost is very near from the pacman agent 
       then it is very bad state. If we are next to capsule it is also goood to it the
       capsule, so less numbers of capsules gives better rating to the state. So by some
       linear combinations of these parameters, We came to the following definition of the
       evalation function.
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance
    score = 0
    foodPos = currentGameState.getFood().asList()
    numFood = currentGameState.getNumFood()
    minFoodDist = float("inf")
    currPos = currentGameState.getPacmanPosition()
    for food in foodPos:
      dist = manhattanDistance(currPos,food)
      if dist < minFoodDist:
        minFoodDist = dist
    if minFoodDist != float("inf"):
      score += 10 * minFoodDist
    score += 10000 * numFood
    ghostPositions = currentGameState.getGhostPositions()
    for ghost in ghostPositions:
      if manhattanDistance(currPos,ghost) < 2:
        score = 9999999999999
      else:
        score -= 0.0001 * manhattanDistance(currPos,ghost)
    numCapsules = len(currentGameState.getCapsules())
    score += 100000 * numCapsules
    return -score

# Abbreviation
better = betterEvaluationFunction
