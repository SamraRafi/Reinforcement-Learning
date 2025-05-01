
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import random


class TicTacToe:
    def __init__(self, dimension):
        self.dimension = dimension
        self.board_size = dimension**2
        self.reward=0
        #game memory for each game
        self.board = np.zeros(self.board_size)
        #self.current_state = np.zeros(dimension**2) #first game state is always empty
        self.gameCache = list(tuple(np.arange(dimension**2))) #sequence of game states from initial to final state
        self.legal_moves = list(np.arange(dimension**2))#moves remining in the game 0-8
        self.taken_moves = [] #sequence of actions 0-8
        
        self.is_end_game = False
        self.winner = None

    def isDraw(self):
        if self.winner ==None:
            return (len(self.legal_moves)==0)

    def hasALine(self):
        size = self.dimension
        board = self.board.reshape(size, size)
        horizontal_sums = board.sum(axis=1)
        vertical_sums = board.sum(axis=0)
        #sum along diagonal and backdiagonal of the board
        d= np.trace(board)
        bd= np.trace(np.fliplr(board))
        for i in range(self.dimension):
            h = horizontal_sums[i]
            v = vertical_sums[i]
            if any(val == size for val in (h, v, d, bd)):
                self.winner = 1
                return True
            elif any(val == -size for val in (h, v, d, bd)):
                self.winner =-1
                return True
        return False

    def isEndGame(self):
        if (self.isDraw() or self.hasALine()):
            self.is_end_game = True
            return True
        else:
            return False

    def getGameInfo(self):
        if self.is_end_game: 
            reward = self.getReward()
            return[self.getTakenMoves(), reward]

    def generateState(self):
        self.board = np.random.choice([-1, 0, 1], size=self.board_size)
        return self.board

    def resetBoard(self):
        self.board = np.zeros(self.board_size)
        self.gameCache = list(np.arange(self.board_size)) #sequence of game states from initial to final state
        self.legal_moves = list(np.arange(self.board_size))#moves remaining in the game [0-8]
        self.taken_moves = [] #sequence of actions [0-8]
        
        self.is_end_game = False
        self.winner = None

    # def printBoard(self):
    #     print(self.board.reshape(self.dimension, self.dimension))

    def placeAction(self, ID, action:int):
        
        if action in self.legal_moves:
            self.board[action] = ID  # reflect move on tictactoe board
            self.taken_moves.append(action)
            self.legal_moves.remove(action)
            self.gameCache.append(tuple(self.board))
            self.setReward(ID)
            #check if move terminated the game
        else:
            self.reward = -100

    # def placeAction(self, ID, action:int):
    #     self.board[action] = ID  # reflect move on tictactoe board
    #     self.taken_moves.append(action)
    #     self.legal_moves.remove(action)
    #     self.gameCache.append(tuple(self.board))

        
    def setReward(self, ID):
        if(self.hasALine()):
            self.reward = int(ID) *10
        elif(self.isDraw()):
            self.reward = 0.5
        else:
            self.reward = 0
        

    def getReward(self):
        return self.reward
    
    def getState(self):
        return self.board

    def getPossibleMoves(self):
        return self.legal_moves

    def getTakenMoves(self):
        return self.taken_moves

    def getBoardSize(self):
        return self.board_size



class Q_Table():
    def __init__(self, learningRate, discountFactor):
        self.Qt = {}
        self.learning_rate = learningRate
        self.discount_factor = discountFactor

    def addState(self, state):

        self.Qt[state] = []

    def addAction(self, state, act):
        self.Qt[state].append((act,0))

    #takes the list of tuples of action-values pairs for the state,
    def updateQtable(self, game_states, actions, reward):
        game_states= reversed(game_states)
        actions = reversed(actions)
        next_state = None
        for state, action in zip(game_states, actions):

            #make sure both action and state are present in the Qtable
            if not self.stateInQtable(state):
                self.addState(state)
                self.addAction(state, action)
            else:
                if not self.st_ActInQtable(action, state):
                    self.addAction(state, action)

            self.updateQval(state, action, next_state, reward)
            next_state = state

        return self.Qt

    def updateQval(self, state, act, n_state, reward):
        if not (n_state == None):
            next_q = max(a[1] for a in self.Qt[n_state])
            reward = 0
        else:
            next_q = 0

        for a in self.Qt[state]:
            if a[0]==act:
                current_q = float(a[1])

        # update q value
        temp_diff = reward + (self.discount_factor*next_q) - current_q
        updated_q = current_q + self.learning_rate*(temp_diff)
        for i, a in enumerate(self.Qt[state]):
            if a[0] == act:
                self.Qt[state][i] = (a[0], updated_q)
    #end


    def getQval(self, boardstate):
        return self.Qt[boardstate]

    def st_ActInQtable(self, act, state):
        acts = [a[0] for a in self.Qt[state]]
        if act in acts: return True
        else: return False

    def stateInQtable(self, state):
        if state in self.Qt: return True
        else:return False

class Player():
    def __init__(self, ID, char, epsilon, learningRate, discountFactor):
        self.ID = ID
        self.char = char
        self.Q_table = Q_Table(learningRate, discountFactor)
        self.epsilon = epsilon
        self.current_state = None
        self.myGameCache = list(tuple(np.arange(3**2)))
        self.myActions =[]

    def DqnMove(self, tictactoe, q_mov):
        if(tictactoe.isEndGame()):
            return 
        
        tictactoe.placeAction(self.ID, q_mov)
        return 
    
    def RandMove(self, tictactoe):
        if(tictactoe.isEndGame()):
            return 
        
        action = random.choice(tictactoe.getPossibleMoves())
        tictactoe.placeAction(self.ID, action)
        return 

    def makeMove(self, tictactoe):
        #check if the game has ended
        if (tictactoe.isEndGame()):
            return tictactoe
        
        #else
        self.myGameCache.append(tuple(tictactoe.getState()))
        self.current_state = tictactoe.getState()
        poss_mov = tictactoe.getPossibleMoves()    

        if (np.random.random() < self.epsilon and self.Q_table.stateInQtable(self.current_state)):
            #choose from qtable
            action =None
            qT = self.Q_table.getQval(self.current_state)
            best_qval = max(t[1]for t in qT)
            #list of actions with the max qvalue
            q_mov = [t[0] for t in qT if t[1]==best_qval]
            poss_qmov = list(set(poss_mov) & set(q_mov))
            if(poss_qmov):
                action = random.choice(poss_qmov)
                tictactoe.placeAction(self.ID, action)
                self.myActions.append(action)
                return tictactoe
        
        #choose random move
        action = random.choice(poss_mov)
        tictactoe.placeAction(self.ID, action)
        self.myActions.append(action)
        return tictactoe


    def getQ_Table(self):
        return self.Q_table.Qt
    
    def setQ_Table(self, qtable):
        self.Q_table.Qt = qtable

    def updateQ_Table(self, tictactoe):
        if (tictactoe.isEndGame()):
            #info = tictactoe.getGameInfo()
            self.Q_table.updateQtable(self.myGameCache, self.myActions, tictactoe.getReward(self.getID))

    def getID(self):
        return self.ID

    def getCurrentState(self):
        return self.current_state


class main():
    def PreTrained():
        epsilon =0.9
        dimension = 3
        tictactoe = TicTacToe(dimension)
        AI = Player(1, "X", epsilon, 0.9, 0.9)
        NPC = Player(-1, "O", epsilon, 0.9, 0.9)
        
        num = 100000
        win_rate=0
        lose_rate=0
        draw_rate=0

        for episode in range(num):
            while not tictactoe.isEndGame():
                tictactoe = AI.makeMove(tictactoe)
                tictactoe = NPC.makeMove(tictactoe)
            AI.updateQ_Table(tictactoe)
            NPC.updateQ_Table(tictactoe)
            
            # Update rates
            if tictactoe.winner == AI.getID():
                win_rate+=1
            elif tictactoe.isDraw():
                draw_rate+=1
            else:
                lose_rate+=1
            
            # Reset the game for the next episode
            tictactoe.resetBoard()

        return AI
    
    Agent = PreTrained()

main()