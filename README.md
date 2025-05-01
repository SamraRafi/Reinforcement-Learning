# Tic Tac Toe with Q Learning 

The TicTacToe player in the problem uses a simple Q learning algorithm to
play N game sequences and record its performance against a similar agent.
Each player maintains a cache which records its moves and states for a game
sequence and a Q table which learns the reward function for an action given
a state Q(st ,a) over total number of game sequences.
The Q table updates after each game iteration following a backtracking
algorithm from the terminating to the first state. The Bellman Temporal
Difference Equation for Q-update is as follows:
Qt+1(st, at) = Qt(st, at) + ( r+  maxaQ(st+1, a) - Qt(st, at) ) 

The code contains a Player class, a Q-Table class, and a TicTacToe class that mimics the game's dynamics and environment
Each Player has a Q_Table object.

### Parallelization with MultiProcessing
An attempt to parallelize the above implementation was made using
multiprocessing. P processors were run asynchronously and executed N/P
game sequences in parallel.
