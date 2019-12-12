# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# Simple reinforcement learning algorithm for learning tic-tac-toe
# Use the update rule: V(s) = V(s) + alpha*(V(s') - V(s))
# Use the epsilon-greedy policy:
#   action|s = argmax[over all actions possible from state s]{ V(s) }  if rand > epsilon
#   action|s = select random action from possible actions from state s if rand < epsilon
#
#
# INTERESTING THINGS TO TRY:
#
# Currently, both agents use the same learning strategy while they play against each other.
# What if they have different learning rates?
# What if they have different epsilons? (probability of exploring)
#   Who will converge faster?
# What if one agent doesn't learn at all?
#   Poses an interesting philosophical question: If there's no one around to challenge you,
#   can you reach your maximum potential?
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

WIDTH = 4
HEIGHT = 4


class Agent:
  def __init__(self, eps=0.1, alpha=0.5):
    self.eps = eps # probability of choosing random action instead of greedy
    self.alpha = alpha # learning rate
    self.verbose = False
    self.state_history = []
  
  def setV(self, V):
    self.V = V

  def set_symbol(self, sym):
    self.sym = sym

  def set_verbose(self, v):
    # if true, will print values for each position on the board
    self.verbose = v

  def reset_history(self):
    self.state_history = []

  def take_action(self, env):
    # choose an action based on epsilon-greedy strategy
    r = np.random.rand()
    best_state = None
    if r < self.eps:
      # take a random action
      if self.verbose:
        print("Taking a random action")

      possible_moves = []
      for i in range(HEIGHT):
        for j in range(WIDTH):
          if env.is_covered(i, j):
            possible_moves.append((i, j))
      idx = np.random.choice(len(possible_moves))
      next_move = possible_moves[idx]
    else:
      # choose the best action based on current values of states
      # loop through all possible moves, get their values
      # keep track of the best value
      pos2value = {} # for debugging
      next_move = None
      best_value = -1
      for i in range(HEIGHT):
        for j in range(WIDTH):
          if env.is_empty(i, j):
            # what is the state if we made this move?
            env.reveal(i,j)
            state = env.get_state()
            env.cover(i,j)                # don't forget to change it back!
            pos2value[(i,j)] = self.V[state]
            if self.V[state] > best_value:
              best_value = self.V[state]
              best_state = state
              next_move = (i, j)

      # if verbose, draw the board w/ the values
      if self.verbose:
        print("Taking a greedy action")
        for j in range(WIDTH):
          print("--", end="")
        print("")
        for i in range(HEIGHT):
          for j in range(WIDTH):
            if env.is_covered(i, j):
              # print the value
              print("%.2f" % pos2value[(i,j)], end="")
            else:
              print("  ", end="")
              if self.board[i,j] == self.bomb:
                print("X", end="")
              elif self.board[i,j] > self.bomb:
                print(" ", end="")
              else:
                print(self.board[i,j], end="")
          print("")
        for j in range(WIDTH):
          print("--", end="")
        print("")

    # make the move
    env.board[next_move[0], next_move[1]] = self.sym

  def update_state_history(self, s):
    # cannot put this in take_action, because take_action only happens
    # once every other iteration for each player
    # state history needs to be updated every iteration
    # s = env.get_state() # don't want to do this twice so pass it in
    self.state_history.append(s)

  def update(self, env):
    # we want to BACKTRACK over the states, so that:
    # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
    # where V(next_state) = reward if it's the most current state
    #
    # NOTE: we ONLY do this at the end of an episode
    # not so for all the algorithms we will study
    reward = env.reward(self.sym)
    target = reward
    for prev in reversed(self.state_history):
      value = self.V[prev] + self.alpha * ( target - self.V[prev] )
      self.V[prev] = value
      target = value
    self.reset_history()


# this class represents a minesweeper game
class Environment:
  def __init__(self):
    self.board = np.zeros((WIDTH, HEIGHT))
    self.unknown = 10   # represents an undiscovered field
    self.bomb = 9       # represents a bomb on the board
    self.won = False
    self.ended = False
    self.num_states = (self.unknown*2)**(WIDTH*HEIGHT)
    print("Number of states: ", self.num_states);
    self.setup_board(10)

  def setup_board(self, num_bombs):
    for i in range(HEIGHT):
      for j in range(WIDTH):
        self.board[i,j] = 0
    while num_bombs > 0:
      x = np.random.choice( WIDTH )
      y = np.random.choice( HEIGHT )
      if self.board[x,y] != self.bomb:
        num_bombs = num_bombs - 1
        self.board[x,y] = self.bomb
        for i in range( max(0, x-1), min(WIDTH, x+2) ):
          for j in range( max(0, y-1), min(HEIGHT, y+2) ):
            if self.board[i,j] != self.bomb:
              self.board[i,j] = self.board[i,j] + 1

  def is_revealed(self, i, j):
    return self.board[i,j] < self.unknown

  def is_covered(self, i, j):
    return self.board[i,j] >= self.unknown
  
  def reveal(self, i, j):
    if self.is_covered(i, j):
      self.board[i,j] = self.board[i,j] - self.unknown

  def cover(self, i, j):
    if self.is_revealed(i, j):
      self.board[i,j] = self.board[i,j] + self.unknown

  def reward(self, sym):
    # no reward until game is over
    if not self.game_over():
      return 0

    # if we get here, game is over
    # sym will be self.x or self.o
    return 1 if self.winner == sym else 0

  def get_state(self):
    # returns the current state, represented as an int
    # from 0...|S|-1, where S = set of all possible states
    # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
    # some states are not possible, e.g. all cells are x, but we ignore that detail
    # this is like finding the integer represented by a base-3 number
    k = 0
    h = 0
    for i in range( WIDTH ):
      for j in range( HEIGHT ):
        v = min( self.board[i,j], self.unknown )
        h += (20**k) * v
        k += 1
    return h

  def game_over(self, force_recalculate=False):
    # returns true if game over (either won or lost)
    # otherwise returns false
    # also sets 'winner' instance variable and 'ended' instance variable
    if not force_recalculate and self.ended:
      return self.ended

    self.won = False
    undiscovered_fields = False

    for i in range( WIDTH ):
      for j in range( HEIGHT ):
        if self.board[i,j] == self.bomb:
          self.won = False
          self.ended = True
          return self.won
        if self.board[i,j] > self.bomb:
          undiscovered_fields = True

    if not undiscovered_field:
      self.won = True
      self.ended = True
      return self.won

    return False

  # Example board
  # -----------------
  #  01X100001110001X
  #  121100002X200011
  #  X10000002X201110
  #  1100000011101X10
  # -----------------
  def draw_board(self):
    for j in range( WIDTH ):
      print("--", end="")
    print("")
    for i in range( HEIGHT ):
      for j in range( WIDTH ):
        print("  ", end="")
        if self.board[i,j] == self.bomb:
          print("X", end="")
        elif self.board[i,j] > self.bomb:
          print(" ", end="")
        else:
          print(self.board[i,j], end="")
      print("")
    for j in range(WIDTH):
      print("--", end="")
    print("")



class Human:
  def __init__(self):
    pass

  def set_symbol(self, sym):
    self.sym = sym

  def take_action(self, env):
    while True:
      # break if we make a legal move
      move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
      i, j = move.split(',')
      i = int(i)
      j = int(j)
      if not env.is_revealed(i, j):
        env.board[i,j] = env.board[i,j] - env.unknown
        break

  def update(self, env):
    pass

  def update_state_history(self, s):
    pass


# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
  results = []

  env.reveal(i,j)      # if new board it should be covered
  if j == WIDTH-1:
    # j goes back to 0, increase i, unless i = 2, then we are done
    if i == HEIGHT-1:
      # the board is full, collect results and return
      state = env.get_state()
      ended = env.game_over( force_recalculate = True )
      won = env.won
      results.append((state, won, ended))
    else:
      results += get_state_hash_and_winner(env, i + 1, 0)
  else:
    # increment j, i stays the same
    results += get_state_hash_and_winner(env, i, j + 1)

  return results

# play all possible games
# need to also store if game is over or not
# because we are going to initialize those values to 0.5
# NOTE: THIS IS SLOW because MANY possible games lead to the same outcome / state
# def get_state_hash_and_winner(env, turn='x'):
#   results = []

#   state = env.get_state()
#   # board_before = env.board.copy()
#   ended = env.game_over(force_recalculate=True)
#   won = env.won
#   results.append((state, won, ended))

#   # DEBUG
#   # if ended:
#   #   if won is not None and env.win_type.startswith('col'):
#   #     env.draw_board()
#   #     print "Winner:", 'x' if won == -1 else 'o', env.win_type
#   #     print "\n\n"
#   #     assert(np.all(board_before == env.board))

#   if not ended:
#     if turn == 'x':
#       sym = env.x
#       next_sym = 'o'
#     else:
#       sym = env.o
#       next_sym = 'x'

#     for i in xrange(HEIGHT):
#       for j in xrange(WIDTH):
#         if not env.is_revealed(i, j):
#           env.board[i,j] = env.board[i,j] - env.unknown
#           results += get_state_hash_and_winner(env, next_sym)
#           env.board[i,j] = env.board[i,j] + env.unknown          # reset it
#   return results


def initialV_x(env, state_winner_triples):
  # initialize state values as follows
  # if x wins, V(s) == 1
  # if x loses or draw, V(s) == 0
  # otherwise, V(s) = 0.5
  V = np.zeros(env.num_states)
  for state, won, ended in state_winner_triples:
    if ended:
      if won:
        v = 1
      else:
        v = 0
    else:
      v = 0.5
    V[state] = v
  return V


def play_game(player, env, draw=False):
  # loops until the game is over
  while not env.game_over():
    # draw the board before the user who wants to see it makes a move
    if draw:
      env.draw_board()

    # current player makes a move
    player.take_action(env)

    # update state histories
    state = env.get_state()
    player.update_state_history(state)

  if draw:
    env.draw_board()

  # do the value function update
  player.update(env)


if __name__ == '__main__':
  # train the agent
  player = Agent()

  # set initial V for player and p2
  env = Environment()
  state_winner_triples = get_state_hash_and_winner(env)

  Vx = initialV_x(env, state_winner_triples)
  player.setV(Vx)

  # give each player their symbol
  player.set_symbol(env.x)

  T = 10000
  for t in range(T):
    if t % 200 == 0:
      print(t)
    play_game(player, Environment())

  # play human vs. agent
  # do you think the agent learned to play the game well?
  human = Human()
  human.set_symbol(env.o)
  while True:
    player.set_verbose(True)
    play_game(player, human, Environment(), draw=2)
    # I made the agent player 1 because I wanted to see if it would
    # select the center as its starting move. If you want the agent
    # to go second you can switch the human and AI.
    answer = input("Play again? [Y/n]: ")
    if answer and answer.lower()[0] == 'n':
      break

