"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    '''
    This evaluation function gives priorities moves to the center, moves that reflect the opponent's current position, and then moves that will maximize the player's available moves over the opponent's available moves.
    '''
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    # Gathers the dimensions of the board, initializes the score, and identifies the current positions of the players
    w, h = game.width, game.height
    score = 0
    position_of_player = game.get_player_location(player)
    position_of_opponent = game.get_player_location(game.get_opponent(player))
    
    # Identifies available moves for each player
    own_avail_moves = game.get_legal_moves(player)
    opp_avail_moves = game.get_legal_moves(game.get_opponent(player)) 
    
    # Determines if there is a center. Then, it checks to see if the center position is available. If it is available, player will prioritize the move towards the center.
    position_center = [(w/2), (h/2)]    
    if position_center[0].is_integer() and position_center[1].is_integer():
        position_center = [(w/2), (h/2)]
        for move in own_avail_moves:
            if move in position_center:
                score += 2          
    
    # This prioritizes moves that would mirror the opponent's location if the position is available
    reflection_move = [(w - position_of_opponent[0]), (h - position_of_opponent[1])]
    for move in own_avail_moves:
        if move in reflection_move:
            score += 1  
    
    # Prioritizes moves that will maximize the player's available moves and minimize the opponent's moves
    num_own_moves = len(own_avail_moves)
    num_opp_moves = len(opp_avail_moves)
    diff_moves = num_own_moves - num_opp_moves
    score += diff_moves
    
    return float(score)
     
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    '''
    This evaluation function outputs a score equal to the difference between the player's score and double the opponent's score
    '''
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    '''
    This heuristic attempts to box an opponent into one quadrant of the gameboard
    '''
    
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    # Gathers the dimensions of the board, initializes the score, and identifies the current positions of the players
    w, h = game.width, game.height
    score = 0
    position_of_player = game.get_player_location(player)
    position_of_opponent = game.get_player_location(game.get_opponent(player))
    
    # Identifies available moves for each player
    own_avail_moves = game.get_legal_moves(player)
    opp_avail_moves = game.get_legal_moves(game.get_opponent(player)) 
    
    # Determines if there is a center. Then, it checks to see if the center position is available. If it is available, player will prioritize the move towards the center.
    position_center = [(w/2), (h/2)]    
    if position_center[0].is_integer() and position_center[1].is_integer():
        position_center = [(w/2), (h/2)]
        for move in own_avail_moves:
            if move in position_center:
                score += 2  
    
    # This prioritizes moves that would box an opponent into one quadrant of the gameboard
    x_axis_move = [position_of_opponent[0], (h/2)]
    y_axis_move = [(w/2), position_of_opponent[1]]
    for move in own_avail_moves:
        if move in x_axis_move:
            score += 1
        if move in y_axis_move:
            score += 1
    
    return float(score)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!

        return self._minimax(game, depth)[-1]

    def _minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, (int, int))
            A tuple of the score and the board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        
        # Determine how many legal moves are available
        legal_moves = game.get_legal_moves()
        
        # Returns the board state and score once the bottom of the tree is reached (depth equals zero)
        if depth == 0:
            return(self.score(game, self), None)
        
        best_move  = (-1, -1)
        
        # Execute minimax loop
        if game.active_player == self:
            best_score = float('-inf')
            max_or_min = max
        else:
            best_score = float('inf')
            max_or_min = min
            
        for move in legal_moves:
            child_board = game.forecast_move(move)
            future_score = self._minimax(child_board, depth - 1)[0]
            if max_or_min(best_score, future_score) == future_score:
                best_move = move
                best_score = max_or_min(best_score, future_score)
                
        return (best_score, best_move)
    
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            selected = [(float("-inf"), (-1, -1))]
            while True:
                best_move = self.alphabeta(game, depth)
                selected.append(best_move)
                depth = depth + 1
 
        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
          
        return self._alphabeta(game, depth, alpha, beta)[1]
    
    def _alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
            
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, (int, int))
            A tuple of the score and the board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        
        # Determine how many legal moves are available
        legal_moves = game.get_legal_moves()
        
        # Returns the board state and score once the bottom of the tree is reached (depth equals zero)
        if depth == 0:
            return self.score(game, self), (-1, -1)
        
        best_move  = (-1, -1)
        
        # Initialize alpha and beta values
        if game.active_player == self:
            best_score = float('-inf')
            max_or_min = max
            is_alpha = True
        else:
            best_score = float('inf')
            max_or_min = min
            is_alpha = False
        
        # Execute alpha beta pruning loop
        for move in legal_moves:
            child_board = game.forecast_move(move)
            future_score = self._alphabeta(child_board, depth - 1, alpha, beta)[0]
            
            if best_score == future_score:
                continue
            
            if future_score == max_or_min(best_score, future_score):
                best_score = future_score
                best_move = move
            
            if is_alpha:
                if best_score >= beta:
                    return best_score, best_move
                else:
                    alpha = max(best_score, alpha)
            else:
                if best_score <= alpha:
                    return best_score, best_move
                else:
                    beta = min(best_score, beta)
                                 
        return (best_score, best_move)