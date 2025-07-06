# Project 3 - Intro to AI (Summer 2025)
# Contributors: Yasaman Saatsaz, Gilbert Teng


# import statements
import numpy as np
import heapq # priority queue for A* search
import random


# === Ship generation ===
def generate_ship(D=30, p=0.5):

    # create an initial map (D x D) of blocked cells
    ship = np.full((D, D), 'B')

    # open a random cell and mark it as open
    start_r, start_c = random.randint(0, D - 1), random.randint(0, D - 1)
    ship[start_r, start_c] = 'O'

    # expand cells if they have exactly one open neighbor
    while True:
        # list for storing potential cells to open
        candidates = []

        # searchy by row and column
        for r in range(D):
            for c in range(D):

                # if cell is blocked
                if ship[r, c] == 'B':

                    # count number of open neighbors
                    open_neighbors = sum(1 for nr, nc in get_neighbors(r, c, D) if ship[nr, nc] == 'O')
                    
                    if open_neighbors == 1:
                        candidates.append((r, c)) # add to the list of candidates
        
        # if no cells left to open
        if not candidates:
            break
    
        # pick a random cell from the list of candidates and open it
        r, c = random.choice(candidates)
        ship[r, c] = 'O'

    # find the dead ends in the ship
    dead_ends = find_dead_ends(ship, D)

    # how many dead ends did we start with?
    initial_dead_ends = len(dead_ends)
    # want to reduce dead ends by a factor of p
    target_dead_ends = int(initial_dead_ends * p)

    
    while len(dead_ends) > target_dead_ends:
        # pick a random dead end cell
        r, c = random.choice(dead_ends)

        # find its blocked neighbors
        blocked_neighbors = [(nr, nc) for nr, nc in get_neighbors(r, c, D) if ship[nr, nc] == 'B']
        
        if blocked_neighbors:
            # randomly open one of the blocked neighbors
            nr, nc = random.choice(blocked_neighbors)
            ship[nr, nc] = 'O'
        dead_ends = find_dead_ends(ship, D) # recalculate dead ends after every expansion

    return ship


# === Ship generation (helper functions)

# get the neighbors of a cell at (r, c)
def get_neighbors(r, c, D):

    # list for storing neighbors' coordinates
    neighbors = []

    # for every adjacent cell in the 4 directions
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:

        # calculate the neighbor's coordinates
        nr, nc = r + dr, c + dc

        # is the neighbor within bounds? (edge case: current cell is at the edge of the map)
        if 0 <= nr < D and 0 <= nc < D:
            neighbors.append((nr, nc)) # if in bounds, add the neighbor to the list

    return neighbors


# finds dead ends in the ship (cells with only one open neighbor)
def find_dead_ends(ship, D):

    # list for storing coordinates of dead end cells
    dead_ends = []

    # search by row and column
    for r in range(D):
        for c in range(D):

            # if the cell is open
            if ship[r, c] == 'O':
                # count the number of open neighbors it has
                open_neighbors = sum(1 for nr, nc in get_neighbors(r, c, D) if ship[nr, nc] == 'O')
                
                # if only one open neighbor (dead end) -> add to the list
                if open_neighbors == 1:
                    dead_ends.append((r, c))

    return dead_ends


# FOR DEBUGGING - what does the generated ship look like?
def print_ship(ship):
    '''
    print each row of the ship
        [B] - blocked cell
        [O] - open cell
    '''
    for row in ship:
        print(''.join(f'[{cell}]' for cell in row))


# === Search algorithm (and helper functions) ===

'''
In project 1, we used BFS with a heuristic as our search algorithm.
But BFS took a long time to run with large ships (we had to use 50 x 50 for project 1).
When we used A* search in project 2, we noticed the runtime was much faster and the shortest path was still guaranteed.
So we're gonna use A* search for project 3 as well to reduce runtime costs while finding the rat in as few moves as possible.
The searching part itself is similar to what we did before in project 2.
'''

# helper function - manhattan distance between two cells (heuristic)
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

'''
To simplify the code a bit more and make it more flexible,
we just made reconstructing the search path as a separate helper function.
'''

# helper function - reconstructs the bot's path from the goal cell to the starting cell
def reconstruct_path(came_from, current):
    # the current cell is added to the bot's movement history
    path = [current]

    # start moving backwards through the bot's path
    while current in came_from:
        # update the bot's current cell with the previous cell
        current = came_from[current]
        path.append(current) # add the new current cell to the bot's movement history
    return path[::-1]  # reverse the path (start -> goal)


# A* search
def a_star_search(ship, start, goal):
    D = len(ship)  # ship dimension

    '''
    initialize priority queue with tuple (f_score, g_score, current_node)
        - f_score (f(n)) = g_score (g(n)) + heuristic (h(n))
        - g_score starts at 0 (distance from starting cell to next cell)
        - heuristic = manhattan_distance from start to goal
    '''

    # initialize the priority queue (open_set) with tuple (f_score, g_score, start)
    open_set = [(manhattan_distance(start, goal), 0, start)]
    came_from = {} # initialize dictionary for storing the path
    g_score = {start: 0} # the starting cell has 0 cost

    # while the queue is not empty (there are still cells to explore)
    while open_set:
        # pop the node with the lowest estimated total cost (f_score)
        # cost is g_score (real cost so far)
        # current is the node's position
        _, cost, current = heapq.heappop(open_set)

        # if the current cell is the goal cell, reconstruct the path
        if current == goal:
            return reconstruct_path(came_from, current)

        # store the current cell's coordinates
        r, c = current

        # loop over the 4 directions (we're gonna look at neighboring cells of the current cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # calculate the neighbor's coordinates
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc) # store as a tuple

            # check if the neighbor cell is within bounds and is an open cell
            if 0 <= nr < D and 0 <= nc < D and ship[nr][nc] == 'O':
                # calculate a tentative score for reaching this neighbor
                tentative_g = cost + 1

                # have we visited this neighbor? or is this path shorter than a previous path we found to it?
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # update the best known cost to reach this neighbor
                    g_score[neighbor] = tentative_g

                    # compute the estimated total cost to goal through the neighbor
                    f_score = tentative_g + manhattan_distance(neighbor, goal)

                    # add neighbor to the queue with its updated scores
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current # record that the neighbor was reached through the current cell

    return []  # No path found


# === Generating T array (and helper functions) ===

# compute value function through iterations for each (bot, rat) configuration in the 4D array T to calculate expected number of moves
def compute_value_function(ship, max_iters=1000, tol=1e-2):
    D = ship.shape[0] # ship dimension

    # creates 4D array T - each element represents a configuration of (bot, rat)
        # T[bx][by][rx][ry] - represents the expected number of moves to catch the rat starting at (bx, by) and (rx, ry)
    T = np.full((D, D, D, D), 1000.0)

    # list of all open cells - cells where the rat could be in or the bot can go to
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r][c] == 'O']

    # for every possible bot position
    for bx, by in open_cells:
        T[bx][by][bx][by] = 0.0  # If bot and rat are at same location, expected number of steps = 0

    # start value iteration
    for _ in range(max_iters):
        delta = 0.0 # maximum change in T-values
        new_T = T.copy() # copy for storing updated values

        # loop over all valid bot and rat positions
        for bx, by in open_cells:
            for rx, ry in open_cells:
                # skip case where bot and rat are in the same position - already accounted for this
                if (bx, by) == (rx, ry): continue

                # initialize minimum expected steps at infinity
                min_expected = float('inf')

                # check each direction the bot can move in
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    # compute the bot's new position
                    nbx, nby = bx + dr, by + dc
                    
                    # if the new position is not within bounds or not an open cell, skip
                    if not (0 <= nbx < D and 0 <= nby < D): continue
                    if ship[nbx][nby] != 'O': continue

                    # compute all valid moves the rat can make from its current position - randomly chooses from valid open neighbors
                    rat_moves = [(rx + rdx, ry + rdy)
                                 for rdx, rdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if 0 <= rx + rdx < D and 0 <= ry + rdy < D and ship[rx + rdx][ry + rdy] == 'O']

                    # if the rat has no valid moves, it stays where it is
                    if not rat_moves:
                        expected = T[nbx][nby][rx][ry]
                    else:
                        # if the rat can move, it picks a random possible position to move to (from the list of valid moves)
                        # calculate the expected value over the new possible rat locations
                        expected = sum(T[nbx][nby][nrx][nry] for nrx, nry in rat_moves) / len(rat_moves)

                    # add 1 for the bot's move
                    min_expected = min(min_expected, 1 + expected) # track the minimum expected steps from all possible bot moves

                # save the new best estimate in the new T array
                new_T[bx][by][rx][ry] = min_expected
                # update delta to track the largest change in this iteration
                delta = max(delta, abs(T[bx][by][rx][ry] - min_expected))

        # update the current T with the updated values
        T = new_T

        # if the max change is smaller than tol, stop - we've converged (no more value iteration necessary)
        if delta < tol:
            print("Value iteration converged.")
            break

    return T # return the T array


# determine the next best move the bot can make to minimize the expected steps to catch the rat
def best_bot_move(ship, bx, by, rx, ry, T):
    D = ship.shape[0] # ship dimension
    best_action = (bx, by) # default best move is to stay in place (in case no better moves are found)
    best_value = float('inf') # start with the worst possible value (infinity) as default

    # for all the directions the bot can move in
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nbx, nby = bx + dr, by + dc # compute its new position

        # if the new position in not within bounds or an open cell, skip it
        if not (0 <= nbx < D and 0 <= nby < D): continue
        if ship[nbx][nby] != 'O': continue

        # compute all valid rat moves from its current position
        rat_moves = [(rx + rdx, ry + rdy)
                     for rdx, rdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= rx + rdx < D and 0 <= ry + rdy < D and ship[rx + rdx][ry + rdy] == 'O']

        # if the rat can't move at all, it stays in place
        if not rat_moves:
            expected_value = T[nbx][nby][rx][ry]
        else:
            # if the rat can move, compute the average expected value (assuming the rat picks a possible position at random to move to)
            expected_value = sum(T[nbx][nby][nrx][nry] for nrx, nry in rat_moves) / len(rat_moves)

        # if the expected value for this move is better (smaller) than the best one so far
        if expected_value < best_value:
            best_value = expected_value # update with the new expected value
            best_action = (nbx, nby) # update the corresponding bot move

    return best_action


# simulate a single move of the rat from its current position
def rat_movement(ship, start):
    r, c = start # starting position of the rat
    D = ship.shape[0] # ship dimension
    neighbors = [] # list for storing valid neighbor positions (for the rat to move to)

    # Possible movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # check each neighbor cell in all 4 directions
    for dr, dc in directions:
        nr, nc = r + dr, c + dc # compute the position of the neighbor

        # if the neighbor is within bounds and an open cell
        if 0 <= nr < D and 0 <= nc < D and ship[nr, nc] == 'O':
            neighbors.append((nr, nc)) # add to the list of neighbors

    # if the rat has no valid neighbors (no positions to move to)
    if not neighbors:
        return start # the rat can't move - stays in place

    # if the rat has valid neighbors, choose a random neighbor to move to
    return random.choice(neighbors)


# simulates the bot searching for the rat
def rat_search(ship, bot_start, rat_start, T, max_steps=1000):
    D = ship.shape[0] # ship dimension

    # track the movement history of the bot and the rat
    bot_history = [bot_start]
    rat_history = [rat_start]

    # run the simulation until the max number of steps is reached (each turn includes 1 bot move and 1 rat move)
    for step in range(max_steps):
        # get the current (most recent) position of the bot and the rat
        bot_x, bot_y = bot_history[-1]
        rat_x, rat_y = rat_history[-1]

        # calculate the bot's best move and add it to its movement history
            # best move is calculated based on its current position, the rat's current position, and the T array of (bot, rat) configurations
        next_bot_move = best_bot_move(ship, bot_x, bot_y, rat_x, rat_y, T)
        bot_history.append(next_bot_move)

        # check if the bot catches the rat (both are in the same position) after moving
        if next_bot_move == (rat_x, rat_y):
            print(f"Bot caught the rat in {step + 1} steps!")
            return step + 1, bot_history, rat_history

        # calculate the rat's best move and add it to its movement history
        next_rat_move = rat_movement(ship, (rat_x, rat_y))
        rat_history.append(next_rat_move)

        # check if rat walks into the bot
        if next_rat_move == next_bot_move:
            print(f"Rat walked into bot in {step + 1} steps!")
            return step + 1, bot_history, rat_history

    print("Max steps reached. Rat not caught.")
    return max_steps, bot_history, rat_history


# === Testing values of T (for write-up question) ===
def identify_easy_T(ship, T):
    D = ship.shape[0] # ship dimension

    # list of open cells
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 'O']
    easy_cases = [] # list for storing easy configurations of T

    # for each possible bot and rat location in the list of open cells
    for bx, by in open_cells:
        for rx, ry in open_cells:

            # if the bot and rat are in the same position - expected number of steps is 0
            if (bx, by) == (rx, ry):
                easy_cases.append((bx, by, rx, ry, 0)) # add to list of easy cases
            else:
                # check if the rat is trapped by computing positions of its neighbors and checking if they are within bounds and open
                rat_neighbors = [(rx + dx, ry + dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                                 if 0 <= rx + dx < D and 0 <= ry + dy < D and ship[rx+dx][ry+dy] == 'O']
                
                # if the rat has no valid neighbors, the rat is stuck
                if not rat_neighbors:
                    # use A* search to get shortest path
                    dist = a_star_search(ship, (bx, by), (rx, ry))

                    # if there is a valid distance between the bot and the rat, add it to the list of easy cases (since it is the shortest path)
                    if dist is not None:
                        easy_cases.append((bx, by, rx, ry, dist))
    return easy_cases


# === Main function ===
def main():
    # generate the ship, dimension, and list of possible open cells
    ship = generate_ship(20, 0.5)
    D = ship.shape[0]
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 'O']
    
    # pick an open cell at random for the bot and the rat
    bot_pos = random.choice(open_cells)
    rat_pos = random.choice(open_cells)

    print("Computing value function...")
    T = compute_value_function(ship) # compute T
    print("Value function ready!")

    '''
    # Test easy-to-compute T values
    easy_cases = identify_easy_T(ship, T)
    print("Easy T values (where we know expected steps):")
    for bx, by, rx, ry, expected in easy_cases[:10]:  # print only first 10
        print(f"T[{bx},{by},{rx},{ry}] = {T[bx][by][rx][ry]:.2f} (Expected: {expected})")
    '''

    # get number of steps (expected and actual number)
    steps, bot_path, rat_path = rat_search(ship, bot_pos, rat_pos, T)
    print(f"Rat caught in {steps} steps!")


main()
