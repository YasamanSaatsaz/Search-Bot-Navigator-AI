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


# === Search algorithm ===

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

def compute_value_function(ship, max_iters=1000, tol=1e-2):
    D = ship.shape[0]
    T = np.full((D, D, D, D), 1000.0)
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r][c] == 'O']

    for bx, by in open_cells:
        T[bx][by][bx][by] = 0.0  # If bot and rat are at same location

    for _ in range(max_iters):
        delta = 0.0
        new_T = T.copy()

        for bx, by in open_cells:
            for rx, ry in open_cells:
                if (bx, by) == (rx, ry): continue

                min_expected = float('inf')

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nbx, nby = bx + dr, by + dc
                    if not (0 <= nbx < D and 0 <= nby < D): continue
                    if ship[nbx][nby] != 'O': continue

                    rat_moves = [(rx + rdx, ry + rdy)
                                 for rdx, rdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if 0 <= rx + rdx < D and 0 <= ry + rdy < D and ship[rx + rdx][ry + rdy] == 'O']

                    if not rat_moves:
                        expected = T[nbx][nby][rx][ry]
                    else:
                        expected = sum(T[nbx][nby][nrx][nry] for nrx, nry in rat_moves) / len(rat_moves)

                    min_expected = min(min_expected, 1 + expected)

                new_T[bx][by][rx][ry] = min_expected
                delta = max(delta, abs(T[bx][by][rx][ry] - min_expected))

        T = new_T
        if delta < tol:
            print("Value iteration converged.")
            break

    return T

def best_bot_move(ship, bx, by, rx, ry, T):
    D = ship.shape[0]
    best_action = (bx, by)
    best_value = float('inf')

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nbx, nby = bx + dr, by + dc
        if not (0 <= nbx < D and 0 <= nby < D): continue
        if ship[nbx][nby] != 'O': continue

        # All valid rat moves
        rat_moves = [(rx + rdx, ry + rdy)
                     for rdx, rdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= rx + rdx < D and 0 <= ry + rdy < D and ship[rx + rdx][ry + rdy] == 'O']

        if not rat_moves:
            expected_value = T[nbx][nby][rx][ry]
        else:
            expected_value = sum(T[nbx][nby][nrx][nry] for nrx, nry in rat_moves) / len(rat_moves)

        if expected_value < best_value:
            best_value = expected_value
            best_action = (nbx, nby)

    return best_action

def rat_movement(ship, start):
    r, c = start
    D = ship.shape[0]  # assuming square grid
    neighbors = []

    # Possible movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        # Check bounds and if the neighbor cell is open
        if 0 <= nr < D and 0 <= nc < D and ship[nr, nc] == 'O':
            neighbors.append((nr, nc))

    if not neighbors:
        # No valid moves (e.g., surrounded by walls)
        return start

    # Choose a random valid neighbor
    return random.choice(neighbors)


def rat_search(ship, bot_start, rat_start, T, max_steps=1000):
    D = ship.shape[0]
    bot_history = [bot_start]
    rat_history = [rat_start]

    for step in range(max_steps):
        bot_x, bot_y = bot_history[-1]
        rat_x, rat_y = rat_history[-1]

        # Bot moves
        next_bot_move = best_bot_move(ship, bot_x, bot_y, rat_x, rat_y, T)
        bot_history.append(next_bot_move)

        # Check if bot catches rat
        if next_bot_move == (rat_x, rat_y):
            print(f"Bot caught the rat in {step + 1} steps!")
            return step + 1, bot_history, rat_history

        # Rat moves
        next_rat_move = rat_movement(ship, (rat_x, rat_y))
        rat_history.append(next_rat_move)

        # Check if rat walks into bot
        if next_rat_move == next_bot_move:
            print(f"Rat walked into bot in {step + 1} steps!")
            return step + 1, bot_history, rat_history

    print("Max steps reached. Rat not caught.")
    return max_steps, bot_history, rat_history



# === Main function ===
def main():
    ship = generate_ship(20, 0.5)
    D = ship.shape[0]
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 'O']
    bot_pos = random.choice(open_cells)
    rat_pos = random.choice(open_cells)

    print("Computing value function...")
    T = compute_value_function(ship)
    print("Value function ready!")

    steps, bot_path, rat_path = rat_search(ship, bot_pos, rat_pos, T)
    print(f"Rat caught in {steps} steps!")

# Run it
main()
