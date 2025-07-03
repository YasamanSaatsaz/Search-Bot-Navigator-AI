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


def main():
    print("this code is running")
    
main()