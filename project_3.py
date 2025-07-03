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

def main():
    print("this code is running")

main()