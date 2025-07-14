# Project 3 - Intro to AI (Summer 2025)
# Contributors: Yasaman Saatsaz, Gilbert Teng


# import statements
import numpy as np
import heapq # priority queue for A* search
import random
# for machine learning
import torch
import torch.nn as nn
import torch.optim as optim


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


# === Generating T array (and helper functions) ===

# compute value function through iterations for each (bot, rat) configuration
# in the 4D array T to calculate expected number of moves
def compute_value_function(ship, max_iters=1000, tol=0.3):
    D = ship.shape[0] # ship dimension

    '''
    # creates 4D array T - each element represents a configuration of (bot, rat)
        # T[bx][by][rx][ry] - represents the expected number of moves to catch the rat starting at (bx, by) and (rx, ry)
    '''
    T = np.full((D, D, D, D), 500.0)
    # list of all open cells - cells where the rat could be in or the bot can go to
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r][c] == 'O']

    # for every possible bot location
    for bx, by in open_cells:
        T[bx][by][bx][by] = 0.0 # if bot and rat at same location, expected steps = 0

    # start value iteratiom
    for _ in range(max_iters):
        delta = 0.0 # maximum change in T-values
        new_T = T.copy() # copy for storing updated values

        # loop over all valid bot and rat positions
        for bx, by in open_cells:
            for rx, ry in open_cells:
                # skip case where bot and rat are in the same position - already accounted for this
                if (bx, by) == (rx, ry):
                    continue

                # initialize minimum expected steps at infinity
                min_expected = float('inf')

                # check each direction the bot can move in
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nbx, nby = bx + dr, by + dc # compute the bot's new position in each direction

                    # if the new position is not within bounds or not an open cell, skip it
                    if not (0 <= nbx < D and 0 <= nby < D):
                        continue
                    if ship[nbx][nby] != 'O':
                        continue

                    # compute all valid moves the rat can make from its current position - only considers valid open neighbors
                    rat_moves = [(rx + rdx, ry + rdy)
                                 for rdx, rdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if 0 <= rx + rdx < D and 0 <= ry + rdy < D and ship[rx + rdx][ry + rdy] == 'O']

                    # if the rat has no valid moves, it stays where it is
                    if not rat_moves:
                        expected = T[nbx][nby][rx][ry]
                    else:
                        # if the rat can move, it picks a random possible position to move to (from list of valid moves)
                        # calculate the expected value over the new possible rat locations
                        expected = sum(T[nbx][nby][nrx][nry] for nrx, nry in rat_moves) / len(rat_moves)

                    # add 1 for the bot's move
                    min_expected = min(min_expected, 1 + expected) # track the min expected steps from all possble bot moves

                new_T[bx][by][rx][ry] = min_expected # save the new best estimate in the new T array

        # Convergence check
        total_change = 0.0
        count = 0
        for bx in range(D):
            for by in range(D):
                for rx in range(D):
                    for ry in range(D):
                        # calculate the change in values for all values in T (old T - new T)
                        change = abs(T[bx][by][rx][ry] - new_T[bx][by][rx][ry])
                        total_change += change
                        count += 1

        # calculate the average change
        average_change = total_change / count
        T = new_T  # always update T

        # if the average change is less than given change
        if average_change < tol:
            print(f"Converged after {_ + 1} iterations with average change {average_change:.4f}")
            break

    T[T > 400] = np.inf
    print(f"Final T stats: min={np.nanmin(T):.2f}, max={np.nanmax(T):.2f}")
    return T


# determine the next best move the bot can make to minimize the expected steps to catch the rat
def best_bot_move(ship, bx, by, rx, ry, T):
    D = ship.shape[0] # ship dimension
    best_action = (bx, by) # default best move is to stay in place (in case no better moves are found)
    best_value = float('inf') # start with worst possible value (infinity) as default

    # for all directions the bot can move in
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nbx, nby = bx + dr, by + dc # compute its new position

        # if new position is not within bounds or an open cell, skip it
        if not (0 <= nbx < D and 0 <= nby < D): continue
        if ship[nbx][nby] != 'O': continue

        # compute all valid rat moves from its current position (only consider valid open neighbors)
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
            best_value = expected_value # update with new expected value
            best_action = (nbx, nby) # update the corresponding bot move

    return best_action


# simulate a single move of the rat from its current position
def rat_movement(ship, start):
    r, c = start # starting position of rat
    D = ship.shape[0]  # ship dimension
    neighbors = [] # list for storing valid neighbor positions (for the rat to move to)

    # Possible movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # check each neighbor cell in all 4 directions
    for dr, dc in directions:
        nr, nc = r + dr, c + dc # compute the position of the rat's neighbor
        # Check bounds and if the neighbor cell is open
        if 0 <= nr < D and 0 <= nc < D and ship[nr, nc] == 'O':
            neighbors.append((nr, nc)) # add to list of neighbors

    # if the rat has no valid neighbors (no positions to move to)
    if not neighbors:
        return start # the rat can't move (stays in place)

    # if the rat has valid neighbors, choose a random neighbor to move to
    return random.choice(neighbors)


# simulate the bot searching for the rat
def rat_search(ship, bot_start, rat_start, T, max_steps=1000):
    D = ship.shape[0] # ship dimension
    bot_history = [bot_start] # movement history - bot
    rat_history = [rat_start] # movement history - rat

    # run the simulation until the max number of steps is reached (each turn includes 1 bot move and 1 rat move)
    for step in range(max_steps):
        # get the current (most recent) position of the bot and the rat
        bot_x, bot_y = bot_history[-1]
        rat_x, rat_y = rat_history[-1]

        # calculater the bot's best move and add it to its movement history
        '''
        best move is calculated based on its current position, the rat's current position, and the T array of (bot, rat) configurations
        '''
        next_bot_move = best_bot_move(ship, bot_x, bot_y, rat_x, rat_y, T)
        bot_history.append(next_bot_move)

        # check if the bot catches the rat (both in same position) after moving
        if next_bot_move == (rat_x, rat_y):
            print(f"Bot caught the rat in {step + 1} steps!")
            return step + 1, bot_history, rat_history

        # calculate the rat's best move and add it to its movement history
        next_rat_move = rat_movement(ship, (rat_x, rat_y))
        rat_history.append(next_rat_move)

        # check if rat walks into bot
        if next_rat_move == next_bot_move:
            print(f"Rat walked into bot in {step + 1} steps!")
            return step + 1, bot_history, rat_history

    print("Max steps reached. Rat not caught.")
    return max_steps, bot_history, rat_history


# === Machine Learning (Part 2) ===

# model class
class RatNet(nn.Module):
    def __init__(self): # "initializer constructor" for the class
        super().__init__() # extend the constructor

        # create the 3 layers (input_size, output_size) of the neural network
        self.fc1 = nn.Linear(4, 64) # input layer
        self.fc2 = nn.Linear(64, 64) # hidden layer
        self.fc3 = nn.Linear(64, 1) # output layer

    # computes the output of the network given input x
    def forward(self, x):
        # apply first layer to input; ReLU activation function used for non-linearity
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) # apply second layer and activation function
        x = self.fc3(x) # apply third layer to produce output (no activation function needed)
        return x # return final output tensor
    

# creating a dataset from T
def build_dataset_from_T(T, D):
    data = [] # for storing dataset examples

    # loop over T for each (bot, rat) configuration
    for bx in range(D):
        for by in range(D):
            for rx in range(D):
                for ry in range(D):
                    # get value from current configuration
                    steps = T[bx][by][rx][ry]
                    if np.isfinite(steps): # check if the value is a finite number (only build dataset considering finite values)
                        data.append(((bx, by, rx, ry), steps)) # add the configuration and its value to the dataset
    return data


# get a random batch of inputs and outputs (for training)
def get_batch(data, batch_size):
    # get random indices from among dataset indices
    batch_indices = random.sample(range(len(data)), k=batch_size)

    # get the input values of the random indices
        # convert each (bot, rat) tuple to a list
        # convert the list of inputs into a tensor
    x_batch = torch.tensor([list(data[i][0]) for i in batch_indices], dtype=torch.float32)
    
    # do the same with the corresponding output values
    y_batch = torch.tensor([data[i][1] for i in batch_indices], dtype=torch.float32).unsqueeze(1)
    return x_batch, y_batch # return the input and output tensors for the random batch


# training loop
def train_model(model, data, epochs=10, batch_size=1024, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr) # create an optimizer for the model with a learning rate (alpha)
    loss_fn = nn.MSELoss() # loss function - mean squared error

    # loop over epochs
    for epoch in range(epochs):
        total_loss = 0 # initialize loss
        num_batches = len(data) // batch_size # calculate how many batches fit into the dataset

        # loop over every batch in the epoch
        for _ in range(num_batches):
            # get random batch of inputs and outputs
            x_batch, y_batch = get_batch(data, batch_size)

            # clears old gradients to prepare for the new batch
            optimizer.zero_grad() 

            # run the model on the input batch to get predictions
            predictions = model(x_batch)

            # compute the mean squared error (loss) between predictions and actual outputs
            loss = loss_fn(predictions, y_batch)
            loss.backward() # perform gradient descent
            optimizer.step() # update model's parameters based on gradients
            total_loss += loss.item() # add this batch's loss to the total loss

        avg_loss = total_loss / num_batches # average loss
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}") # print avg loss for each epoch


# evaluates the trained model on test data
def evaluate_model(model, test_data):
    # convert all inputs in the testing data into a tensor
    x_test = torch.tensor([list(pos) for pos, _ in test_data], dtype=torch.float32)
    
    # convert all outputs in the testing data into a tensor, reshape the tensor for a single output
    y_test = torch.tensor([label for _, label in test_data], dtype=torch.float32).unsqueeze(1)
    model.eval() # set to evaluation mode

    # calculations should be done without keeping track of gradients
    with torch.no_grad():
        # run the model on all test inputs --> get predictions
        predictions = model(x_test)
        mse = nn.MSELoss()(predictions, y_test).item() # compute loss on test predictions
    print(f"Test MSE: {mse:.4f}") # print loss
    return predictions # return the predicted values for the testing data


'''
def train_model_from_T(T):
    D = T.shape[0]

    # Step 1: Build dataset from T, skipping infinite values
    X = []
    y = []

    for bot_r in range(D):
        for bot_c in range(D):
            for rat_r in range(D):
                for rat_c in range(D):
                    expected_steps = T[bot_r][bot_c][rat_r][rat_c]
                    if np.isinf(expected_steps):  # Skip invalid entries
                        continue
                    X.append([bot_r, bot_c, rat_r, rat_c])
                    y.append(expected_steps)

    print(f"Total valid training samples: {len(X)}")
    print(f"Total skipped (invalid) entries: {D**4 - len(X)}")

    # Step 2: Convert to tensors and normalize inputs
    X = torch.tensor(X, dtype=torch.float32) / D  # normalize to [0, 1]
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Step 3: Define a simple feedforward neural network
    class TNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.model(x)

    model = TNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 4: Train model
    epochs = 200  # You can increase if loss is still going down
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        if torch.isnan(loss):
            print("⚠️ Loss is NaN! Check your data!")
            break
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Step 5: Test prediction on a random config
    with torch.no_grad():
        test_index = random.randint(0, len(X) - 1)
        test_input_raw = X[test_index] * D  # undo normalization for display
        test_input = X[test_index].unsqueeze(0)  # model expects batch dim
        true_value = y[test_index].item()
        predicted = model(test_input).item()
        print("\n✅ Sample Test:")
        print(f"Input config (bot_r, bot_c, rat_r, rat_c): {test_input_raw.numpy().astype(int)}")
        print(f"True expected steps: {true_value:.2f}")
        print(f"Model predicted steps: {predicted:.2f}")

    return model
    '''


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

    print("Building dataset from T...")
    data = build_dataset_from_T(T, D)
    print(f"Total data points: {len(data)}")

    # Shuffle and create a fixed sample
    sample_size = 10000
    train_sample = random.sample(data, sample_size)
    test_data = [d for d in data if d not in train_sample]

    model = RatNet() # create the model

    # train the model
    epochs = 10000
    alpha = 0.002
    train_model(model, train_sample, 10000, sample_size, alpha)

    # evaluate on the test set
    evaluate_model(model, test_data)

    '''
    # Train the model on T
    print("Training ML model to predict T...")
    model = train_model_from_T(T)
    '''


    '''
    steps, bot_path, rat_path = rat_search(ship, bot_pos, rat_pos, T)
    print(f"Rat caught in {steps} steps!")
    '''


# Run it
main()
