import numpy as np
import matplotlib.pyplot as plt
from Cell import Cell
from Grid import Grid
import time

seed = int(np.random.uniform(0,10000))
print("Random Seed: " + str(seed))

# maxCells = int(input("How many paths will be added to the maze? "))

Dims = {"x": 10, "y": 5}
# Dims["x"]=2*Dims["x"]+1
# Dims["y"]=2*Dims["y"]+1

b = Grid()
start_time = time.time()
b.genMazeArea(seed,Dims)
end_time = time.time()

print("Run time = " + str(end_time - start_time))
b.showMaze()


# for cell in b.maze:
# 	print(b.maze[cell].getArray().shape)
