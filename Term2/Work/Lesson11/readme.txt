Here is one possible implementation for Hybrid A* using the "distance to goal" heuristic function. 
In this implementation, we have added an f value to the maze_s struct, which is set in the expand function. 
Additionally, we've added two new functions: heuristic and compare_maze_s. 
The compare_maze_s function is used for comparison of maze_s objects when sorting the opened stack.

To get an even lower number of expansions, try reducing NUM_THETA_CELLS in hybrid_breadth_first.h to reduce the 
total number of cells that can be searched in the closed array. 
Be careful though! Making NUM_THETA_CELLS too small might result in the algorithm being unable to find a path 
through the maze.

Another possibility for improvement is to use the regular A* algorithm to assign a cost value to each grid cell. 
This grid of costs can then be used as the heuristic function, which will lead to an extremely efficient search. 
If you are looking for additional practice, give this a try!