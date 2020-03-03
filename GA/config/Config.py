import os
import math as ma
import numpy as np


dim = 2
            

def load_map():
        """function that loads the map which was created from a .txt file
    
        Returns
        
        matrix of map as np.array
        coordinates of obstacles as a list of coordinate arrays
        coordinates of free spaces as a list of coordinate arrays
        
        """
        
        map_filename = "map5.txt"
        txt_files = list(np.sort([file for file in os.listdir(os.getcwd()) if file.endswith(".txt")]))
        if len(txt_files) > 0:
            for txt_file in txt_files:
                map_filename = os.path.join(os.getcwd(), txt_file)

            if map_filename is None:
                print("no map found")
                return None
            else:
                with open(map_filename, "r") as file:
                    map_data = file.readlines()
                map_matrix = np.loadtxt(map_data)
                map_plot = []
                path_points=[]
                for y in range(len(map_matrix)):
                    for x in range(len(map_matrix)):
                        if map_matrix[y, x] == 1:
                            #create a list of coordinates of obstacles
                            map_plot.append([x, y])
                        if map_matrix[y, x] == 0:
                            #create  a list of the coordinates of free spaces
                            path_points.append([x, y])   
                return map_matrix,map_plot,path_points
    

map_matrix, map_plot, path_points = load_map()  # this is the list of coordinates that can be travelled through
npts = len(path_points)
#Number of paths for the starting population which can be varied
pop_max = 50
# Rate of mutation which can be varied
mutation_rate = 0.01
#index of the start point ie the first element in path points list
start_index = int(0)
#index of the end point ie the last element of the path points list
end_index = npts - 1
#this is used in the main() for printing generation number
generations = 1
#in main()
prev_best_fitness = 0
#no of points between the start and goal points
nobs = 149
#not sure just left as is. I think it is to make sure the number stays as an integer 
#because when it's not an integer all hell breaks lose
nbits = ma.log10(npts) / ma.log10(2)
#length of each path which is nobs plus the start and end points
chr_len = int(((nobs+2)*nbits)/nbits)

#specify tournament size
tournament_size = 25

#in main()
stop_criteria = 0
#used in iteration loop in main()
stop_generation = False
img_iter_no = 1
#range of axis for graph plot which depends on the size of our map 
plt_tolerance = -1
plt_ax_x_min = -1
plt_ax_x_max = map_matrix.shape[0] + 1
plt_ax_y_min = -1
plt_ax_y_max = map_matrix.shape[0] + 1



        
def define_links():
    """
    This function defines the links b/w path points
    
    Returns
    -------
    [numpy.ndarray]
        [Every path point has a number of allowed connection with other path 
        points. Those allowed connections are defined below. During calculation
        of fitness of population if two consecutive path points are connected
        then the fitness of that chromosome increases]
        the allowed connections are the eight squares that surround a single square
        (including diagonals)
    """

    x = path_points
    no_of_points = len(x)
    link = -1 * np.ones((no_of_points, 8))
    
    neighbours = [[-1, -1], [0, -1],[1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
    
    for i in range(len(x)):
    # for each path point, create a list of all other path points it can move directly into
    #if a neighburing points is an obstacle, give a value of -1 and cannot be accessed
    
        
        for j in range(len(neighbours)):
            neighbour = [x + y for x, y in zip(neighbours[j], x[i])]
           
            if neighbour in x:
                link[i][j] = int(x.index(neighbour))
                
        link[i][0] = int(i)     
                
    return link