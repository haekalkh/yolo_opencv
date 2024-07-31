# inverse_kinematic.py

import numpy as np

def get_goal_position():
    """
    Prompts the user to enter the goal position coordinates and returns them as a numpy array.
    
    Output:
    :return: A numpy array containing the x, y, z coordinates of the goal position.
    """
    x = float(input("Enter the x coordinate of the end effector goal position: "))
    y = float(input("Enter the y coordinate of the end effector goal position: "))
    z = float(input("Enter the z coordinate of the end effector goal position: "))
    return np.array([x, y, z])
