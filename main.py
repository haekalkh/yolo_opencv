import numpy as np
from object_detection import detect_object  # Import the function from the modified object_detection.py
from inverse_kinematic import RoboticArm

def main():
    '''
    Given a six degree of freedom robotic arm and a desired end position of the end effector,
    calculate and print the six joint angles.
    '''
    # Define the axes of rotation and translations for a 6 DOF arm
    k = np.array([
        [0, 0, 1],  # Joint 1
        [0, 1, 0],  # Joint 2
        [0, 1, 0],  # Joint 3
        [1, 0, 0],  # Joint 4
        [0, 1, 0],  # Joint 5
        [1, 0, 0]   # Joint 6
    ])
     
    t = np.array([
        [0, 0, 0],    # Base to Joint 1
        [0, 0, 10],   # Joint 1 to Joint 2
        [10, 0, 0],   # Joint 2 to Joint 3
        [10, 0, 0],   # Joint 3 to Joint 4
        [0, 0, 10],   # Joint 4 to Joint 5
        [0, 10, 0]    # Joint 5 to Joint 6
    ])
     
    # Link lengths (assuming link lengths are given in the same order as the joints)
    link_lengths = [10, 10, 10, 10, 10, 10]  # Adjust according to your robot

    # Create an object of the RoboticArm class
    robotic_arm = RoboticArm(k, t, link_lengths)
     
    # Starting joint angles in radians (joint 1 to joint 6)
    q_0 = np.zeros(6)

    # Get the detected object position from the object detection function
    endeffector_goal_position = detect_object()
    if endeffector_goal_position is None:
        print("No object detected. Exiting.")
        return

    # Position of end effector in joint 6 frame (if required, adjust accordingly)
    p_eff_6 = [0, 0, 10]
     
    # Return joint angles that result in the end effector reaching endeffector_goal_position
    final_q = robotic_arm.pseudo_inverse(q_0, p_eff_N=p_eff_6, goal_position=endeffector_goal_position, max_steps=500)
     
    # Print Final Joint Angles in Degrees   
    print('\nFinal Joint Angles in Degrees:')
    for i, angle in enumerate(final_q):
        print(f'Joint {i + 1}: {np.degrees(angle):.2f}')

if __name__ == '__main__':
    main()
