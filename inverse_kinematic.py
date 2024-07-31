import numpy as np

def axis_angle_rot_matrix(k, q):
    """
    Creates a 3x3 rotation matrix in 3D space from an axis and an angle.
    
    Input
    :param k: A 3 element array containing the unit axis to rotate around (kx, ky, kz)
    :param q: The angle (in radians) to rotate by
    
    Output
    :return: A 3x3 element matrix containing the rotation matrix
    """
    c_theta = np.cos(q)
    s_theta = np.sin(q)
    v_theta = 1 - np.cos(q)
    kx, ky, kz = k
    
    r00 = kx * kx * v_theta + c_theta
    r01 = kx * ky * v_theta - kz * s_theta
    r02 = kx * kz * v_theta + ky * s_theta
    
    r10 = kx * ky * v_theta + kz * s_theta
    r11 = ky * ky * v_theta + c_theta
    r12 = ky * kz * v_theta - kx * s_theta
    
    r20 = kx * kz * v_theta - ky * s_theta
    r21 = ky * kz * v_theta + kx * s_theta
    r22 = kz * kz * v_theta + c_theta
    
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                        
    return rot_matrix

def hr_matrix(k, t, q):
    '''
    Create the Homogenous Representation matrix that transforms a point from Frame B to Frame A.
    
    Input
    :param k: A 3 element array containing the unit axis to rotate around (kx, ky, kz) 
    :param t: The translation from the current frame (e.g. Frame A) to the next frame (e.g. Frame B)
    :param q: The rotation angle (i.e. joint angle)
    
    Output
    :return: A 4x4 Homogenous representation matrix
    '''
    rot_matrix_A_B = axis_angle_rot_matrix(k, q)
    translation_vec_A_B = t

    t_x, t_y, t_z = translation_vec_A_B
    translation_vec_A_B = np.array([[t_x],
                                    [t_y],
                                    [t_z]])
                                 
    homgen_mat = np.concatenate((rot_matrix_A_B, translation_vec_A_B), axis=1)
    extra_row_homgen = np.array([[0, 0, 0, 1]])
    homgen_mat = np.concatenate((homgen_mat, extra_row_homgen), axis=0)
        
    return homgen_mat

class RoboticArm:
    def __init__(self, k_arm, t_arm):
        '''
        Creates a robotic arm class for computing position and velocity.

        Input
        :param k_arm: A 2D array that lists the different axes of rotation (rows) for each joint.
        :param t_arm: A 2D array that lists the translations from the previous joint to the current joint
                      The first translation is from the global (base) frame to joint 1 (which is often equal to the global frame)
                      The second translation is from joint 1 to joint 2, etc.
        '''
        self.k = np.array(k_arm)
        self.t = np.array(t_arm)
        assert k_arm.shape == t_arm.shape, 'Warning! Improper definition of rotation axes and translations'
        self.N_joints = k_arm.shape[0]

    def position(self, Q, index=-1, p_i=[0, 0, 0]):
        '''
        Compute the position in the global (base) frame of a point given in a joint frame
        (default values will assume the input position vector is in the frame of the last joint)
        
        Input
        :param p_i: A 3 element vector containing a position in the frame of the index joint
        :param index: The index of the joint frame being converted from (first joint is 0, the last joint is N_joints - 1)

        Output
        :return: A 3 element vector containing the new position with respect to the global (base) frame
        '''
        p_i_x, p_i_y, p_i_z = p_i
        this_joint_position = np.array([[p_i_x],
                                        [p_i_y],
                                        [p_i_z],
                                        [1]])

        if index == -1:
            index = self.N_joints - 1
        
        orig_joint_index = index
        running_multiplication = None
        
        while index >= 0:
            if index == orig_joint_index:
                running_multiplication = hr_matrix(self.k[index], self.t[index], Q[index]) @ this_joint_position
            else: 
                running_multiplication = hr_matrix(self.k[index], self.t[index], Q[index]) @ running_multiplication
        
            index -= 1
        
        x = running_multiplication[0][0]
        y = running_multiplication[1][0]
        z = running_multiplication[2][0]
        
        position_global_frame = np.array([x, y, z])
        
        return position_global_frame

    def pseudo_inverse(self, theta_start, p_eff_N, goal_position, max_steps=np.inf):
        '''
        Performs the inverse kinematics using the pseudoinverse of the Jacobian

        :param theta_start: An N element array containing the current joint angles in radians (e.g. np.array([np.pi/8, np.pi/4, np.pi/6]))
        :param p_eff_N: A 3 element vector containing translation from the last joint to the end effector in the last joints frame of reference
        :param goal_position: A 3 element vector containing the desired end position for the end effector in the global (base) frame
        :param max_steps: (Optional) Maximum number of iterations to compute 

        Output
        :return: An N element vector containing the joint angles that result in the end effector reaching xend (i.e. the goal)
        '''
        v_step_size = 0.05
        theta_max_step = 0.2
        Q_j = theta_start
        p_end = np.array(goal_position)
        p_j = self.position(Q_j, p_i=p_eff_N)
        delta_p = p_end - p_j
        j = 0
        
        while np.linalg.norm(delta_p) > 0.01 and j < max_steps:
            #print(f'j{j}: Q[{Q_j}] , P[{p_j}]')
            
            v_p = delta_p * v_step_size / np.linalg.norm(delta_p)
            J_j = self.jacobian(Q_j, p_eff_N)
            J_invj = np.linalg.pinv(J_j)
            v_Q = J_invj @ v_p
            
            Q_j = Q_j + np.clip(v_Q, -theta_max_step, theta_max_step)
 
            p_j = self.position(Q_j, p_i=p_eff_N)
             
            j += 1
            delta_p = p_end - p_j
         
        return Q_j

    def jacobian(self, Q, p_eff_N=[0, 0, 0]):
        '''
        Computes the Jacobian (just the position, not the orientation)

        :param Q: An N element array containing the current joint angles in radians
        :param p_eff_N: A 3 element array containing the translation from the last joint to the end effector

        Output
        :return: A 3xN element matrix containing the Jacobian 
        '''
        p_eff = self.position(Q, -1, p_eff_N)
         
        first_iter = True
        jacobian_matrix = None
         
        for i in range(self.N_joints):
            p_eff_minus_this_p = p_eff - self.position(Q, index=i)
            k = self.k[i]
             
            this_jacobian = np.cross(k, p_eff_minus_this_p)
            this_jacobian = np.array([[this_jacobian[0]],
                                      [this_jacobian[1]],
                                      [this_jacobian[2]]])
              
            if first_iter:
                jacobian_matrix = this_jacobian
                first_iter = False
            else:
                jacobian_matrix = np.concatenate((jacobian_matrix, this_jacobian), axis=1)
                 
        return jacobian_matrix
