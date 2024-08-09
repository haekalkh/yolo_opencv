import numpy as np
from scipy.optimize import minimize
from pymycobot.mycobot import MyCobot
import matplotlib.pyplot as plt


# Fungsi forward kinematics 3D
def forward_kinematics(joint_angles, L):
    x, y, z = 0, 0, 0
    for i in range(len(joint_angles)):
        x += L[i] * np.cos(joint_angles[i])
        y += L[i] * np.sin(joint_angles[i])
        z += 0  # Jika ada perubahan z, ini perlu dimodifikasi
    return np.array([x, y, z])

# Fungsi untuk menghitung kesalahan (error) antara posisi target dan posisi yang dicapai
def objective_function(joint_angles, target, L):
    position = forward_kinematics(joint_angles, L)
    return np.linalg.norm(position - target)

# Parameter robot dalam sentimeter
L = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # Panjang link dalam cm
target = [30.0, 30.0, 20.0]  # Posisi target dalam cm

# Inisialisasi sudut sendi
initial_joint_angles = np.zeros(6)

# Optimisasi menggunakan scipy.optimize.minimize
result = minimize(objective_function, initial_joint_angles, args=(target, L), method='BFGS')

# Hasil
joint_angles_solution = result.x

print(f"Sudut sendi yang dihitung (rad): {joint_angles_solution}")

# Mengirim sudut sendi ke robot
import mc  # Asumsi bahwa mc adalah modul yang mengelola komunikasi robot

# Mengirim sudut sendi yang dihitung
mc.send_angles(joint_angles_solution)

# Plotting (untuk visualisasi 2D)
def plot_robot(joint_angles, L):
    x, y, z = [0], [0], [0]
    current_x, current_y, current_z = 0, 0, 0
    
    for i in range(6):
        current_x += L[i] * np.cos(joint_angles[i])
        current_y += L[i] * np.sin(joint_angles[i])
        current_z += 0  # Jika ada perubahan z, ini perlu dimodifikasi
        x.append(current_x)
        y.append(current_y)
        z.append(current_z)
    
    # Visualisasi dalam 2D
    plt.plot(x, y, 'ro-')
    plt.plot(target[0], target[1], 'bx')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Robot 6 DOF dengan Target")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.grid(True)
    plt.show()

plot_robot(joint_angles_solution, L)
