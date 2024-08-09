import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fungsi forward kinematics (sederhana untuk contoh)
def forward_kinematics(joint_angles, L):
    # Menghitung posisi akhir robot berdasarkan sudut sendi dan panjang link
    x = sum(L[i] * np.cos(joint_angles[i]) for i in range(6))
    y = sum(L[i] * np.sin(joint_angles[i]) for i in range(6))
    z = 0  # Asumsi sederhana
    return np.array([x, y, z])

# Fungsi untuk menghitung kesalahan (error) antara posisi target dan posisi yang dicapai
def objective_function(joint_angles, target, L):
    position = forward_kinematics(joint_angles, L)
    return np.linalg.norm(position - target)

# Parameter robot dalam sentimeter
L = [17.3, 13.5, 12.0, 9.5, 8.8, 6.5]  # Panjang link dalam cm
target = [30.0, 30.0, 0.0]  # Posisi target dalam cm

# Inisialisasi sudut sendi
initial_joint_angles = np.zeros(6)

# Optimisasi menggunakan scipy.optimize.minimize
result = minimize(objective_function, initial_joint_angles, args=(target, L), method='BFGS')

# Hasil
joint_angles_solution = result.x

print(f"Sudut sendi yang dihitung (rad): {joint_angles_solution}")

# Plotting (untuk visualisasi 2D)
def plot_robot(joint_angles, L):
    x = [0]
    y = [0]
    current_x, current_y = 0, 0
    
    for i in range(6):
        current_x += L[i] * np.cos(joint_angles[i])
        current_y += L[i] * np.sin(joint_angles[i])
        x.append(current_x)
        y.append(current_y)
    
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
