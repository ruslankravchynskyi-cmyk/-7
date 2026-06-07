import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Задані константи
l = 1.0       # довжина стрижня, м
g = 9.81      # прискорення вільного падіння, м/с^2
theta0_deg = 30.0
theta0 = np.radians(theta0_deg) # переведення початкового кута в радіани

# 2. Визначення системи диференціальних рівнянь для точної моделі
def pendulum_equations(t, y, g, l):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Часовий проміжок для моделювання (наприклад, 5 секунд)
t_span = (0, 5)
t_eval = np.linspace(0, 5, 500) # 500 точок для плавності графіка

# Чисельне розв'язання точного рівняння
# y0 = [початковий кут, початкова кутова швидкість]
solution = solve_ivp(pendulum_equations, t_span, [theta0, 0.0], 
                     args=(g, l), t_eval=t_eval, method='RK45')

# Переводимо отриманий кут назад у градуси для наочності на графіку
theta_exact = np.degrees(solution.y[0])

# 3. Обчислення гармонійного (лінійного) наближення для малих кутів
omega_small = np.sqrt(g / l)
theta_harmonic = np.degrees(theta0 * np.cos(omega_small * t_eval))

# 4. Візуалізація та порівняння результатів
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(t_eval, theta_exact, 'b-', linewidth=2.5, label='Точне рішення (SciPy solve_ivp)')
plt.plot(t_eval, theta_harmonic, 'r--', linewidth=2, label='Гармонійне наближення ($\sin\\theta \\approx \\theta$)')

plt.title(f'Порівняння руху маятника при початковому відхиленні $\\theta_0 = {theta0_deg}^\\circ$', fontsize=13, fontweight='bold')
plt.xlabel('Час, $t$ (секунди)', fontsize=11)
plt.ylabel('Кут відхилення, $\\theta$ (градуси)', fontsize=11)
plt.axhline(0, color='black', linestyle=':', alpha=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')

# Обчислення різниці (похибки наближення) наприкінці інтервалу
max_diff = np.max(np.abs(theta_exact - theta_harmonic))
print(f"Максимальна розбіжність між моделями на інтервалі 5 сек: {max_diff:.2f} градусів.")

plt.tight_layout()
plt.show()
