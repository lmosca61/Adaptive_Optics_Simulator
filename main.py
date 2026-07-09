import numpy as np
import matplotlib.pyplot as plt
import control as ct
import math
import warnings

from zernike import create_grid, zernike_sum, show_zernike_3d, show_zernike_2d, psf_generator, complex_pupil, PSF

warnings.filterwarnings("ignore")

# Zernike modes up to order 5
modes = [
    (1, -1), (1, 1),
    (2, 0), (2, -2), (2, 2),
    (3, -1), (3, 1), (3, -3), (3, 3),
    (4, 0), (4, -2), (4, 2), (4, -4), (4, 4),
]

# Parameters
N = len(modes)                
dt = 0.001

tau_1 = 0.005715026151380664
alpha_1 = 0.007654266245552365
tau_2 = 10660589.177999858
alpha_2 = 8.270039193705596e-12
mu = 1

num_R1 = [(2 * tau_1 + dt), (-2 * tau_1 + dt)]
den_R1 = [(2 * alpha_1 * tau_1 + dt), (-2 * alpha_1 * tau_1 + dt)]
num_R2 = [(2 * tau_2 + dt), (-2 * tau_2 + dt)]
den_R2 = [(2 * alpha_2 * tau_2 + dt), (-2 * alpha_2 * tau_2 + dt)]
num_GG = [1e-9, 3e-9, 3e-9, 1e-9]
den_GG = [240, -720, 720, -240]

R1 = ct.TransferFunction(num_R1, den_R1, dt)
R2 = ct.TransferFunction(num_R2, den_R2, dt)
GG = ct.TransferFunction(num_GG, den_GG, dt)

RR = mu * R1 * R2
LL = GG * RR
FF = LL / (1 + LL)

ritardo = ct.tf([1], [1, 0], dt=dt)

# Composite system with N copies of FF
systems = [FF * ritardo for _ in range(N)]

# Diagonal MIMO system with control.append
F_mimo = ct.append(*systems)  # System with N independent inputs and outputs

t = np.arange(0, dt * 1000, dt)
print("LEN", len(t))

# Inputs: step on all channels
u = np.ones((N, len(t)))

i = 0
t_resp, y_resp = ct.step_response(F_mimo, T=t, input=i)

plt.figure()
y_plot = np.squeeze(y_resp[i])  # Only output i
plt.plot(t_resp, y_plot, label=f'Output {i}', color='tab:blue')

mu = 0.45236917288633804  # SISO optimization

num_R1 = [(2 * tau_1 + dt), (-2 * tau_1 + dt)]
den_R1 = [(2 * alpha_1 * tau_1 + dt), (-2 * alpha_1 * tau_1 + dt)]
num_R2 = [(2 * tau_2 + dt), (-2 * tau_2 + dt)]
den_R2 = [(2 * alpha_2 * tau_2 + dt), (-2 * alpha_2 * tau_2 + dt)]
num_GG = [1e-9, 3e-9, 3e-9, 1e-9]
den_GG = [240, -720, 720, -240]

R1 = ct.TransferFunction(num_R1, den_R1, dt)
R2 = ct.TransferFunction(num_R2, den_R2, dt)
GG = ct.TransferFunction(num_GG, den_GG, dt)

RR = mu * R1 * R2
LL = GG * RR
FF = LL / (1 + LL)

ritardo = ct.tf([1], [1, 0], dt=dt)

# Composite system with N copies of FF
systems = [FF * ritardo for _ in range(N)]

# Diagonal MIMO system with control.append
F_mimo = ct.append(*systems)  # System with N independent inputs and outputs

t = np.arange(0, dt * 1000, dt)
print("LEN", len(t))

# Inputs: step on all channels
u = np.ones((N, len(t)))

i = 0
    
t_resp, y_resp = ct.step_response(F_mimo, T=t, input=i)
y_plot = np.squeeze(y_resp[i])  # Only output i
plt.plot(t_resp, y_plot, label=f'Output {i}', color='tab:red')
    
plt.title(f'Step response - Input {i} (Output {i} only)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.xlim(-0.001, dt * 100)
plt.grid(True)
plt.show()

# Random aberration: generates an array of uniform random numbers
coeffs = np.random.uniform(-3, 3, N)

C = np.array([
    [0.4557, 0,      0,      0,      0,      0,     -0.0144, 0,      0,      0,      0,      0,      0,      0],
    [0,      0.4557, 0,      0,      0,     -0.0144, 0,      0,      0,      0,      0,      0,      0,      0],
    [0,      0,      0.0236, 0,      0,      0,      0,      0,      0,      0,     -0.0039, 0,      0,      0],
    [0,      0,      0,      0.0236, 0,      0,      0,      0,      0,      0,      0,      0,     -0.0039, 0],
    [0,      0,      0,      0,      0.0236, 0,      0,      0,      0,      0,      0,     -0.0039, 0,      0],
    [0,     -0.0144, 0,      0,      0,      0.0063, 0,      0,      0,      0,      0,      0,      0,      0],
    [-0.0144, 0,     0,      0,      0,      0,      0.0063,  0,      0,      0,      0,      0,      0,      0],
    [0,      0,      0,      0,      0,      0,      0,      0.0063, 0,      0,      0,      0,      0,      0],
    [0,      0,      0,      0,      0,      0,      0,      0,      0.0063, 0,      0,      0,      0,      0],
    [0,      0,     -0.0039, 0,      0,      0,      0,      0,      0,      0,      0.0025, 0,      0,      0],
    [0,      0,      0,      0,     -0.0039, 0,      0,      0,      0,      0,      0,      0.0025, 0,      0],
    [0,      0,      0,     -0.0039, 0,      0,      0,      0,      0,      0,      0,      0,      0.0025, 0],
    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0.0025, 0],
    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0.0025],
])
        
U, S, U_T = np.linalg.svd(C)

B = coeffs  # Generated coefficients -> K_L

# Active coefficients for Zernike (overriding previous calculations based on your script)
A = [1.46300418, -2.97931061, 0.83997741, 2.35268212, -0.39967532, 2.80328405, 1.85497300, -2.36442946, -1.93991169, -2.36104877, 2.26435252, -2.39057924, -2.36320192, -2.16545925]
print("A", A)  # Coefficients for Zernike

check = U @ A

ingressi = np.zeros((N, len(t)))

for i in range(N):
    rumore = np.random.uniform(-0.1, 0.1, N)
    R = U_T @ rumore
    ingressi[i][0] = A[i] + R[i]  # First column contains the found coefficients
    
# Dynamics data
V = 10  # m/s: wind speed
D = 2   # m: mirror diameter with unit radius
n = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]  # Radial order of modes

for i in range(N):
    for j in range(1, len(t), 1):
        rumore = np.random.uniform(-0.1, 0.1, N)
        R = U_T @ rumore
        ingressi[i][j] = ingressi[i][j-1] * math.exp(-0.3 * (n[i] + 1) * V * dt / D) + R[i]
        
selected_inputs = [0, 4, 13]

# Time axis (optional, if you have a custom time vector)
time = np.arange(1000)  # 0, 1, ..., 999

# Plot
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, N))  # Viridis color scale for N outputs
for idx in selected_inputs:
    plt.plot(time, ingressi[idx], label=f'Input {idx+1}', color=colors[idx])

plt.title('Evolution of 3 inputs over time')
plt.xlabel('k')
plt.ylabel('y_ref(k)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
        
# Create grid
rho, theta, mask, X, Y = create_grid(256)

# Calculate the sum of Zernike modes
Z = zernike_sum(A, modes, rho, theta, mask)

# Show 3D and 2D images of Zernike modes
show_zernike_3d(X, Y, Z)
show_zernike_2d(X, Y, Z)

# Generate the point spread function
psf, Strehl_Ratio = psf_generator(Z, mask)

Z[np.isnan(Z)] = 0

pupil_com = complex_pupil(Z, mask)
psf = PSF(pupil_com)
plt.figure(figsize=(18, 10))
plt.imshow(np.log(np.abs(psf)), cmap='viridis')
plt.colorbar()

print("Strehl Ratio before correction = " + str(Strehl_Ratio))

t_out, y_out = ct.forced_response(F_mimo, T=t, U=ingressi)

colors = plt.cm.viridis(np.linspace(0, 1, N))  # Viridis color scale for N outputs

plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t_out, y_out[i, :],  color=colors[i], label=f'Output y{i+1}')
plt.title(f'Response - Discrete MIMO System ({N} modes)')
plt.xlabel('k')
plt.ylabel('y(k)')
plt.ylim(-5, 5)
plt.xlim(0, dt * 1000)
plt.grid(True)
plt.legend(loc='lower right', ncol=3, fontsize='small')
plt.tight_layout()
plt.show()

coeffs_corretti = np.zeros((N, len(t)))

for i in range(len(y_out)):
    for j in range(1, len(t), 1):
        coeffs_rilevati = y_out[:, j]  # j-th column of outputs -> coefficient values at time j
        coeffs_corretti[i][j] = -coeffs_rilevati[i] + ingressi[i][j]

rho, theta, mask, X, Y = create_grid(256)
Z_corretto = zernike_sum(coeffs_corretti[:, -1], modes, rho, theta, mask)

# For visualization, take the corrected coefficients at the last simulation instant
show_zernike_3d(X, Y, Z_corretto)
show_zernike_2d(X, Y, Z_corretto)

# Generate the point spread function
psf, Strehl_Ratio = psf_generator(Z_corretto, mask)

Z_corretto[np.isnan(Z_corretto)] = 0

pupil_com = complex_pupil(Z_corretto, mask)
psf = PSF(pupil_com)
plt.figure(figsize=(18, 10))
plt.imshow(np.log(np.abs(psf)), cmap='viridis')
plt.colorbar()

print("Strehl Ratio after correction = " + str(Strehl_Ratio))

plt.figure(figsize=(10, 6))

for i, riga in enumerate(coeffs_corretti):
    plt.plot(t, riga, label=f'phi_res_{i+1}')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time evolution of phi_res_i')
plt.legend(loc='upper right', ncol=2)  # Can be disabled if there are too many
plt.grid(True)
plt.tight_layout()
plt.xlim(-0.001, 1)
plt.ylim(-4.2, 4.2)
plt.show()

plt.figure(figsize=(10, 6))

for i, riga in enumerate(coeffs_corretti):
    plt.plot(t, riga, label=f'phi_res_{i+1}')

plt.show()
