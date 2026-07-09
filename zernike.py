import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MaxNLocator

def create_grid(n_points):
    """
    Creates a 2D grid of points (X, Y) from -1 to 1.
    Returns the polar coordinates R, Theta, the unit disk mask, and the X, Y matrices.
    """
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0
    return R, Theta, mask, X, Y

def zernike(n, m, rho, theta):
    """
    Calculates the Zernike polynomial Z_n^m(rho, theta) within the polar domain.
    """
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        c = ((-1)**k * math.factorial(n - k)) / \
            (math.factorial(k) * math.factorial((n + abs(m))//2 - k) * math.factorial((n - abs(m))//2 - k))
        R += c * rho**(n - 2*k)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(-m * theta)
    else:
        return R

def zernike_sum(coeffs, modes, rho, theta, mask):
    """
    Calculates the weighted sum of Zernike polynomials, applying the circular mask.
    """    
    Z = np.zeros_like(rho)
    mask = rho <= 1
    
    for c, (n, m) in zip(coeffs, modes):
        Z[mask] += c * zernike(n, m, rho[mask], theta[mask])
    
    Z[~mask] = np.nan
    return Z

def show_zernike_3d(X, Y, Z):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', vmin=-1, vmax=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-1, 1)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    
    plt.show()

def show_zernike_2d(X, Y, Z):
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()),
               origin='lower', cmap='viridis', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(label="Wavefront deformation")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.linspace(X.min(), X.max(), 5))
    plt.yticks(np.linspace(Y.min(), Y.max(), 5))
    plt.grid(False)
    plt.show()

def psf_generator(Z, mask):
    Z_clean = np.zeros_like(Z)
    Z_clean[mask] = Z[mask]

    pupil = np.zeros_like(Z)
    pupil[mask] = 1.0

    pupil_function = pupil * np.exp(1j * Z_clean)

    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function)))**2
    
    pupil_ideal = pupil
    psf_ideal = np.abs(np.fft.fftshift(np.fft.fft2(pupil_ideal)))**2
    
    Strehl_Ratio = psf.max() / psf_ideal.max()
    return psf, Strehl_Ratio

def show_psf_zoomed(psf, zoom_size):
    center = psf.shape[0] // 2
    half = zoom_size // 2
    zoomed = psf[center - half:center + half, center - half:center + half]

    plt.figure(figsize=(6, 6))
    plt.imshow(zoomed, cmap='Greys_r', extent=[-half, half, -half, half])
    plt.title(f"Central zoom of the PSF ({zoom_size}x{zoom_size})")
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    plt.colorbar(label='Intensity (normalized)')
    plt.show()
    
def show_psf(psf):
    plt.figure(figsize=(6, 6))
    plt.imshow(psf, cmap='Greys_r')
    plt.title("PSF")
    plt.colorbar(label='Intensity (normalized)')
    plt.show()

def complex_pupil(A, Mask):
    abbe = np.exp(1j * A)
    abbe_z = Mask * abbe
    return abbe_z

def PSF(complx_pupil):
    # Using a different variable name to avoid shadowing the function name
    PSF_val = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(complx_pupil))) 
    PSF_val = (np.abs(PSF_val))**2
    PSF_val = PSF_val / PSF_val.sum()
    return PSF_val


# Zernike modes up to the 5th order
modes = [
    (0, 0),  
    (1, -1), (1, 1),
    (2, -2), (2, 0),  (2, 2),
    (3, -3), (3, -1), (3, 1),  (3, 3),
    (4, -4), (4, -2), (4, 0),  (4, 2),  (4, 4)
]

# Active initialization (old commented arrays have been removed)
coeffs = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    print("Executing the main script")
    
    rho, theta, mask, X, Y = create_grid(256)
    Z = zernike_sum(coeffs, modes, rho, theta, mask)
    
    show_zernike_3d(X, Y, Z)
    show_zernike_2d(X, Y, Z)
    
    psf, Strehl_Ratio = psf_generator(Z, mask)
    
    show_psf(psf)
    show_psf_zoomed(psf, 50)
    
    print("Strehl Ratio = " + str(Strehl_Ratio))
    
    Z[np.isnan(Z)] = 0

    pupil_com = complex_pupil(Z, mask)
    psf_log = PSF(pupil_com)
    
    plt.figure(figsize=(18, 10))
    plt.imshow(np.log(np.abs(psf_log)), cmap='viridis')
    plt.colorbar()
    
    fig, axes = plt.subplots(5, 9, figsize=(16, 10))
    fig.suptitle("Zernike Modes (1-15) in a pyramidal layout", fontsize=18)

    layout = {
        0: [4],
        1: [3, 5],
        2: [4, 6, 2],
        3: [3, 5, 1, 7],
        4: [4, 2, 6, 0, 8]
    }

    idx = 0
    for row in range(5):
        for col in range(9):
            ax = axes[row, col]
            ax.axis('off')
            if col in layout[row] and idx < len(modes):
                n, m = modes[idx]
                Z_pyramid = zernike(n, m, rho, theta)
                Z_pyramid[~mask] = np.nan
                ax.imshow(Z_pyramid, extent=(X.min(), X.max(), Y.min(), Y.max()),
                          origin='lower', cmap='viridis', vmin=-1, vmax=1)
                idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()