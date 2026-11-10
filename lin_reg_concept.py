import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from fix_data import get_random_subset, load_and_center_data

df = load_and_center_data('Pendulum.xlsx')
subset_df = get_random_subset(df, 10)

t_start_orig = subset_df['Time'].iloc[0]
t_end_orig = subset_df['Time'].iloc[-1]

print(f"Subset original time window: {t_start_orig:.1f} s to {t_end_orig:.1f} s")

subset_df['Time'] = subset_df['Time'] - subset_df['Time'].iloc[0]

t_data = np.array(subset_df['Time'].values)
theta_data = np.array(subset_df['Angle'].values)

def f_series(t_data, theta_data, K_max = 10, tol = 0.01):
    peaks, _ = signal.find_peaks(theta_data) # find peaks

    t_diff = t_data[peaks[-1]] - t_data[peaks[0]]
    T_avg = t_diff / (len(peaks) - 1) # average period

    freq = 2 * np.pi / T_avg

    first_peak = theta_data[peaks[0]]
    last_peak = theta_data[peaks[-1]]
    Lambda = np.log(first_peak / last_peak) / t_diff # decay constant
    decay = np.exp(-Lambda * t_data)

    for K in range(1, K_max + 1):
        cols = []
        for k in range(1, K + 1):
            cols.append(decay * np.cos(k * freq * t_data))
            cols.append(decay * np.sin(k * freq * t_data))
            
        # linear regression to determine A_k and B_k 
        X_k = np.column_stack(cols)
        Y_k = theta_data

        a_k = np.linalg.solve(X_k.T @ X_k, X_k.T @ Y_k)

        theta_model = X_k @ a_k

        MSE = np.mean((theta_data - theta_model)**2)

        if K > 1: # this step is skipped on first iteration 
            rel_improvement = (prev_MSE - MSE) / prev_MSE
            if rel_improvement < tol:
                break

        a = a_k
        K_total = K
        prev_MSE = MSE

    return a, Lambda, freq, K_total

a, Lambda, freq, K_total = f_series(t_data, theta_data)
print(f'Number of cos/sin series in Fourier Series is k = {K_total}')

a = jnp.array(a)
def theta_func(t_m): # m data points (t_m, theta_m)
    decay_t = jnp.exp(-Lambda * t_m)
    terms = jnp.array([
        a[2*(k-1)] * jnp.cos(k * freq * t_m) +
        a[2*(k-1)+1] * jnp.sin(k * freq * t_m)
        for k in range(1, K_total + 1)
    ])
    return decay_t * jnp.sum(terms)

# plot Fourier Series Solution

t_plot = jnp.linspace(t_data[0], t_data[-1], 300)
theta_plot = jax.vmap(theta_func)(t_plot)

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t_plot, theta_plot, label='Continuous $\\theta(t)$', color='black', linewidth=2)
plt.plot(t_data, theta_data, 'ro', label='Data Points')
plt.xlabel('Time (s)')
plt.ylabel('Angle $\\theta$ (rad)')
plt.title('Continuous Fourier Fit vs. Raw Data')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


@jax.jit # for fast computation of jax.grad(jax.grad())
def get_derivatives(t): 
    
    dtheta_dt = jax.vmap(jax.grad(theta_func))(t)
    d2theta_dt2 = jax.vmap(jax.grad(jax.grad(theta_func)))(t)
    
    return dtheta_dt, d2theta_dt2

theta_dot, theta_ddot = get_derivatives(t_data)

# linear regression againts pendulum model: Xa = Y
# physics model: x'' = -a1 * x' - a2 * sin(x), x = theta

Y = theta_ddot
X = jnp.column_stack([-theta_dot, -jnp.sin(theta_data)])

a = jnp.linalg.solve(X.T @ X, X.T @ Y)
a1 = float(a[0])
a2 = float(a[1])
                                   
# a1 = c / m*L^2, a2 = g/L
g = 9.81 # m / s^2                                                                   
L_data = 1.615 # m                                                                   
m_data = 1 # kg                                                                      

L_model = g / a2
L_err = jnp.abs(L_model - L_data) * 100
C_model = a1 * m_data * (L_model**2)

print(f'L_model: {L_model:.3f} m')
print(f'L_model error: {float(L_err):.3f} cm')
print(f'a1: {a1:.3f}')
print(f'Air-Resistance Coef (C): {C_model:.3f}')
