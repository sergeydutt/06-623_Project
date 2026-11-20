import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp # for autograd
from jax import jit # for fast computing

import numpy as np # use for everything else except derivatives
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp # to check overfitting

from fix_data import get_random_subset, load_and_center_data

df = load_and_center_data("Pendulum.xlsx")
total_peaks = 20
train_peaks = 10
subset_df = get_random_subset(df, total_peaks) 

original_t_start = subset_df["Time"].iloc[0]
original_t_end = subset_df["Time"].iloc[-1]

print(f"Original Subset Time Window: {original_t_start:.1f} s to {original_t_end:.1f} s\
")

def get_t_and_theta(t, theta):
    # start subset at t = 0                                                             
    subset_df[t] = subset_df[t] - subset_df[t].iloc[0]

    x = np.array(subset_df[t].values)
    y = np.array(subset_df[theta].values)
Because only discrete measurements of θ
θ are available, we first reconstruct a continuous, smooth, and differentiable representation of the trajectory. Once established, automatic differentiation (via JAX) can be applied directly to compute the time derivatives θ˙(t)
θ˙(t) and θ¨(t)
θ¨(t). Alternatively, one could estimate a1
a1​ and a2
a2​ using nonlinear least-squares optimization. However, because Eq. (8) lacks a closed-form analytical solution—and is inherently nonlinear in θ
θ even in the undamped case (Eq. 2)—nonlinear optimization requires repeated numerical integration and good initial parameter guesses. By leveraging automatic differentiation to evaluate the derivatives directly, we bypass iterative optimization and initial guesses altogether, allowing a1
a1​ and a2
a2​ to be solved efficiently via standard linear regression.
    print(f"Number of Data Points {len(x):.1f}")

    return x, y

t_full, theta_full = get_t_and_theta(t = "Time", theta = "Angle")

peaks, _ = signal.find_peaks(theta_full)
split = peaks[train_peaks]

t_data = t_full[:split]
theta_data = theta_full[:split]
t_test = t_full[split:]
theta_test = theta_full[split:]

print(f"Train Set: {len(t_data)} pts ({t_data[0]:.2f}s to {t_data[-1]:.2f}s)")
print(f"Test Set: {len(t_test)} pts ({t_test[0]:.2f}s to {t_test[-1]:.2f}s)")

def f_series(t_data, theta_data,
             K_max = 10, tol = 0.01,
             search_for_freq_opt=True, freq_opt=None):

    peaks, _ = signal.find_peaks(theta_data)

    t_diff = t_data[peaks[-1]] - t_data[peaks[0]]
    avg_period = t_diff / (len(peaks) - 1)
    freq_guess = 2 * np.pi / avg_period

    first_peak = theta_data[peaks[0]]
    last_peak = theta_data[peaks[-1]]
    Lambda = np.log(first_peak / last_peak) / t_diff
    decay = np.exp(-Lambda * t_data)

    def lin_reg(f, K):
        cols = []
        for k in range(1, K + 1):
            cols.append(decay * np.cos(k * f * t_data))
            cols.append(decay * np.sin(k * f * t_data))

        X = np.column_stack(cols)
        a = np.linalg.solve(X.T @ X, X.T @ theta_data)
        theta_model = X @ a
        mse = np.mean((theta_data - theta_model) ** 2)
        return a, mse

    # Mode 1: Find optimal frequency                                                    
    if search_for_freq_opt:
        res = minimize_scalar(
            lambda f: lin_reg(f, K_max)[1],
            bounds=(freq_guess * 0.95, freq_guess * 1.05),
            method="bounded",
        )
        freq_opt = res.x
        print(f"Optimal Frequency is: {freq_opt:.3f}")
        return freq_opt

    # Mode 2: Fit Fourier series using freq_opt
    for K in range(1, K_max + 1):
        a_k, MSE = lin_reg(freq_opt, K)

        if K > 1:
            rel_improvement = (prev_MSE - MSE) / prev_MSE
            if rel_improvement < tol:
                break

        a = a_k
        K_total = K
        prev_MSE = MSE

    return Lambda, a, K_total

freq_opt = f_series(t_data, theta_data)

Lambda, a_fourier, K_total = f_series(
    t_data = t_data,
    theta_data = theta_data,
    search_for_freq_opt = False,
    freq_opt = freq_opt,
)

print(f"Number of cos/sin series in Fourier Series is k = {K_total}")

a_fourier = jnp.array(a_fourier)
def theta_func(t_m): # m data points (t_m, theta_m)                                     
    decay_t = jnp.exp(-Lambda * t_m)
    terms = jnp.array([
        a_fourier[2*(k-1)] * jnp.cos(k * freq_opt * t_m) +
        a_fourier[2*(k-1)+1] * jnp.sin(k * freq_opt * t_m)
        for k in range(1, K_total + 1)
    ])
    return decay_t * jnp.sum(terms)

@jit # for fast computation of jax.grad(jax.grad())                                     
def get_derivatives(t):

    dtheta_dt = jax.vmap(jax.grad(theta_func))(t)
    d2theta_dt2 = jax.vmap(jax.grad(jax.grad(theta_func)))(t)

    return dtheta_dt, d2theta_dt2

theta_dot, theta_ddot = get_derivatives(t_data)

# plot Fourier Series Solution                                                          
t_plot = jnp.linspace(t_data[0], t_data[-1], 300)
theta_plot = np.asarray(jax.vmap(theta_func)(jnp.asarray(t_plot)))
theta_dot, theta_ddot = map(np.asarray, get_derivatives(jnp.asarray(t_data)))

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t_plot, theta_plot, label="Continuous $\\theta(t)$", color="black", linewidth=\
2)
plt.plot(t_data, theta_data, "ro", label="Data Points")
plt.xlabel("Time (s)")
plt.ylabel("Angle $\\theta$ (rad)")
plt.title("Continuous Fourier Fit vs. Raw Data")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.show()


# linear regression againts pendulum model: Xa = Y                                      
# physics model: x'' = -a1 * x' - a2 * sin(x), x = theta                                
Y = theta_ddot
X = np.column_stack([-theta_dot, -np.sin(theta_data)])

a = np.linalg.solve(X.T @ X, X.T @ Y)
a1 = float(a[0])
a2 = float(a[1])

# a1 = c / m*L^2, a2 = g/L                                                              
g = 9.81 # m / s^2                                                                      
m_data = 1 # kg                                                                         
L_model = g / a2
C_model = a1 * m_data * (L_model**2)

print(f"L_model: {L_model:.3f} m")
print(f'Air-Resistance Coef (C): {C_model:.3f}')

# use a1 and a2 from linear regression and created simulated data with solve_ivp
# compare simulated data to actual data

def pendulum_ode(t_val, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -a1 * omega - a2 * np.sin(theta)
    return [dtheta_dt, domega_dt]

theta_0 = float(theta_func(t_data[0]))
omega_0 = float(theta_dot[0])
y0 = [theta_0, omega_0]

sol = solve_ivp(
    fun = pendulum_ode,
    t_span = (t_full[0], t_full[-1]),
    y0 = y0,
    t_eval = t_full,
    method = "RK45",
    rtol = 1e-9,
    atol = 1e-11
)
theta_sim = sol.y[0]
dtheta_sim = sol.y[1]

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t_plot, theta_plot, label="Continuous $\\theta(t)$", color="black", linewidth=\
2)
plt.plot(t_full, theta_full, "ro", label="Data Points")
plt.plot(t_full, theta_sim, "bo", label="Simulated Data Points")
plt.xlabel("Time (s)")
plt.ylabel("Angle $\\theta$ (rad)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.show()

theta_sim_full = sol.y[0]

# Split simulated trajectory into train and unseen test segments
theta_sim_train = theta_sim_full[:split]
theta_sim_test = theta_sim_full[split:]

train_rmse_deg = np.degrees(np.sqrt(np.mean((theta_data - theta_sim_train) ** 2)))
test_rmse_deg = np.degrees(np.sqrt(np.mean((theta_test - theta_sim_test) ** 2)))

print(f"\nTrain RMSE (Peaks 1–10):  {train_rmse_deg:.3f} deg")
print(f"Test RMSE  (Peaks 11–20): {test_rmse_deg:.3f} deg")

# Plot Train vs. Unseen Test 
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), dpi=110, sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1]}
)

ax1.plot(t_data, np.degrees(theta_data), "ko", markersize=3.5, alpha=0.6, label="Train Data (Peaks 1–10)")
ax1.plot(t_test, np.degrees(theta_test), "bo", markersize=3.5, alpha=0.6, label="Unseen Test Data (Peaks 11–20)")
ax1.plot(t_full, np.degrees(theta_sim_full), color="crimson", linewidth=2,
         label=rf"ODE ($a_1={a1:.4f}, a_2={a2:.4f}$)")
ax1.axvline(t_full[split], color="gray", linestyle="--", linewidth=1.2, label="Train / Test Split")
ax1.set_ylabel("Angle (deg)")
ax1.set_title("Temporal Generalization: Fit on First 10 Peaks, Forecast Next 10 Peaks")
ax1.legend(loc="upper right", fontsize=8.5)
ax1.grid(True, alpha=0.25)

error_full_deg = np.degrees(theta_full - theta_sim_full)
ax2.plot(t_data, error_full_deg[:split], color="black", linewidth=1.2, label=f"Train RMSE = {train_rmse_deg:.3f}°")
ax2.plot(t_test, error_full_deg[split:], color="tab:blue", linewidth=1.2, label=f"Test RMSE = {test_rmse_deg:.3f}°")
ax2.axvline(t_full[split], color="gray", linestyle="--", linewidth=1.2)
ax2.axhline(0, color="k", linestyle=":", linewidth=0.9)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Error (deg)")
ax2.legend(loc="upper left", fontsize=8.5)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()
