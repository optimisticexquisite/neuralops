import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from pseudo_spectral_initial import velocity_from_omega

L = 100.0  # Length of the box
dt = 0.01  # Time step
num_steps = 500  # Number of steps

# Initial predator and prey parameters
V_predator = 1.0  # Predator speed
V_prey = 1.2  # Prey speed
B_predator = 0.1  # Predator control/mass parameter
B_prey = 0.2  # Prey control/mass parameter
o_predator_angle = 0.0  # Predator direction angle (relative to line joining predator and prey)
o_prey_angle = np.pi  # Prey direction angle (relative to line joining predator and prey)

# Initialize positions and directions
np.random.seed(20)
predator_pos = np.random.rand(2) * L
prey_pos = np.random.rand(2) * L

predator_dir = np.random.rand(2) - 0.5
predator_dir /= np.linalg.norm(predator_dir) 
prey_dir = np.random.rand(2) - 0.5
prey_dir /= np.linalg.norm(prey_dir)

def apply_periodic_boundary(pos, L):
    return np.mod(pos, L)

def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def vector_to_angle(vector):
    return np.arctan2(vector[1], vector[0])

def rk2_update(pos, direction, velocity, control_direction, B):
    # First step
    k1_pos = velocity * direction * dt
    k1_dir = (1/(2*B)) * (control_direction - np.dot(control_direction, direction) * direction) * dt

    # Temporary position and direction
    temp_pos = pos + k1_pos
    temp_dir = direction + k1_dir
    temp_dir /= np.linalg.norm(temp_dir)  # Normalize direction

    # Second step
    k2_pos = velocity * temp_dir * dt
    k2_dir = (1/(2*B)) * (control_direction - np.dot(control_direction, temp_dir) * temp_dir) * dt

    # Update final position and direction
    new_pos = pos + (k1_pos + k2_pos) / 2
    new_dir = direction + (k1_dir + k2_dir) / 2
    new_dir /= np.linalg.norm(new_dir)  # Normalize direction

    return new_pos, new_dir

def rk2_update_turbulent(pos, direction, velocity, omega, control_direction, B, L, N=256, dt=0.01):
    """
    Perform a second-order Runge-Kutta (RK2) update for a particle in a fluid with
    velocity and vorticity effects.
    """
    # Step 1: Calculate fluid velocity using omega
    u, v = velocity_from_omega(omega)

    # Map the particle position in the LxL box to the NxN fluid grid
    fluid_x = (pos[0] / L) * (N - 1)
    fluid_y = (pos[1] / L) * (N - 1)

    # Grid coordinates
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)

    # RegularGridInterpolator expects axes in array-index order: (y, x)
    u_interp_fn = RegularGridInterpolator((y, x), u, method='linear', bounds_error=False, fill_value=None)
    v_interp_fn = RegularGridInterpolator((y, x), v, method='linear', bounds_error=False, fill_value=None)
    omega_interp_fn = RegularGridInterpolator((y, x), omega, method='linear', bounds_error=False, fill_value=None)

    # Interpolate at current position
    point = np.array([fluid_y, fluid_x])
    u_interp = float(u_interp_fn(point))
    v_interp = float(v_interp_fn(point))
    omlag = float(omega_interp_fn(point))

    # First step of RK2 for position and direction
    k1_pos = (
        (u_interp + velocity * direction[0]) * dt,
        (v_interp + velocity * direction[1]) * dt
    )

    k1_dir = (
        ((1 / (2 * B)) * (control_direction[0] - np.dot(control_direction, direction) * direction[0])
         - 0.5 * omlag * direction[1]) * dt,
        ((1 / (2 * B)) * (control_direction[1] - np.dot(control_direction, direction) * direction[1])
         + 0.5 * omlag * direction[0]) * dt
    )

    # Temporary position and direction
    temp_pos = pos + np.array(k1_pos)
    temp_dir = direction + np.array(k1_dir)
    temp_dir /= np.linalg.norm(temp_dir)

    # Map the updated particle position to the fluid grid
    fluid_x = (temp_pos[0] / L) * (N - 1)
    fluid_y = (temp_pos[1] / L) * (N - 1)

    # Interpolate again at updated position
    point = np.array([fluid_y, fluid_x])
    u_interp = float(u_interp_fn(point))
    v_interp = float(v_interp_fn(point))
    omlag = float(omega_interp_fn(point))

    k2_pos = (
        (u_interp + velocity * temp_dir[0]) * dt,
        (v_interp + velocity * temp_dir[1]) * dt
    )

    k2_dir = (
        ((1 / (2 * B)) * (control_direction[0] - np.dot(control_direction, temp_dir) * temp_dir[0])
         - 0.5 * omlag * temp_dir[1]) * dt,
        ((1 / (2 * B)) * (control_direction[1] - np.dot(control_direction, temp_dir) * temp_dir[1])
         + 0.5 * omlag * temp_dir[0]) * dt
    )

    # Update final position and direction
    new_pos = pos + (np.array(k1_pos) + np.array(k2_pos)) / 2
    new_dir = direction + (np.array(k1_dir) + np.array(k2_dir)) / 2
    new_dir /= np.linalg.norm(new_dir)

    return new_pos, new_dir

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')

    axcolor = 'lightgoldenrodyellow'

    # Predator speed slider
    ax_pred_speed = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_pred_speed = Slider(ax_pred_speed, 'Predator Speed', 0.1, 3.0, valinit=V_predator)

    # Prey speed slider
    ax_prey_speed = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
    slider_prey_speed = Slider(ax_prey_speed, 'Prey Speed', 0.1, 3.0, valinit=V_prey)

    # Predator B slider (mass-like parameter)
    ax_B_predator = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
    slider_B_predator = Slider(ax_B_predator, 'Predator B', 0.01, 1.0, valinit=B_predator)

    # Prey B slider (mass-like parameter)
    ax_B_prey = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
    slider_B_prey = Slider(ax_B_prey, 'Prey B', 0.01, 1.0, valinit=B_prey)

    # Predator direction angle slider
    ax_o_predator = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
    slider_o_predator = Slider(ax_o_predator, 'Predator Direction (Angle)', -np.pi, np.pi, valinit=o_predator_angle)

    # Prey direction angle slider
    ax_o_prey = plt.axes([0.1, 0.3, 0.65, 0.03], facecolor=axcolor)
    slider_o_prey = Slider(ax_o_prey, 'Prey Direction (Angle)', -np.pi, np.pi, valinit=o_prey_angle)

    # Update values when sliders are changed
    def update_predator_speed(val):
        global V_predator
        V_predator = slider_pred_speed.val

    def update_prey_speed(val):
        global V_prey
        V_prey = slider_prey_speed.val

    def update_B_predator(val):
        global B_predator
        B_predator = slider_B_predator.val

    def update_B_prey(val):
        global B_prey
        B_prey = slider_B_prey.val

    def update_o_predator(val):
        global o_predator_angle
        o_predator_angle = slider_o_predator.val

    def update_o_prey(val):
        global o_prey_angle
        o_prey_angle = slider_o_prey.val

    # Connect the sliders to the update functions
    slider_pred_speed.on_changed(update_predator_speed)
    slider_prey_speed.on_changed(update_prey_speed)
    slider_B_predator.on_changed(update_B_predator)
    slider_B_prey.on_changed(update_B_prey)
    slider_o_predator.on_changed(update_o_predator)
    slider_o_prey.on_changed(update_o_prey)

    # Animate the simulation
    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=30
    )

    plt.savefig("predator_prey_simulation_q2_turbulent.png")
