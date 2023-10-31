import numpy as np


# 从位置估计速度：扩展卡尔曼
def calculate_velocity(position_sequence, dt):
    velocities = []
    for i in range(1, len(position_sequence)):
        dx = position_sequence[i][0] - position_sequence[i - 1][0]
        dy = position_sequence[i][1] - position_sequence[i - 1][1]
        dz = position_sequence[i][2] - position_sequence[i - 1][2]
        velocity = [dx / dt, dy / dt, dz / dt]
        velocities.append(velocity)
    return velocities

def ekf_state_transition(state, dt):
    # Nonlinear state transition function for 3D position and velocity
    x, y, z, vx, vy, vz = state
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_z = z + vz * dt
    new_state = np.array([new_x, new_y, new_z, vx, vy, vz])
    return new_state

def ekf_measurement_model(state):
    # Nonlinear measurement model for 3D position
#     x, y, z, _, _, _ = state
    x, y, z = state
    measurement = np.array([x, y, z])
    return measurement

def ekf_with_velocity_estimation(position_sequence, initial_estimate, initial_error, process_noise, measurement_noise, dt):
    # Estimate velocities from position sequence using one-step differentiation
    velocities = calculate_velocity(position_sequence, dt)

    # EKF parameters
    Q = np.array([[process_noise, 0, 0, 0, 0, 0],       # Process noise covariance
                  [0, process_noise, 0, 0, 0, 0],
                  [0, 0, process_noise, 0, 0, 0],
                  [0, 0, 0, process_noise, 0, 0],
                  [0, 0, 0, 0, process_noise, 0],
                  [0, 0, 0, 0, 0, process_noise]])

    R = np.array([[measurement_noise, 0, 0],           # Measurement noise covariance
                  [0, measurement_noise, 0],
                  [0, 0, measurement_noise]])

    estimated_states = [initial_estimate]

    for i in range(1, len(position_sequence)):
        dt = 1  # Assuming constant time intervals for simplicity

        # Prediction step
        predicted_state = ekf_state_transition(estimated_states[-1], dt)
        F = np.eye(6) + np.array([[0, 0, 0, dt, 0, 0],    # Jacobian of state transition function
                                  [0, 0, 0, 0, dt, 0],
                                  [0, 0, 0, 0, 0, dt],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])

        predicted_P = np.dot(np.dot(F, initial_error), F.T) + Q

        # Update step
        measurement = ekf_measurement_model(position_sequence[i])
        H = np.array([[1, 0, 0, 0, 0, 0],    # Jacobian of measurement model
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

        y = measurement - np.dot(H, predicted_state)
        S = np.dot(np.dot(H, predicted_P), H.T) + R
        K = np.dot(np.dot(predicted_P, H.T), np.linalg.inv(S))

        updated_state = predicted_state + np.dot(K, y)
        updated_P = predicted_P - np.dot(np.dot(K, H), predicted_P)

        estimated_states.append(updated_state)

    return estimated_states

# ========从位置估计速度 ========
# ========直接输入numpy序列：尺寸（N，3）即可，注意时间间隔：100Hz即为0.01s
def pos_estimate_vel(position_sequence, time_interval):
#     position_sequence = [[10, 20, 30], [12, 22, 28], [18, 26, 32], [20, 30, 35], [22, 32, 38]]  # Replace this with your actual 3D position data
    initial_estimate = np.concatenate((position_sequence[0], np.array([0, 0, 0])))  # Add initial velocity estimation of [0, 0, 0]
    initial_error = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    process_noise_variance = 0.01
    measurement_noise_variance = 3.0

    estimated_states = ekf_with_velocity_estimation(position_sequence,
                                                    initial_estimate,
                                                    initial_error,
                                                    process_noise_variance,
                                                    measurement_noise_variance,
                                                    time_interval)
    return np.array(estimated_states) # 包含位置、速度的观测，N*6维度

# pos_data = np.random.randn(10000, 3)
# pos_estimate_vel(pos_data, 0.01)

# 【由位置点计算速度】
# 输入为N*2的pos li
# 输出格式
# type, yaw, pitch, roll,  vx,    vy,   vz  duration  gps_vis # 可自由加减行
# [  1,    0,    0,    0,    0,    0,    0,    1,    1], 
# [  2,    0,    0,    0,   0,  10,    0,    10,    1], 
# [  2,    0,    0,    0,   10,  0,    0,    10,    1], 
def get_vel_li_from_pos(pos_li, vel_avg):
    
    # 遍历计算
    for i, pos_line in enumerate(pos_li):
        if i == 0 :
            continue
        vel_raw.append(pos_li[i] - pos_li[i-1])
        
        
        
    
    return vel_li
    