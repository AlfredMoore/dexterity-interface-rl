#include "robot_motion_interface/panda_interface.hpp"

#include <iostream>
#include <stdexcept>

namespace robot_motion_interface {


PandaInterface::PandaInterface(std::string hostname, std::string urdf_path, std::vector<std::string> joint_names,
    const Eigen::VectorXd& kp, const Eigen::VectorXd& kd, double max_joint_delta)
    : robot_(hostname) {
    
    std::array<double, 7> lower_torque_thresholds_nominal{{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.}};
    std::array<double, 7> upper_torque_thresholds_nominal{{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
    std::array<double, 7> lower_torque_thresholds_acceleration{{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0}};
    std::array<double, 7> upper_torque_thresholds_acceleration{{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
    std::array<double, 6> lower_force_thresholds_nominal{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_nominal{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
    std::array<double, 6> lower_force_thresholds_acceleration{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_acceleration{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
    robot_.setCollisionBehavior(
        lower_torque_thresholds_acceleration, upper_torque_thresholds_acceleration,
        lower_torque_thresholds_nominal, upper_torque_thresholds_nominal,
        lower_force_thresholds_acceleration, upper_force_thresholds_acceleration,
        lower_force_thresholds_nominal, upper_force_thresholds_nominal);
    
    
    rp_ = std::make_unique<robot_motion::RobotProperties>(joint_names, urdf_path);
    controller_ = std::make_unique<robot_motion::JointTorqueController>(*rp_, kp, kd, false, max_joint_delta);
};


void PandaInterface::set_joint_positions(const Eigen::VectorXd& q){
    controller_->set_setpoint(q);
};



Eigen::VectorXd PandaInterface::joint_state() {
    if (control_loop_running_) {
        std::lock_guard<std::mutex> lock(this->control_loop_mutex_);
        return control_loop_state_;

    } else { 
        // This can only be used when control loop is NOT running
        franka::RobotState robot_state = robot_.readOnce();

        Eigen::VectorXd q = array_to_eigen(robot_state.q);
        Eigen::VectorXd dq = array_to_eigen(robot_state.dq);
        Eigen::VectorXd state(14); state << q, dq;

        return state;
    }

};



void PandaInterface::start_loop() {
    // freeze before starting
    franka::RobotState initial_state = robot_.readOnce();
    Eigen::VectorXd q_init = array_to_eigen(initial_state.q);
    controller_->set_setpoint(q_init);

    // Put in own thread
    control_thread_ = std::thread([this]() {

        std::function<franka::Torques(const franka::RobotState&, franka::Duration)> 
        callback = [this](const franka::RobotState& robot_state, franka::Duration time_step) -> franka::Torques {
            Eigen::VectorXd q = array_to_eigen(robot_state.q);
            Eigen::VectorXd dq = array_to_eigen(robot_state.dq);
            Eigen::VectorXd state(14); state << q, dq;
          
            {  // Update shared variable within mutex lock
                std::lock_guard<std::mutex> lock(this->control_loop_mutex_);
                this->control_loop_state_ = state;
    
            }

            Eigen::VectorXd tau = this->controller_->step(state);
            
            franka::Torques torques(eigen_to_array<7>(tau));
            
            return torques;
        };
    
        try {
            this->control_loop_running_ =  true;
            this->robot_.control(callback, true);
        } catch (const franka::Exception& e) {
            std::cout << e.what() << std::endl;
            this->control_loop_running_ =  false;
        }
        
    });

};

void PandaInterface::stop_loop() {
    if (!control_loop_running_) return;

    control_loop_running_ =  false;

    try {
        robot_.stop();
    } catch (const franka::Exception& e) {
        std::cerr << "[Recovering from Franka Stop Error] " << e.what() << std::endl;
        robot_.automaticErrorRecovery();

    }
    if (control_thread_.joinable()) control_thread_.join();
}


// ---------------------------------------------------------------------------
// SYSID read-only diagnostics. All require the control loop STOPPED (they call
// robot_.readOnce()/loadModel(), which cannot run concurrently with control()).
// ---------------------------------------------------------------------------

Eigen::VectorXd PandaInterface::sysid_load_info() {
    if (control_loop_running_)
        throw std::runtime_error("sysid_load_info requires the control loop stopped");
    franka::RobotState s = robot_.readOnce();
    Eigen::VectorXd out(15);
    out(0) = s.m_ee;
    out(1) = s.m_load;
    out(2) = s.m_total;
    for (int i = 0; i < 3; ++i) out(3 + i) = s.F_x_Cload[i];
    for (int i = 0; i < 9; ++i) out(6 + i) = s.I_load[i];
    return out;
}

Eigen::VectorXd PandaInterface::sysid_gravity(const Eigen::VectorXd& q) {
    if (control_loop_running_)
        throw std::runtime_error("sysid_gravity requires the control loop stopped");
    if (!model_) model_ = std::make_unique<franka::Model>(robot_.loadModel());

    franka::RobotState s = robot_.readOnce();
    std::array<double, 7> q_arr;
    if (q.size() == 0) {
        q_arr = s.q;  // current measured pose
    } else if (q.size() == 7) {
        for (int i = 0; i < 7; ++i) q_arr[i] = q(i);
    } else {
        throw std::runtime_error("sysid_gravity: q must be empty or length 7");
    }
    // Gravity torque under the CONFIGURED load (m_total / F_x_Ctotal from Desk).
    std::array<double, 7> g = model_->gravity(q_arr, s.m_total, s.F_x_Ctotal);
    Eigen::VectorXd out(7);
    for (int i = 0; i < 7; ++i) out(i) = g[i];
    return out;
}

Eigen::VectorXd PandaInterface::sysid_tau_ext() {
    if (control_loop_running_)
        throw std::runtime_error("sysid_tau_ext requires the control loop stopped");
    franka::RobotState s = robot_.readOnce();
    return array_to_eigen(s.tau_ext_hat_filtered);
}

Eigen::VectorXd PandaInterface::sysid_tau_measured() {
    if (control_loop_running_)
        throw std::runtime_error("sysid_tau_measured requires the control loop stopped");
    franka::RobotState s = robot_.readOnce();
    return array_to_eigen(s.tau_J);
}

bool PandaInterface::sysid_set_load(double mass, const Eigen::Vector3d& com,
                                    const Eigen::Matrix3d& inertia) {
    if (control_loop_running_)
        throw std::runtime_error("sysid_set_load requires the control loop stopped");
    std::array<double, 3> com_arr{com(0), com(1), com(2)};
    std::array<double, 9> I_arr;  // row-major 3x3
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) I_arr[3 * r + c] = inertia(r, c);
    try {
        robot_.setLoad(mass, com_arr, I_arr);
        return true;
    } catch (const franka::Exception& e) {
        std::cerr << "[PandaInterface] sysid_set_load failed: " << e.what() << std::endl;
        return false;
    }
}

} 
