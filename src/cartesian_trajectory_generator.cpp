#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstdlib> 

int main() {
    const double dt = 0.001; // 1 ms sampling
    const double total_time = 35.0; // total trajectory duration [s] (includes initial hold)
    const double initial_hold = 5.0; // initial flat segment [s]
    const double dwell_time = 3.0; // dwell duration at each peak [s]

    // Initial pose
    const double x0 = 0.5;
    const double y0 = 0.0;
    const double z0 = 0.4;

    const double amplitude = 0.3; // [m]
    const double frequency = 0.1; // [Hz]

    // Fixed orientation: "end-effector pointing down"
    const double qx = 1.0;
    const double qy = 0.0;
    const double qz = 0.0;
    const double qw = 0.0;

    // Output paths
    const char* home = std::getenv("HOME");
    if (!home) {
        std::cerr << "ERROR: Cannot determine $HOME directory.\n";
        return 1;
    }
    std::string CSV_OUT = std::string(home) + "/cartesian_test_z.csv"; //TODO: change path if you want

    std::string CSV_OUT_REF = std::string(home) + 
        "/franka_ros2_ws/src/nonlinear_deepc_controller/performance_evaluation/cartesian_ref_z.csv"; //TODO: change path if you want

    std::ofstream file(CSV_OUT);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << CSV_OUT << " for writing.\n";
        return 1;
    }

    std::ofstream file_ref(CSV_OUT_REF);
    if (!file_ref.is_open()) {
        std::cerr << "Failed to open " << CSV_OUT_REF << " for writing.\n";
        return 1;
    }

    // time + position + quaternion
    file << "time_s,x,y,z,qx,qy,qz,qw\n";
    file << std::fixed << std::setprecision(6);
    file_ref << "time_s,x,y,z,qx,qy,qz,qw\n";
    file_ref << std::fixed << std::setprecision(6);

    // Trajectory generation loop
    double t = 0.0; // global time [s]

    // Local sinusoid time (starts AFTER initial_hold)
    double t_motion = 0.0;

    double x = x0, y = y0, z = z0;

    bool in_dwell = false;
    double dwell_elapsed  = 0.0;

    // Peaks of sin(2pi f t_motion) occur at:
    // t_peak_k = (1/(4f)) + k*(1/(2f)), k = 0,1,2,...
    const double half_period = 1.0 / (2.0 * frequency); // T/2
    const double first_peak  = 1.0 / (4.0 * frequency); // T/4
    double next_peak_t_motion = first_peak; // first peak in local motion time

    while (t <= total_time) {

        if (t < initial_hold) {
            // INITIAL HOLD: flat at start pose
            x = x0;
            y = y0;
            z = z0;
        } else {
            // AFTER initial hold: sinusoid + dwell at peaks
            if (!in_dwell) {
                // Motion phase
                if (t_motion >= next_peak_t_motion) {
                    // Enter dwell at current pose (peak)
                    in_dwell = true;
                    dwell_elapsed = 0.0;
                    // Do not advance t_motion while dwelling
                } else {
                    // Normal sinusoidal motion in y TODO: adjust sinusoid axis if needed
                    x = x0; //+ amplitude * std::sin(2.0 * M_PI * frequency * t_motion);
                    y = y0; //+ amplitude * std::sin(2.0 * M_PI * frequency * t_motion);
                    z = z0 + amplitude * std::sin(2.0 * M_PI * frequency * t_motion);

                    t_motion += dt; // advance local motion time only in motion phase
                }
            } else {
                // Dwell phase
                dwell_elapsed += dt;
                // x,y,z remain at last peak pose
                if (dwell_elapsed >= dwell_time) {
                    // Finish dwell -> go back to motion
                    in_dwell = false;
                    dwell_elapsed = 0.0;
                    next_peak_t_motion += half_period;  // schedule next peak
                }
            }
        }

        // Write current sample (pose + fixed orientation)
        // Row string
        std::string row = 
            std::to_string(t) + "," +
            std::to_string(x) + "," +
            std::to_string(y) + "," +
            std::to_string(z) + "," +
            std::to_string(qx) + "," +
            std::to_string(qy) + "," +
            std::to_string(qz) + "," +
            std::to_string(qw) + "\n";

        file << row;
        file_ref << row;

        t += dt; // increment global time
    }

    file.close();
    file_ref.close();
    std::cout << "Trajectory written to " << CSV_OUT << "\n";
    std::cout << "Trajectory written to " << CSV_OUT_REF << "\n";

    return 0;
}
