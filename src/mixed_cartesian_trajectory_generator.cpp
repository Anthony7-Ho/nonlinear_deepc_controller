#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstdlib>

static constexpr double TWO_PI = 6.28318530717958647692;

int main() {
  const double dt = 0.001;
  const double total_time = 75.0;
  const double initial_hold = 5.0;

  // Motion/dwell schedule
  const double motion_block_time = 10.0;
  const double dwell_time = 7.0;

  // Initial pose
  const double x0 = 0.4, y0 = 0.0, z0 = 0.4;

  // Amplitudes
  const double Ax = 0.2;
  const double Ay = 0.2;
  const double Az = 0.2;

  // Same frequency for all axes
  const double f = 0.10;
  const double w = TWO_PI * f;

  // Phase shifts
  const double phix = 0.0;
  const double phiy = 2.0 * TWO_PI / 3.0;
  const double phiz = 4.0 * TWO_PI / 3.0;

  // Fixed orientation
  const double qx = 1.0, qy = 0.0, qz = 0.0, qw = 0.0;

  // Output paths
  const char* home = std::getenv("HOME");
  if (!home) {
    std::cerr << "ERROR: Cannot determine $HOME directory.\n";
    return 1;
  }

  std::string CSV_OUT = std::string(home) + "/cartesian_test_xyz.csv";
  std::string CSV_OUT_REF = std::string(home) +
      "/franka_ros2_ws/src/nonlinear_deepc_controller/performance_evaluation/cartesian_ref_xyz.csv";

  std::ofstream file(CSV_OUT), file_ref(CSV_OUT_REF);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << CSV_OUT << "\n";
    return 1;
  }
  if (!file_ref.is_open()) {
    std::cerr << "Failed to open " << CSV_OUT_REF << "\n";
    return 1;
  }

  file << "time_s,x,y,z,qx,qy,qz,qw\n";
  file_ref << "time_s,x,y,z,qx,qy,qz,qw\n";
  file << std::fixed << std::setprecision(6);
  file_ref << std::fixed << std::setprecision(6);

  double t = 0.0;

  // advances ONLY during motion
  double t_motion = 0.0;

  double motion_elapsed = 0.0;
  double dwell_elapsed = 0.0;
  bool in_dwell = false;

  double x_hold = x0, y_hold = y0, z_hold = z0;

  while (t <= total_time + 1e-12) {
    double x = x0, y = y0, z = z0;

    if (t < initial_hold) {
      // initial hold
      x = x0; y = y0; z = z0;
    } else {
      if (!in_dwell) {
        // Motion phase: one sinusoid per axis, same frequency, phase-shifted
        x = x0 + Ax * std::sin(w * t_motion + phix);
        y = y0 + Ay * std::sin(w * t_motion + phiy);
        z = z0 + Az * std::sin(w * t_motion + phiz);

        // advance motion time
        t_motion += dt;
        motion_elapsed += dt;

        if (motion_elapsed >= motion_block_time) {
          // Start dwell: freeze pose at the last motion sample
          in_dwell = true;
          dwell_elapsed = 0.0;
          motion_elapsed = 0.0;

          x_hold = x;
          y_hold = y;
          z_hold = z;
        }
      } else {
        // Dwell phase: hold pose
        x = x_hold;
        y = y_hold;
        z = z_hold;

        dwell_elapsed += dt;
        if (dwell_elapsed >= dwell_time) {
          in_dwell = false;
          dwell_elapsed = 0.0;
        }
      }
    }

    file << t << "," << x << "," << y << "," << z << "," << qx << "," << qy << "," << qz << "," << qw << "\n";
    file_ref << t << "," << x << "," << y << "," << z << "," << qx << "," << qy << "," << qz << "," << qw << "\n";

    t += dt;
  }

  std::cout << "Trajectory written to " << CSV_OUT << "\n";
  std::cout << "Trajectory written to " << CSV_OUT_REF << "\n";
  return 0;
}
