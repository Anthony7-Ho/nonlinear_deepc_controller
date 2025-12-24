// include/nonlinear_deepc_controller/logger.hpp
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <initializer_list>

#include <Eigen/Dense>

namespace nonlinear_deepc_controller {

/**
 * CSV logger.
 * Controller defines the schema (header) once.
 * Controller pushes rows at runtime.
 */
class Logger {
public:
  void configure(int decimation, bool stop_at_end, double post_window) {
    decimation_  = std::max(1, decimation);
    stop_at_end_ = stop_at_end;
    post_window_ = std::max(0.0, post_window);
  }

  void setHeader(const std::vector<std::string>& header) {
    header_ = header;
  }

  void reset() {
    if (header_.empty()) {
      throw std::runtime_error("Logger: header not set. Call setHeader(...) before reset().");
    }
    log_counter_ = 0;
    active_ = true;
    rows_.clear();
  }

  // Gate logging based on elapsed time
  void updateActiveWindow(double elapsed_time, double t_end) {
    if (stop_at_end_ && elapsed_time > t_end + post_window_) {
      active_ = false;
    }
  }

  bool active() const { return active_; }

  // Push a full row
  void pushRow(const std::vector<double>& row) {
    if (!active_) return;

    if (++log_counter_ < decimation_) return;
    log_counter_ = 0;

    if (row.size() != header_.size()) {
      throw std::runtime_error(
        "Logger: row size (" + std::to_string(row.size()) +
        ") does not match header size (" + std::to_string(header_.size()) + ")."
      );
    }

    rows_.push_back(row);
  }

  // Helper class to build a row
  class RowBuilder {
  public:
    explicit RowBuilder(size_t expected_cols) : expected_(expected_cols) { vals_.reserve(expected_cols); }

    RowBuilder& add(double v) {
      vals_.push_back(v);
      return *this;
    }

    template <typename Derived>
    RowBuilder& addEigenVector(const Eigen::MatrixBase<Derived>& v) {
      // Accepts Eigen vectors/matrices; appends all entries in column-major order of v
      for (int i = 0; i < v.size(); ++i) vals_.push_back(v(i));
      return *this;
    }

    const std::vector<double>& vec() const { return vals_; }

    void checkSizeOrThrow() const {
      if (vals_.size() != expected_) {
        throw std::runtime_error("Logger::RowBuilder: built row size (" + std::to_string(vals_.size()) + 
        ") does not match expected (" + std::to_string(expected_) + ")."
        );
      }
    }

  private:
    size_t expected_;
    std::vector<double> vals_;
  };

  RowBuilder makeRow() const {
    return RowBuilder(header_.size());
  }

  // Write to CSV
  void write(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("Logger: cannot open log file: " + path);

    // Header
    for (size_t i = 0; i < header_.size(); ++i) {
      out << header_[i];
      if (i + 1 < header_.size()) out << ",";
    }
    out << "\n";

    out << std::fixed << std::setprecision(6);

    // Rows
    for (const auto& r : rows_) {
      for (size_t i = 0; i < r.size(); ++i) {
        out << r[i];
        if (i + 1 < r.size()) out << ",";
      }
      out << "\n";
    }
  }

  // ============ small helpers to build common headers ==========
  static std::vector<std::string> cartesian_header() {
    std::vector<std::string> h;
    h.reserve(1 + 7 + 7 + 7 + 3);
    h.push_back("time_s");
    for (int j = 0; j < 7; ++j) h.push_back("tau_ext_" + std::to_string(j));
    for (int j = 0; j < 7; ++j) h.push_back("friction_pred_" + std::to_string(j));
    for (int j = 0; j < 7; ++j) h.push_back("tau_residual_" + std::to_string(j));
    h.push_back("x"); h.push_back("y"); h.push_back("z");
    return h;
  }

  static std::vector<std::string> joint_header(
      const std::string& tau_cmd_prefix = "tau_cmd_",
      const std::string& tau_ext_prefix = "tau_ext_",
      const std::string& q_prefix = "q_state_",
      const std::string& dq_prefix = "dq_state_") {
    std::vector<std::string> h;
    h.reserve(1 + 7 + 7 + 7 + 7);
    h.push_back("time_s");
    for (int j = 0; j < 7; ++j) h.push_back(tau_cmd_prefix + std::to_string(j));
    for (int j = 0; j < 7; ++j) h.push_back(tau_ext_prefix + std::to_string(j));
    for (int j = 0; j < 7; ++j) h.push_back(q_prefix + std::to_string(j));
    for (int j = 0; j < 7; ++j) h.push_back(dq_prefix + std::to_string(j));
    return h;
  }

private:
  int decimation_{1};
  int log_counter_{0};
  bool stop_at_end_{true};
  double post_window_{0.0};
  bool active_{true};

  std::vector<std::string> header_;
  std::vector<std::vector<double>> rows_;
};

}  // namespace nonlinear_deepc_controller

