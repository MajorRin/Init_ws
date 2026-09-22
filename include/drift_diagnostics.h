#pragma once

#include <glog/logging.h>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include "common_lib.h"

// Versioned JSON inside glog. Estimator and batch frames are deliberately
// unaligned; quaternion order is w,x,y,z. Non-finite numbers are JSON null.
namespace drift_diag {
inline double missing() { return std::numeric_limits<double>::quiet_NaN(); }
inline void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}
template<class Derived>
inline void array(std::ostream& out, const Eigen::MatrixBase<Derived>& values) {
    out << '[';
    for (int i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        number(out, values.derived().coeff(i));
    }
    out << ']';
}
inline double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}
} // namespace drift_diag
#include <memory>
#include <atomic>
#include <unistd.h>
#include <ros/ros.h>
#include "Dynamic_init/Dynamic_init.h"
namespace drift_diag {

// One process-wide context, shared by all schema2 channels.
struct Context {
    std::atomic<unsigned long long> sequence{0};
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::string run_id;
    int pass_id = 0, batch_id = 0;
    double handoff_time = missing(), previous_ros = missing();
    bool clock_reversed = false;
    double accumulation_ms = missing(), batch_ms = 0;
    size_t valid_frames = 0, matched_points = 0;
    Context() {
        std::ostringstream s;
        s << std::chrono::system_clock::now().time_since_epoch().count() << '-' << getpid();
        run_id = s.str();
    }
};
inline Context& context() { static Context c; return c; }
inline void prefix(std::ostream& out, const char* channel, double time) {
    auto& c = context();
    out << std::setprecision(17) << '[' << channel << "] {\"schema\":2,\"run_id\":\""
        << c.run_id << "\",\"sequence\":" << ++c.sequence << ",\"steady_ms\":" << elapsed(c.start)
        << ",\"time\":"; number(out, time);
    out << ",\"data_time\":"; number(out, time);
    out << ",\"pass_id\":" << c.pass_id << ",\"batch_id\":" << c.batch_id;
}
struct ClockSample {
    double now, lag_ms = missing();
    bool sim;
    const char* reason;
    explicit ClockSample(double time) : now(ros::Time::now().toSec()), sim(ros::Time::isSimTime()) {
        auto& c = context();
        if (std::isfinite(c.previous_ros) && now < c.previous_ros) c.clock_reversed = true;
        c.previous_ros = now;
        reason = !sim ? "non_sim_time_base_unverified" :
            (!(now > 0) ? "clock_not_ready" : (c.clock_reversed ? "clock_reversed" :
            (!(time > 0) || !std::isfinite(time) ? "invalid_data_time" :
            (now < time ? "negative_lag" : "valid"))));
        if (std::string(reason) == "valid") lag_ms = (now - time) * 1000;
    }
    void write(std::ostream& out) const {
        out << ",\"ros_time\":"; number(out, now);
        out << ",\"lag_ms\":"; number(out, lag_ms);
        out << ",\"lag_s\":"; number(out, lag_ms / 1000);
        out << ",\"clock_mode\":\"" << (sim ? "sim" : "system")
            << "\",\"lag_reason\":\"" << reason
            << "\",\"playback_rate\":null,\"playback_state\":\"unknown\"";
    }
};
inline void snapshot(std::ostream& out, const StatesGroup& s) {
    out << "{\"position\":"; array(out, s.pos_end);
    out << ",\"velocity\":"; array(out, s.vel_end);
    out << ",\"gravity\":"; array(out, s.gravity);
    out << ",\"bias_g\":"; array(out, s.bias_g);
    out << ",\"bias_a\":"; array(out, s.bias_a);
    Eigen::Quaterniond q(s.rot_end);
    out << ",\"quaternion_wxyz\":"; array(out, Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    out << ",\"covariance_diagonal\":"; array(out, s.cov.diagonal());
    out << ",\"covariance_finite\":" << s.cov.diagonal().allFinite()
        << ",\"covariance_nonnegative\":" << (s.cov.diagonal().array() >= 0).all() << '}';
}
struct Frame {
    bool enabled;
    std::string before, predicted;
    std::unique_ptr<StatesGroup> updated;
    bool rtk_found = false;
    double rtk_time = missing();
    Eigen::Vector3d rtk_velocity = Eigen::Vector3d::Constant(missing());
    void update(const StatesGroup& state) {
        if (!enabled) return;
        const auto begin = std::chrono::steady_clock::now();
        updated.reset(new StatesGroup(state));
        diagnostic_ms += elapsed(begin);
    }
    bool emitted = false;
    int cache_index = -1;
    double raw_points = missing(), undistorted_points = missing(), map_points = missing();
    double publish_ms = missing(), build_ms = missing();
    double diagnostic_ms = 0;
    int id;
    double time;
    const char* stage;
    const char* status = "skipped";
    const char* reason = "unprocessed";
    bool applied = false, committed = false, propagated = false;
    double points = missing(), effective = missing(), iterations = missing(), searches = missing();
    double residual = missing(), condition = missing(), noise = missing();
    double frame_ms = missing(), imu_ms = missing(), down_ms = missing();
    double icp_ms = missing(), match_ms = missing(), solve_ms = missing();
    double map_ms = missing();
    Eigen::Matrix<double, 6, 1> eigenvalues;
    Eigen::Matrix<double, 6, 1> information_diagonal;
    Eigen::Vector3d predicted_position;
    Eigen::Matrix3d predicted_rotation;
    bool has_prediction = false;
    std::chrono::steady_clock::time_point start;

    Frame(bool on, const StatesGroup& s, int frame, double stamp, const char* phase)
        : enabled(on), id(frame), time(stamp), stage(phase) {
        if (!enabled) return;
        start = std::chrono::steady_clock::now();
        std::ostringstream out;
        out << std::setprecision(17);
        snapshot(out, s);
        before = out.str();
        eigenvalues.setConstant(missing());
        information_diagonal.setConstant(missing());
        diagnostic_ms += elapsed(start);
    }
    void prediction(const StatesGroup& state) {
        if (!enabled) return;
        const auto begin = std::chrono::steady_clock::now();
        std::ostringstream out;
        out << std::setprecision(17);
        snapshot(out, state);
        predicted = out.str();
        predicted_position = state.pos_end;
        predicted_rotation = state.rot_end;
        has_prediction = true;
        diagnostic_ms += elapsed(begin);
    }
    // Last linearization's weighted 6x6 pose block; mixed rad/m units.
    // This is NOT a marginalized information matrix or an observability proof.
    void information(const Eigen::Matrix<double, 6, 6>& matrix) {
        if (!enabled || !matrix.allFinite()) return;
        information_diagonal = matrix.diagonal();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(matrix);
        if (solver.info() != Eigen::Success) return;
        eigenvalues = solver.eigenvalues();
        if (eigenvalues[5] > 0 && eigenvalues[0] > eigenvalues[5] * 1e-12)
            condition = eigenvalues[5] / eigenvalues[0];
    }
    void finish(const StatesGroup& boundary) {
        if (!enabled || emitted) return;
        if (!updated) update(boundary); // Explicit skipped-frame boundary.
        const StatesGroup& state = *updated;
        emitted = true;
        const auto diag_start = std::chrono::steady_clock::now();
        std::ostringstream out;
        prefix(out, "DLI_FRAME", time);
        out << ",\"frame\":" << id << ",\"cache_index\":" << cache_index
            << ",\"stage\":\"" << stage
            << "\",\"status\":\"" << status << "\",\"reason\":\"" << reason
            << "\",\"applied\":" << applied << ",\"committed\":" << committed
            << ",\"propagated\":" << propagated;
        auto scalar = [&](const char* name, double v) {
            out << ",\"" << name << "\":"; number(out, v);
        };
        scalar("points", points);
        scalar("effective", effective);
        scalar("iterations", iterations);
        scalar("searches", searches);
        ClockSample(time).write(out);
        scalar("relative_time", time - context().handoff_time);
        scalar("effective_ratio", points > 0 ? double(effective) / points : missing());
        scalar("raw_points", raw_points);
        scalar("undistorted_points", undistorted_points);
        scalar("map_points", map_points);
        scalar("publish_ms", publish_ms);
        scalar("build_ms", build_ms);
        out << ",\"before_propagation\":" << before
            << ",\"after_propagation\":" << (predicted.empty() ? "null" : predicted)
            << ",\"after_update\":";
        snapshot(out, state);
        out << ",\"covariance_source\":\"" << (committed ? "committed" :
            (applied ? "partially_applied" : (propagated ? "predicted" : "unchanged")))
            << "\",\"state_frame\":\"estimator_world_unaligned\""
            << ",\"residual_source\":\"last_linearization_mean_absolute_point_plane\""
            << ",\"information_source\":\"last_linearization_pose_block_mixed_rad_m\"";
        scalar("residual", residual);
        scalar("condition", condition);
        scalar("cov_lidar", noise);
        scalar("frame_ms", frame_ms);
        scalar("imu_ms", imu_ms);
        scalar("down_ms", down_ms);
        scalar("icp_ms", icp_ms);
        scalar("match_ms", match_ms);
        scalar("solve_ms", solve_ms);
        scalar("map_ms", map_ms);
        out << ",\"diagnostic_timing_scope\":\"snapshot_and_statistics_excludes_log_write\"";
        Eigen::Vector3d correction = Eigen::Vector3d::Constant(missing());
        if (has_prediction) correction = state.pos_end - predicted_position;
        scalar("rotation_correction_deg", has_prediction
               ? Eigen::AngleAxisd(predicted_rotation.transpose() * state.rot_end).angle()
                 * 180.0 / 3.14159265358979323846 : missing());
        out << ",\"position_correction\":"; array(out, correction);
        out << ",\"position\":"; array(out, state.pos_end);
        out << ",\"velocity\":"; array(out, state.vel_end);
        out << ",\"gravity\":"; array(out, state.gravity);
        out << ",\"bias_g\":"; array(out, state.bias_g);
        out << ",\"bias_a\":"; array(out, state.bias_a);
        Eigen::Quaterniond q(state.rot_end);
        Eigen::Vector4d wxyz(q.w(), q.x(), q.y(), q.z());
        out << ",\"quaternion_wxyz\":"; array(out, wxyz);
        out << ",\"covariance_diagonal\":"; array(out, state.cov.diagonal());
        out << ",\"pose_information_eigenvalues\":"; array(out, eigenvalues);
        out << ",\"pose_information_diagonal\":"; array(out, information_diagonal);
        scalar("speed", state.vel_end.norm());
        scalar("gravity_norm", state.gravity.norm());
        scalar("vertical_velocity", state.gravity.allFinite() && state.gravity.norm() > 0
               ? -state.vel_end.dot(state.gravity.normalized()) : missing());
        scalar("rtk_time", rtk_found ? rtk_time : missing());
        scalar("rtk_time_difference", rtk_found ? rtk_time - time : missing());
        const bool rtk_valid = rtk_found && std::isfinite(rtk_time) && rtk_velocity.allFinite();
        out << ",\"rtk_found\":" << rtk_found << ",\"rtk_valid\":" << rtk_valid
            << ",\"rtk_reason\":\"" << (!rtk_found ? "no_sample" :
                (rtk_valid ? "nearest_finite_no_age_threshold" : "nonfinite"))
            << "\",\"rtk_frame\":\"ENU_unaligned\",\"rtk_velocity\":";
        if (rtk_found) array(out, rtk_velocity); else out << "null";
        scalar("rtk_speed", rtk_valid ? rtk_velocity.norm() : missing());
        out << ",\"rtk_axis_error\":null,\"rtk_alignment_reason\":\"unaligned\"";
        scalar("diag_scope_ms", diagnostic_ms + elapsed(diag_start));
        out << '}';
        LOG(INFO) << out.str();
    }
};
inline void event(const char* name, double time, const char* reason) {
    std::ostringstream out;
    prefix(out, "DLI_EVENT", time);
    out << ",\"event\":\"" << name << "\",\"reason\":\"" << reason << "\"}";
    LOG(INFO) << out.str();
}
inline void heartbeat(bool initialized, size_t lidar, size_t imu, double latest,
                      double processed, double latest_imu) {
    std::ostringstream out;
    prefix(out, "DLI_QUEUE", processed);
    out << ",\"initialized\":" << initialized
        << ",\"lidar_queue\":" << lidar << ",\"imu_queue\":" << imu;
    out << ",\"latest_lidar\":"; number(out, latest);
    out << ",\"processed\":"; number(out, processed);
    out << ",\"latest_imu\":"; number(out, latest_imu);
    ClockSample(processed).write(out);
    out << '}';
    LOG(INFO) << out.str();
}
// Borrowed only while mapping synchronously runs one batch; no estimator writes.
struct Batch : Dynamic_init::SolveObserver {
    Dynamic_init& init;
    Dynamic_init::SolveObserver* previous_observer;
    double time;
    bool with_ba;
    bool finite = true, emitted = false, attached = true;
    int observations = 0;
    Batch(Dynamic_init& value, double stamp, bool estimate_ba)
        : init(value), previous_observer(value.solve_observer), time(stamp), with_ba(estimate_ba) {
        init.solve_diagnostic_ms = 0;
        init.solve_observer = this;
    }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
    void detach() {
        if (attached) init.solve_observer = previous_observer;
        attached = false;
    }
    ~Batch() override { detach(); }
    void window(std::ostream& out) const {
        out << ",\"window_start\":";
        number(out, init.system_state.empty() ? missing() : init.system_state.front().timeStamp);
        out << ",\"window_end\":";
        number(out, init.system_state.empty() ? missing() : init.system_state.back().timeStamp);
        out << ",\"window_size\":" << init.system_state.size()
            << ",\"cache_size\":" << init.Initialized_data.size()
            << ",\"configured_length\":" << init.data_accum_length
            << ",\"estimate_acc_bias\":" << with_ba;
    }
    void velocities(std::ostream& out, const Eigen::VectorXd& x) const {
        out << ",\"velocity_frame\":\"per_sample_local_solver_frame\""
            << ",\"velocity_transform_reason\":\"unaligned_no_axis_comparison\""
            << ",\"velocities\":[";
        for (size_t i = 0; i < init.system_state.size(); ++i) {
            if (i) out << ',';
            out << "{\"cache_index\":" << i << ",\"time\":";
            number(out, init.system_state[i].timeStamp);
            out << ",\"velocity\":";
            if (3 * i + 3 <= size_t(x.size())) array(out, x.segment<3>(3 * i));
            else out << "null";
            out << ",\"R_local_to_batch_column_major\":";
            array(out, init.system_state[i].R);
            out << '}';
        }
        out << ']';
    }
    void solved(const char* stage, int refinement, const Eigen::MatrixXd& A,
                const Eigen::VectorXd& b, const Eigen::VectorXd& x,
                const Eigen::Vector3d& g, Eigen::ComputationInfo decomposition,
                Eigen::ComputationInfo solve) override {
        const bool gyro = std::string(stage) == "gyro_bias";
        const bool valid = A.allFinite() && b.allFinite() && x.allFinite();
        finite = finite && valid && decomposition == Eigen::Success && solve == Eigen::Success;
        ++observations;
        std::ostringstream out;
        prefix(out, "DLI_BATCH", time);
        out << ",\"stage\":\"" << stage << "\",\"refinement\":" << refinement;
        window(out);
        out << ",\"decomposition_status\":" << int(decomposition)
            << ",\"solve_status\":" << int(solve) << ",\"solution_finite\":" << x.allFinite()
            << ",\"system_finite\":" << valid << ",\"normal_equation_relative_residual\":";
        number(out, valid && b.norm() > 0 ? (A * x - b).norm() / b.norm() : missing());
        out << ",\"residual_reason\":\"" << (!valid ? "nonfinite" :
            (b.norm() > 0 ? "valid_numerical_diagnostic_not_observability" : "zero_rhs"))
            << "\",\"solution\":"; array(out, x);
        out << ",\"gravity\":";
        if (gyro) out << "null"; else array(out, g);
        out << ",\"gravity_reason\":\"" << (gyro ? "not_applicable" : "batch_frame") << '"';
        if (!gyro) {
            velocities(out, x);
            out << ",\"gravity_tangent_increment\":";
            const int offset = int(init.system_state.size()) * 3;
            if (refinement >= 0 && x.size() >= offset + 2) array(out, x.segment<2>(offset));
            else out << "null";
        }
        out << '}';
        LOG(INFO) << out.str();
    }
    void finish(const Eigen::VectorXd& x, const Eigen::Vector3d& v,
                const Eigen::Vector3d& g, const Eigen::Vector3d& bg,
                const Eigen::Vector3d& ba, double delta_v, double threshold,
                double wall_ms, double observer_ms, double pass_ms) {
        if (emitted) return; // One-shot: each batch result is reported exactly once.
        emitted = true;
        const auto begin = std::chrono::steady_clock::now();
        finite = finite && v.allFinite() && g.allFinite() && bg.allFinite()
            && ba.allFinite() && x.allFinite();
        std::ostringstream out;
        prefix(out, "DLI_BATCH", time);
        out << ",\"stage\":\"batch_end\",\"refinement\":null";
        window(out);
        out << ",\"initial_velocity\":"; array(out, v);
        out << ",\"initial_velocity_time\":";
        number(out, init.system_state.empty() ? missing() : init.system_state.front().timeStamp);
        out << ",\"initial_velocity_frame\":\"window_start_local_solver_frame\""
            << ",\"initial_speed\":"; number(out, v.norm());
        out << ",\"gravity\":"; array(out, g);
        out << ",\"gravity_norm\":"; number(out, g.norm());
        out << ",\"gravity_frame\":\"batch_frame_unaligned\",\"bias_g\":"; array(out, bg);
        out << ",\"bias_a\":"; array(out, ba);
        out << ",\"bias_a_source\":\"" << (with_ba ? "estimated" : "fixed_not_estimated") << '"';
        out << ",\"delta_v\":"; number(out, delta_v);
        out << ",\"delta_v_reason\":\"" << (std::isfinite(delta_v) ? "adjacent_batch" : "no_previous_solution") << '"';
        out << ",\"velocity_threshold\":"; number(out, threshold);
        out << ",\"finite\":" << finite << ",\"observations\":" << observations;
        out << ",\"quality_reason\":\"" << (finite && observations == 6 ? "finite_same_solve_status_success" : "quality_warning") << '"';
        velocities(out, x);
        out << ",\"batch_wall_ms\":"; number(out, wall_ms);
        out << ",\"batch_algorithm_ms\":"; number(out, wall_ms - observer_ms);
        out << ",\"observer_ms\":"; number(out, observer_ms);
        out << ",\"pass_wall_ms\":"; number(out, pass_ms);
        out << ",\"timing_scope\":\"pass_includes_frames_batch_and_diagnostics\"";
        out << ",\"serialization_ms_excludes_log_write\":"; number(out, elapsed(begin));
        out << '}';
        LOG(INFO) << out.str();
    }
};
}  // namespace drift_diag