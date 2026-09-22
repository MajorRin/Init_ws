// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//#include "ros/package.h"
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include "IMU_Processing.hpp"
#include "drift_diagnostics.h"
#include "ros/package.h"
#include <unistd.h>
#include <Python.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <algorithm>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <livox_ros_driver/CustomMsg.h>
#include <geometry_msgs/Vector3Stamped.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <Dynamic_init/Dynamic_init.h>

#include <math.h>
#include <cstdlib>   // 初始化对比实验: std::getenv
#ifndef DEPLOY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

double ACC_N = 0.01;
double ACC_W = 1e-4;
double GYR_N = 0.01;
double GYR_W = 1e-4;
Eigen::Vector3d G{0.0, 0.0, 9.8};

V3D mean_acc = Zero3d;

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
bool Iteration_begin = false;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, \
 effect_feat_num = 0, scan_count = 0, publish_count = 0;

int feats_points_size = 0;

int feats_points_full_size = 0;

int NUM_MAX_UNDISTORT = 0;

double res_mean_last = 0.05;
double gyr_cov = 0.1, acc_cov = 0.1, grav_cov = 0.0001, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double last_timestamp_lidar = 0, last_timestamp_imu = 0.0;
double filter_size_surf_min = 0, filter_size_map_min = 0;
double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;

// Time Log Variables
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0;

fstream time_log;
int distort_time = 0;

int lidar_type, pcd_save_interval = -1, pcd_index = 0;
bool lidar_pushed, flg_exit = false, flg_EKF_inited = true, flg_reset = true;
bool imu_en = false;
bool LGO_MODE = true;
bool LO_MODE = true;
bool LIO_MODE = true;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool runtime_pos_log = false, pcd_save_en = false, extrinsic_est_en = true, path_en = true;

// LI-Init Parameters
bool cut_frame = true, data_accum_finished = false, data_accum_start = false, online_calib_finish = false, refine_print = false;
int cut_frame_num = 1, orig_odom_freq = 10, frame_num = 0;
double time_lag_IMU_wtr_lidar = 0.0, move_start_time = 0.0, online_calib_starts_time = 0.0, mean_acc_norm = 9.81;
ofstream time_result;
double cov_lidar = 0.001;

// ================= 数据分析: RTK速度缓存 (geometry_msgs/Vector3Stamped, m/s) =================
mutex mtx_rtk;
deque<pair<double, V3D>> rtk_buf;   // (时间戳, 速度)
string rtk_topic = "/dji_osdk_ros/rtk_velocity";

void rtk_velocity_cbk(const geometry_msgs::Vector3Stamped::ConstPtr &msg_in)
{
    double t = msg_in->header.stamp.toSec();
    if (t <= 0) t = ros::Time::now().toSec();
    V3D v(msg_in->vector.x, msg_in->vector.y, msg_in->vector.z);
    std::lock_guard<mutex> lock(mtx_rtk);
    rtk_buf.push_back(make_pair(t, v));
    if (rtk_buf.size() > 20000) rtk_buf.pop_front();
}

// 取与指定时刻最近的RTK速度样本, 无数据返回false
bool get_rtk_at(double t_query, double &rtk_t, V3D &rtk_v)
{
    std::lock_guard<mutex> lock(mtx_rtk);
    if (rtk_buf.empty()) return false;
    size_t best_i = 0;
    double best_dt = fabs(rtk_buf.front().first - t_query);
    for (size_t i = 1; i < rtk_buf.size(); i++)
    {
        double dt = fabs(rtk_buf[i].first - t_query);
        if (dt < best_dt) { best_dt = dt; best_i = i; }
    }
    rtk_t = rtk_buf[best_i].first;
    rtk_v = rtk_buf[best_i].second;
    return true;
}

/// 前向传播自适应噪声参数
bool adaptive_cov = false;
// 表达式参数
double K_cov = 100;
double a_cov = 5;
double b_cov = 10;
double boundary = 10.0;
// 维护量和步长
double scale_cov_t = 0.0;
double scale_cov_R = 0.0;

double step_cov = 0.5;

int history_window_size = 5;

deque<double> history_delta_t;
deque<double> history_delta_R;

shared_ptr<ImuProcess> p_imu(new ImuProcess());
StatesGroup state_propagat;

vector<BoxPointType> cub_needrm;
deque<PointCloudXYZI::Ptr> lidar_buffer;
/// Point3D表示的雷达buffer
deque<std::vector<Point3D>> lidar_points_buffer;

deque<double> time_buffer;
deque<sensor_msgs::Imu::Ptr> imu_buffer;
vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
bool point_selected_surf[100000] = {0};
float res_last[100000] = {0.0};
double total_residual;

//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

PointCloudXYZI::Ptr feats_map(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_map_down(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_map_down_fir(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;

/// 全局变量 当前帧的点云Point3D表示
std::vector<Point3D> feats_points;
std::vector<Point3D> feats_points_full;

double current_dt = 0.1;
double current_end_time = 0.1;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
pcl::VoxelGrid<PointType> downSizeMap;
pcl::VoxelGrid<PointType> downSizeMap_fir;

KD_TREE ikdtree;

M3D last_rot(M3D::Zero());
V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D last_odom(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
double Last_g;
double current_g;
// 论文式(21)收敛判据: 相邻两次迭代的初速度向量差范数 ||v0^(i) - v0^(i-1)|| < Iteration_v
V3D Last_v = V3D::Zero();    // 上一轮迭代解出的初速度
V3D current_v = V3D::Zero(); // 本轮迭代解出的初速度

//estimator inputs and output;
MeasureGroup Measures;
StatesGroup state;
StatesGroup last_state;

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr qujibian_map_save(new PointCloudXYZI());
PointCloudXYZI::Ptr yuanshi_map_save(new PointCloudXYZI());
PointCloudXYZI::Ptr xuanzhaun_map_save(new PointCloudXYZI());


pcl::PCDWriter pcd_writer;
string all_points_dir;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
sensor_msgs::Imu IMU_sync;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<Dynamic_init> dynamic_init(new Dynamic_init());


float calc_dist(PointType p1, PointType p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

void calcBodyVar(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &var) {
    float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    float range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
            pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3d direction(pb);
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
            -direction(0), -direction(1), direction(0), 0;
    Eigen::Vector3d base_vector1(1, 1,
                                 -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
            base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    var = direction * range_var * direction.transpose() +
          A * direction_var * A.transpose();
}

void SigHandle(int sig) {
    time_result.close();
    p_pre->log_pre.close();
    if (pcd_save_en && pcd_save_interval < 0){
        all_points_dir = string(root_dir + "/PCD/PCD_all" + string(".pcd"));
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp) {
    V3D rot_ang(Log(state.rot_end));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1), state.pos_end(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1), state.vel_end(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1), state.bias_g(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1), state.bias_a(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1), state.gravity(2)); // Bias_a  
    fprintf(fp, "\r\n");
    fflush(fp);
}


void pointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state.rot_end * (state.offset_R_L_I * p_body + state.offset_T_L_I) + state.pos_end);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->normal_x = pi->normal_x;
    po->normal_y = pi->normal_y;
    po->normal_z = pi->normal_z;
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state.rot_end * (state.offset_R_L_I * p_body + state.offset_T_L_I) + state.pos_end);
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointTypeRGB *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state.rot_end * (state.offset_R_L_I * p_body + state.offset_T_L_I) + state.pos_end);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->r = pi->normal_x;
    po->g = pi->normal_y;
    po->b = pi->normal_z;

    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity * 10000;
}

int points_cache_size = 0;

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
    for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}


BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
int kdtree_delete_counter = 0;
void lasermap_fov_segment()
{
    cub_needrm.clear();

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = state.pos_end;

    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;
    points_cache_collect();
    if(!imu_en){
        if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    }
}

double timediff_imu_wrt_lidar = 0.0;
bool timediff_set_flg = false;

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_WARN("lidar loop back, clear buffer");
        lidar_buffer.clear();
        time_buffer.clear();
        lidar_points_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (abs(last_timestamp_imu - last_timestamp_lidar) > 1.0 && !timediff_set_flg && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_imu_wrt_lidar = last_timestamp_imu - last_timestamp_lidar;
        printf("Self sync IMU and LiDAR, HARD time lag is %.10lf \n \n", timediff_imu_wrt_lidar);
    }

    if (cut_frame && !dynamic_init->dynamic_init_fished)
    {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        deque<std::vector<Point3D>> points_lidar;

        p_pre->process_cut_frame_livox(msg, ptr, timestamp_lidar, points_lidar, cut_frame_num, scan_count);

        while (!ptr.empty() && !timestamp_lidar.empty() && !points_lidar.empty())
        {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));//unit:s
            timestamp_lidar.pop_front();
            lidar_points_buffer.push_back(points_lidar.front());
            points_lidar.pop_front();
        }
    }
    else
    {
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        std::vector<Point3D> points;
        p_pre->process(msg, ptr, points);
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp_lidar);
        lidar_points_buffer.push_back(points);
    }
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear Lidar buffer.");
        lidar_buffer.clear();
        time_buffer.clear();
    }

    last_timestamp_lidar = msg->header.stamp.toSec();
    if (abs(last_timestamp_imu - last_timestamp_lidar) > 1.0 && !timediff_set_flg && !imu_buffer.empty()) {
        timediff_set_flg = true;
        timediff_imu_wrt_lidar = last_timestamp_imu - last_timestamp_lidar;
        printf("Self sync IMU and LiDAR, HARD time lag is %.10lf \n \n", timediff_imu_wrt_lidar);
    }

    if ((lidar_type == VELO || lidar_type == OUSTER || lidar_type == PANDAR || lidar_type == ROBOSENSE) && cut_frame)
    {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        deque<std::vector<Point3D>> pts_lidar;
        p_pre->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, pts_lidar, cut_frame_num, scan_count);
        while (!ptr.empty() && !timestamp_lidar.empty() && !pts_lidar.empty())
        {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));//unit:s
            timestamp_lidar.pop_front();
            lidar_points_buffer.push_back(pts_lidar.front());
            pts_lidar.pop_front();
        }
    }
    else 
    {
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        std::vector<Point3D> points;

        p_pre->process(msg, ptr, points);
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(msg->header.stamp.toSec());
        lidar_points_buffer.push_back(points);
    }
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in, const ros::Publisher &pubIMU_sync) {
    publish_count++;
    mtx_buffer.lock();
    static double IMU_period, time_msg_in, last_time_msg_in;
    static int imu_cnt = 0;
    time_msg_in = msg_in->header.stamp.toSec();

    if (imu_cnt < 100) {
        imu_cnt++;
        mean_acc += (V3D(msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z) -
                    mean_acc) / (imu_cnt);
        if (imu_cnt > 1) {
            IMU_period += (time_msg_in - last_time_msg_in - IMU_period) / (imu_cnt - 1);
        }
        if (imu_cnt == 99) {
            cout << endl << "Acceleration norm  : " << mean_acc.norm() << endl;
            if (IMU_period > 0.01) {
                cout << "IMU data frequency : " << 1 / IMU_period << " Hz" << endl;
                ROS_WARN("IMU data frequency too low. Higher than 150 Hz is recommended.");
            }
            cout << endl;
        }
    }
    last_time_msg_in = time_msg_in;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu) {
        ROS_WARN("IMU loop back, clear IMU buffer.");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/*!
 * @brief 同步消息 当前已经不再检索IMU
 * @param meas
 * @return
 */
double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || lidar_points_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /** push a lidar scan **/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.points = lidar_points_buffer.front();
        meas.lidar_beg_time = time_buffer.front(); //unit:s

        if (meas.lidar->points.size() <= 1)
        {
            ROS_WARN("Too few input point cloud!\n");
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            lidar_points_buffer.pop_front();
            return false;
        }

        meas.lidar_beg_time = time_buffer.front(); //unit:s

        if (lidar_type == L515)
            lidar_end_time = meas.lidar_beg_time;
        else
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //unit:s

        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;
    }


    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /** push imu data, and pop from imu buffer **/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    if(imu_time > lidar_end_time){
        lidar_buffer.pop_front();
        lidar_points_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return false;
    }
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    lidar_points_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

bool sync_packages_only_lidar(MeasureGroup &meas) {
    if (lidar_buffer.empty())
        return false;

    /** push a lidar scan **/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();

        if (meas.lidar->points.size() <= 1) {

            ROS_WARN("Too few input point cloud!\n");
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }

        meas.lidar_beg_time = time_buffer.front(); //unit:s

        if (lidar_type == L515)
            lidar_end_time = meas.lidar_beg_time;
        else
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //unit:s

        lidar_pushed = true;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

PointCloudXYZI point3DtoPCL(std::vector<Point3D> &v_point_temp, int type)
{
    PointCloudXYZI::Ptr p_cloud_temp(new PointCloudXYZI);
    switch (type)
    {
        case RAW:
            for(int i = 0; i < v_point_temp.size(); i++)
            {
                PointType cloud_temp;
                cloud_temp.x = v_point_temp[i].raw_point.x();
                cloud_temp.y = v_point_temp[i].raw_point.y();
                cloud_temp.z = v_point_temp[i].raw_point.z();
                cloud_temp.normal_x = 0;
                cloud_temp.normal_y = 0;
                cloud_temp.normal_z = 0;
                cloud_temp.intensity = v_point_temp[i].intensity;
                cloud_temp.curvature = v_point_temp[i].relative_time;
                p_cloud_temp->points.push_back(cloud_temp);
            }
            break;
        case UNDISTORT:
            for(int i = 0; i < v_point_temp.size(); i++)
            {
                PointType cloud_temp;
                cloud_temp.x = v_point_temp[i].undistort_lidar_point.x();
                cloud_temp.y = v_point_temp[i].undistort_lidar_point.y();
                cloud_temp.z = v_point_temp[i].undistort_lidar_point.z();
                cloud_temp.normal_x = 0;
                cloud_temp.normal_y = 0;
                cloud_temp.normal_z = 0;
                cloud_temp.intensity = v_point_temp[i].intensity;
                cloud_temp.curvature = v_point_temp[i].relative_time;
                p_cloud_temp->points.push_back(cloud_temp);
            }
            break;
        case WORLD:
            for(int i = 0; i < v_point_temp.size(); i++)
            {
                PointType cloud_temp;
                cloud_temp.x = v_point_temp[i].world_point.x();
                cloud_temp.y = v_point_temp[i].world_point.y();
                cloud_temp.z = v_point_temp[i].world_point.z();
                cloud_temp.normal_x = 0;
                cloud_temp.normal_y = 0;
                cloud_temp.normal_z = 0;
                cloud_temp.intensity = v_point_temp[i].intensity;
                cloud_temp.curvature = v_point_temp[i].relative_time;

                p_cloud_temp->points.push_back(cloud_temp);
            }
            break;
        default:
            for(int i = 0; i < v_point_temp.size(); i++)
            {
                PointType cloud_temp;
                cloud_temp.x = v_point_temp[i].raw_point.x();
                cloud_temp.y = v_point_temp[i].raw_point.y();
                cloud_temp.z = v_point_temp[i].raw_point.z();
                cloud_temp.normal_x = 0;
                cloud_temp.normal_y = 0;
                cloud_temp.normal_z = 0;
                cloud_temp.intensity = v_point_temp[i].intensity;
                cloud_temp.curvature = v_point_temp[i].relative_time;

                p_cloud_temp->points.push_back(cloud_temp);
            }
            break;
    }
    return *p_cloud_temp;
}

std::vector<Point3D> PCLtoPoint3D(const PointCloudXYZI::Ptr &pcl_cloud, int type)
{
    std::vector<Point3D> v_point_temp;

    switch (type)
    {
        case RAW:
            for (int i = 0; i < pcl_cloud->points.size(); ++i)
            {
                Point3D point_temp;
                point_temp.raw_point.x() = pcl_cloud->points[i].x;
                point_temp.raw_point.y() = pcl_cloud->points[i].y;
                point_temp.raw_point.z() = pcl_cloud->points[i].z;
                point_temp.intensity = pcl_cloud->points[i].intensity;
                point_temp.relative_time = pcl_cloud->points[i].curvature;
                v_point_temp.push_back(point_temp);
            }
            break;
        case UNDISTORT:
            for (int i = 0; i < pcl_cloud->points.size(); ++i)
            {
                Point3D point_temp;
                point_temp.undistort_lidar_point.x() = pcl_cloud->points[i].x;
                point_temp.undistort_lidar_point.y() = pcl_cloud->points[i].y;
                point_temp.undistort_lidar_point.z() = pcl_cloud->points[i].z;
                point_temp.intensity = pcl_cloud->points[i].intensity;
                point_temp.relative_time = pcl_cloud->points[i].curvature;
                v_point_temp.push_back(point_temp);
            }
            break;
        case WORLD:
            for (int i = 0; i < pcl_cloud->points.size(); ++i)
            {
                Point3D point_temp;
                point_temp.world_point.x() = pcl_cloud->points[i].x;
                point_temp.world_point.y() = pcl_cloud->points[i].y;
                point_temp.world_point.z() = pcl_cloud->points[i].z;
                point_temp.intensity = pcl_cloud->points[i].intensity;
                point_temp.relative_time = pcl_cloud->points[i].curvature;
                v_point_temp.push_back(point_temp);
            }
            break;
        default:
            for (int i = 0; i < pcl_cloud->points.size(); ++i)
            {
                Point3D point_temp;
                point_temp.raw_point.x() = pcl_cloud->points[i].x;
                point_temp.raw_point.y() = pcl_cloud->points[i].y;
                point_temp.raw_point.z() = pcl_cloud->points[i].z;
                point_temp.intensity = pcl_cloud->points[i].intensity;
                point_temp.relative_time = pcl_cloud->points[i].curvature;
                v_point_temp.push_back(point_temp);
            }
            break;
    }

    return v_point_temp;
}

PointType point3DtoPCLPoint(Point3D& point_3d, int type)
{
    PointType point_temp;
    switch (type)
    {
        case RAW:
            point_temp.x = point_3d.raw_point.x();
            point_temp.y = point_3d.raw_point.y();
            point_temp.z = point_3d.raw_point.z();
            break;
        case UNDISTORT:
            point_temp.x = point_3d.undistort_lidar_point.x();
            point_temp.y = point_3d.undistort_lidar_point.y();
            point_temp.z = point_3d.undistort_lidar_point.z();
            break;
        case WORLD:
            point_temp.x = point_3d.world_point.x();
            point_temp.y = point_3d.world_point.y();
            point_temp.z = point_3d.world_point.z();
            break;
        default:
            point_temp.x = point_3d.raw_point.x();
            point_temp.y = point_3d.raw_point.y();
            point_temp.z = point_3d.raw_point.z();
            break;
    }
    point_temp.intensity = point_3d.intensity;
    return point_temp;
}

bool log_time = false;

/*!
 * @brief 去畸变 投影世界坐标系
 * @param current_point
 */
void iterCurrentPoint3D(Point3D& current_point)
{
    Sophus::SE3d T_begin(last_state.rot_end, last_state.pos_end);
    Sophus::SE3d T_end(state.rot_end, state.pos_end);
    double offset_time = current_dt - (current_end_time - current_point.relative_time) / 1000.0;
    double scale = offset_time / current_dt;
    scale = scale >= 1 ? 1 : scale <= 0 ? 0 : scale;
//    if (log_time)
//        f << "offset_time = " << offset_time << " current_dt = " << current_dt << " " << "current_end_time = "  << current_end_time << " relative_time = " <<  current_point.relative_time << " scale = " << scale << endl;

//     T_wj = inter(T_wb, T_we)   T_ej = T_ew * T_wj   P_e = T_ej * P_j  P_w = T_wj * P_j
//    Sophus::SE3d T_j = Sophus::interpolate(T_begin, T_end, scale);
//    Sophus::SE3d T_ej = T_end.inverse() * T_wj;
//    current_point.undistort_lidar_point = T_ej * current_point.raw_point;
//    current_point.world_point = T_j * current_point.raw_point;
}

/*!
 * @brief 用于在迭代中去畸变 求解世界坐标
 * @param current_point
 * @param current_converge
 */
void iterCurrentPoint3D(Point3D& current_point, int iter_num)
{
    Sophus::SE3d T_begin(last_state.rot_end, last_state.pos_end);
    Sophus::SE3d T_end(state.rot_end, state.pos_end);

    if (iter_num == 0  || iter_num == 18  )
    {
        double offset_time = current_dt - (current_end_time - current_point.relative_time) / 1000.0;
        double scale = offset_time / current_dt;
        scale = scale >= 1 ? 1 : scale <= 0 ? 0 : scale;
        // T_wj = inter(T_wb, T_we)   T_ej = T_ew * T_wj   P_e = T_ej * P_j  P_w = T_wj * P_j
        Sophus::SE3d T_j = Sophus::interpolate(T_begin, T_end, scale);
        Sophus::SE3d T_ej = T_end.inverse() * T_j;
        Sophus::SE3d T_ej_without_rotation(Eigen::Matrix3d::Identity(), T_ej.translation());
        current_point.undistort_lidar_point = T_ej_without_rotation * current_point.raw_point;
        current_point.world_point = T_j * current_point.raw_point;
    }
    else
    {
        current_point.world_point = T_end * current_point.undistort_lidar_point;
    }
}


/*!
 * @brief 用于在迭代中去畸变 求解世界坐标
 * @param current_point
 * @param flag
 */
void iterCurrentPoint3D(Point3D& current_point, bool converge)
{
    Sophus::SE3d T_begin(last_state.rot_end, last_state.pos_end);
    Sophus::SE3d T_end(state.rot_end, state.pos_end);
    /// 滤波器每收敛一次 去一次畸变 去畸变会更新雷达坐标系下的点undistort_lidar_point
    /// 滤波器未收敛时 只需要根据当前的undistort_lidar_point 根据最新的state 维护最新的world_point即可
    if (converge)
    {
        double offset_time = current_dt - (current_end_time - current_point.relative_time) / 1000.0;
        double scale = offset_time / current_dt;
        scale = scale >= 1 ? 1 : scale <= 0 ? 0 : scale;
        // T_wj = inter(T_wb, T_we)   T_ej = T_ew * T_wj   P_e = T_ej * P_j  P_w = T_wj * P_j
        Sophus::SE3d T_j = Sophus::interpolate(T_begin, T_end, scale);  // 根据当前时间，插值得到当前时间的变换矩阵T_j
        Sophus::SE3d T_ej = T_end.inverse() * T_j;  // 计算当前时间点到帧末时间点的变换矩阵T_ej
        if(LO_MODE){
            current_point.undistort_lidar_point = T_ej * current_point.raw_point;
        }else{
            if(!imu_en){
                Sophus::SE3d T_ej_without_rotation(Eigen::Matrix3d::Identity(), T_ej.translation());
                current_point.undistort_lidar_point = T_ej_without_rotation * current_point.raw_point;
            }else{
                Sophus::SE3d T_ej_without_rotation_tran(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
                current_point.undistort_lidar_point = T_ej_without_rotation_tran * current_point.raw_point;
            }
        }
        current_point.world_point = T_j * current_point.raw_point;
    }
    else
    {
        current_point.world_point = T_end * current_point.undistort_lidar_point;
    }
}

int process_increments = 0;

void map_incremental_imu() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_points_size);
    PointNoNeedDownsample.reserve(feats_points_size);
    for (int i = 0; i < feats_points_size; i++) {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        } else {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

void map_incremental()
{
    /// 地图更新
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_points_size);
    PointNoNeedDownsample.reserve(feats_points_size);
    for (int i = 0; i < feats_points_size; i++)
    {
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(point3DtoPCLPoint(feats_points[i], WORLD).x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(point3DtoPCLPoint(feats_points[i], WORLD).y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(point3DtoPCLPoint(feats_points[i], WORLD).z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            float dist = calc_dist(point3DtoPCLPoint(feats_points[i], WORLD), mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(point3DtoPCLPoint(feats_points[i], WORLD));
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(point3DtoPCLPoint(feats_points[i], WORLD));
        }
        else
        {
            PointToAdd.push_back(point3DtoPCLPoint(feats_points[i], WORLD));
        }
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes) {
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_down_body : feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            if (lidar_type == L515)
                RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorldRGB->points[i]);
            else
                pointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *feats_map += *laserCloudWorld;
        sensor_msgs::PointCloud2 laserCloudmsg;
        if (lidar_type == L515)
            pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
        else
            pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }


    /**************** save map ****************/
    /* 1. make sure you have enough memories
       2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) {
        boost::filesystem::create_directories(root_dir + "/PCD");
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
        }

        *pcl_wait_save += *laserCloudWorld;
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            all_points_dir = string(root_dir + "/PCD/PCD") + to_string(pcd_index) + string(".pcd");
            cout << "current scan saved to " << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFullRes_body) {
    PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_undistort, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes_body.publish(laserCloudmsg);
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect) {
    PointCloudXYZI::Ptr laserCloudWorld(\
                    new PointCloudXYZI(effect_feat_num, 1));
    for (int i = 0; i < effect_feat_num; i++) {
        pointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap) {
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T &out)
{
    if (!imu_en) {
        out.position.x = state.pos_end(0);
        out.position.y = state.pos_end(1);
        out.position.z = state.pos_end(2);
    } else {
        //Pubulish LiDAR's pose and position
        V3D pos_cur_lidar = state.rot_end * state.offset_T_L_I + state.pos_end;
        out.position.x = pos_cur_lidar(0);
        out.position.y = pos_cur_lidar(1);
        out.position.z = pos_cur_lidar(2);
    }
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped"));
}

void publish_mavros(const ros::Publisher &mavros_pose_publisher) {
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);

    msg_body_pose.header.frame_id = "camera_odom_frame";
    set_posestamp(msg_body_pose.pose);
    mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath, int measures_num) {
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    if(!Iteration_begin || dynamic_init->dynamic_init_fished == true){
        path.poses.push_back(msg_body_pose);
    }else{
        path.poses[measures_num-1] = msg_body_pose;
    }
    pubPath.publish(path);
}


void print_refine_result() {
    cout.setf(ios::fixed);
    cout << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << setprecision(6)
         << "Rotation LiDAR to IMU    = " << RotMtoEuler(state.offset_R_L_I).transpose() * 57.3 << " deg" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Translation LiDAR to IMU = " << state.offset_T_L_I.transpose() << " m" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    printf("Time Lag IMU to LiDAR    = %.8lf s \n", time_lag_IMU_wtr_lidar + timediff_imu_wrt_lidar);
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Bias of Gyroscope        = " << state.bias_g.transpose() << " rad/s" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Bias of Accelerometer    = " << state.bias_a.transpose() << " m/s^2" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Gravity in World Frame   = " << state.gravity.transpose() << " m/s^2" << endl;
}

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\033[1A\r");
    printf(BOLDMAGENTA "[Refinement] ");
    if (percentage < 1) {
        printf(BOLDYELLOW "Online Refinement: ");
        printf(YELLOW "%3d%% [%.*s%*s]\n", val, lpad, PBSTR, rpad, "");
        cout << RESET;
    } else {
        printf(BOLDGREEN " Online Refinement ");
        printf(GREEN "%3d%% [%.*s%*s]\n", val, lpad, PBSTR, rpad, "");
        cout << RESET;
    }
    fflush(stdout);
}


void subSampleFrame(std::vector<Point3D>& frame, double size_voxel)
{
    std::tr1::unordered_map<voxel, std::vector<Point3D>, std::hash<voxel>> grid;
    for (int i = 0; i < (int) frame.size(); i++)
    {
        auto kx = static_cast<short>(frame[i].raw_point[0] / size_voxel);
        auto ky = static_cast<short>(frame[i].raw_point[1] / size_voxel);
        auto kz = static_cast<short>(frame[i].raw_point[2] / size_voxel);
        grid[voxel(kx, ky, kz)].push_back(frame[i]);
    }
    frame.resize(0);
    int step = 0;
    for(const auto &n: grid)
    {
        if(n.second.size() > 0)
        {
            frame.push_back(n.second[0]);
            step++;
        }
    }
}

void adjustCVCov()
{
    // 后验状态与先验状态的差值
    auto delta_state = state - state_propagat;

    // 速度和角速度的差值
    double delta_v = delta_state.block<3, 1>(12, 0).norm() ;
    double delta_omega = delta_state.block<3, 1>(15, 0).norm() ;

    double cov_v = K_cov / (1 + exp(-1 * a_cov * delta_v + b_cov)) + 0.01;
    double cov_omega = K_cov / (1 + exp(-1 * a_cov * delta_omega + b_cov)) +  0.01;
    V3D new_cov_v(cov_v, cov_v, cov_v);
    V3D new_cov_omega(cov_omega, cov_omega, cov_omega);

    p_imu->Reforward_propagation_without_imu(last_state, state, new_cov_v, new_cov_omega);

    cout << " current delta_t = " << delta_v << " scale = " << delta_v << " acc_cov = " << cov_v << endl;
    cout << " current delta_R = " << delta_omega << " scale = " << delta_omega << " gyr_cov = " << cov_omega << endl;
    cout << endl;

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "D_Li_init");
    ros::NodeHandle nh;
    bool diagnostics_enabled = false;
    std::string diagnostics_dir;
    nh.param<bool>("diagnostics/enabled", diagnostics_enabled, false);
    nh.param<std::string>("diagnostics/glog_dir", diagnostics_dir, "/tmp");
    if (diagnostics_enabled) {
        if (access(diagnostics_dir.c_str(), W_OK | X_OK) != 0) {
            ROS_FATAL("Diagnostic directory is not writable: %s", diagnostics_dir.c_str());
            return 1;
        }
        FLAGS_log_dir = diagnostics_dir;
        FLAGS_logbufsecs = 1;
        FLAGS_max_log_size = 256;
        FLAGS_stop_logging_if_full_disk = true;
        google::InitGoogleLogging(argv[0]);
        drift_diag::event("startup", 0, "world_IMU_pose_wxyz_cov_order_R_P_RLI_TLI_V_BG_BA_G");
    }
    double diagnostics_processed = 0;
    auto diagnostics_heartbeat = std::chrono::steady_clock::now();

    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 10);
    nh.param<int>("mapping/max_undistort", NUM_MAX_UNDISTORT, 3);
    nh.param<bool>("adaptive_cov/use", adaptive_cov, false);
    nh.param<double>("adaptive_cov/K", K_cov, 1);
    nh.param<double>("adaptive_cov/a", a_cov, 1.5);
    nh.param<double>("adaptive_cov/b", b_cov, 5.0);
    nh.param<int>("initialization/cut_frame_init_num", p_pre->cut_frame_init_num, 5);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 2);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<double>("mapping/filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("mapping/filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("mapping/cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 1.0);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<bool>("preprocess/feature_extract_en", p_pre->feature_enabled, 0);
    nh.param<bool>("initialization/cut_frame", cut_frame, true);
    nh.param<int>("initialization/cut_frame_num", cut_frame_num, 1);
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, 1);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, 1);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, 1);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<double>("mapping/cov_lidar", cov_lidar, 0.001);
    nh.param<double>("initialization/mean_acc_norm", p_imu->IMU_mean_acc_norm, 1.0);
    nh.param<double>("initialization/mean_acc_norm", dynamic_init->mean_acc_norm, 1.0);
    
    cout << "lidar_type: " << lidar_type << endl;
    cout << "LiDAR-only odometry starts." << endl;

    path.header.stamp = ros::Time().fromSec(lidar_end_time);
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    VD(DIM_STATE) solution;
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    V3D rot_add, T_add, vel_add, gyr_add;

    PointType pointOri, pointSel, coeff;

    double deltaT, deltaR;
    bool flg_EKF_converged = 0, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    downSizeMap.setLeafSize(0.1, 0.1, 0.1);
    downSizeMap_fir.setLeafSize(0.1, 0.1, 0.1);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    state.set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);

    last_state = state;

    /*** debug record ***/
    boost::filesystem::create_directories(diagnostic_log_directory());
    boost::filesystem::create_directories(root_dir + "/result");
    ofstream Li_D_init_fastlio;
    Li_D_init_fastlio.open(DEBUG_FILE_DIR("Li_D_init_fastlio.txt"), ios::out);

    ofstream Li_D_init_iter_time;
    Li_D_init_iter_time.open(DEBUG_FILE_DIR("Li_D_init_iter_time.txt"), ios::out);
    Li_D_init_iter_time << "# pass_id solve_ms pass_total_ms delta_g converged" << endl;

    ofstream Li_D_init_lio_time;
    Li_D_init_lio_time.open(DEBUG_FILE_DIR("Li_D_init_lio_time.txt"), ios::out);
    Li_D_init_lio_time << "# time frame_total_ms imu_ms downfilter_ms icp_ms knn_ms solve_ms map_increment_ms t_wall_ms effect_feat_num feats_down_size feats_full_size" << endl;

    ofstream Li_D_init_frame_time;
    Li_D_init_frame_time.open(DEBUG_FILE_DIR("Li_D_init_frame_time.txt"), ios::out);
    Li_D_init_frame_time << "# t t_wall_ms frame_id pass_id measures_num row_type imu_en frame_total_ms"
        " imu_ms downfilter_ms icp_ms knn_ms solve_ms publish_odom_ms map_inc_ms publish_tail_ms"
        " iter_count knn_count rematch_num effect_feat_num feats_down_size feats_full_size"
        " kdtree_size_st kdtree_size_end res_mean_last build_ms" << endl;
    Li_D_init_frame_time << std::fixed << std::setprecision(6);

    ofstream Li_D_init_lag;
    Li_D_init_lag.open(DEBUG_FILE_DIR("Li_D_init_lag.txt"), ios::out);
    Li_D_init_lag << "# t_frame t_sim_now lag_ms pass_id converged" << endl;
    Li_D_init_lag << std::fixed << std::setprecision(6);

    time_result.open(DEBUG_FILE_DIR("time_log.txt"), std::ios::out);
    time_result.precision(9);
    time_result.setf(std::ios::fixed);


    std::string cmp_dir = std::getenv("INIT_CMP_DIR") ? std::getenv("INIT_CMP_DIR")
                        : root_dir + "/Log/init_comparison/default";
    boost::filesystem::create_directories(cmp_dir);
    ofstream cmp_events, cmp_states;
    cmp_events.open(cmp_dir + "/A_events.log", ios::out);
    cmp_states.open(cmp_dir + "/A_states.log", ios::out);
    cmp_events << "# record_type t ..." << endl;
    cmp_events << "# init_done t reason(=converged|max_iter) vel(3) bg(3) grav(3) pos(3) vel_prop(3) bg_prop(3) grav_prop(3)" << endl;
    cmp_events << "# batch_solution t_first V0(3) V0_imu(3) gyro_bias(3) Grav_L0(3) grav_imu(3)" << endl;
    cmp_events << "# first_post t vel(3) bg(3) grav(3)" << endl;
    cmp_events << std::fixed << std::setprecision(9);
    cmp_states << "# t_state record_type applied cov_committed propagated flg_EKF_inited imu_en effect_feat_num vel(3) bg(3) grav(3)" << endl;
    cmp_states << std::fixed << std::setprecision(9);
    bool cmp_applied = false, cmp_cov_committed = false, cmp_propagated = false;
    bool cmp_first_post_written = false;

    if (Li_D_init_fastlio && time_result)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;


    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
         nh.subscribe(lid_topic, 200000, standard_pcl_cbk);

    ros::Publisher pubIMU_sync = nh.advertise<sensor_msgs::Imu>
            ("/livox/imu/async_lidar_init", 100000);
    ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>
            (imu_topic, 200000, boost::bind(&imu_cbk, _1, pubIMU_sync));

    // 数据分析: 订阅RTK速度 (geometry_msgs/Vector3Stamped, m/s), 随速度日志一同记录
    nh.param<string>("common/rtk_vel_topic", rtk_topic, "/dji_osdk_ros/rtk_velocity");
    ros::Subscriber sub_rtk = nh.subscribe<geometry_msgs::Vector3Stamped>
            (rtk_topic, 200000, rtk_velocity_cbk);


    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/lidar_init_cloud_registered", 100000);
    ros::Publisher pubLaserCloudFullRes_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/lidar_init_cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/lidar_init_cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/lidar_init_Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/lidar_init_aft_mapped_to_init", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>
            ("/lidar_init_path", 100000);
    int Iteration_NUM;
    double Iteration;
    // 论文式(21): 初速度变化量阈值 Γv (m/s), 收敛判据 ||v0^(i)-v0^(i-1)|| < Iteration_v
    double Iteration_v = 0.1;
    double d_init_solve_time = 0;
    double sum_d_init_solve_time = 0;
    // 数据分析标记列: 0=动态初始化完成前, 1=初始化完成时刻(写入V_0初值行), 2=初始化完成后
    int init_frame_flag = 0;
    // 数据分析: 动态初始化迭代计时
    std::chrono::steady_clock::time_point t_pass_begin = std::chrono::steady_clock::now();
    int pass_id = 0;
    nh.getParam("/Iteration_NUM", Iteration_NUM);
    nh.getParam("/Data_accum_length", dynamic_init->data_accum_length);
    nh.getParam("/Iteration", Iteration);
    nh.getParam("/Iteration_v", Iteration_v);
    nh.getParam("/LGO_MODE", LGO_MODE);
    nh.getParam("/LO_MODE", LO_MODE);
    nh.getParam("/LIO_MODE", LIO_MODE);
    p_imu->LGO_MODE = LGO_MODE;
    p_imu->LO_MODE = LO_MODE;
    p_imu->LIO_MODE = LIO_MODE;
    // 如果 LIO = 0，则认为初始化完成，直接进入 LIO 模式
    if(LIO_MODE){
        dynamic_init->dynamic_init_fished = true;
        dynamic_init->Data_processing_fished = true;
        p_imu->Dynamic_init_done = false;
        imu_en = true;
        p_imu->imu_en = imu_en;
        init_frame_flag = 2;  // LIO_MODE直接跳过动态初始化,无初始化时刻标记
    }
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    // Diagnostic replay must keep draining after rosbag stops publishing /clock.
    ros::WallRate diagnostics_rate(5000);
    bool status = ros::ok();
    int frame_id = 0;
    int iteration_num = 0;
    int measures_num = -1;
    bool data_alignment = 0;
    std::chrono::steady_clock::time_point t_program_begin = std::chrono::steady_clock::now();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if (diagnostics_enabled && drift_diag::elapsed(diagnostics_heartbeat) >= 1000) {
            drift_diag::heartbeat(dynamic_init->dynamic_init_fished, lidar_buffer.size(),
                                  imu_buffer.size(), last_timestamp_lidar,
                                  diagnostics_processed, last_timestamp_imu);
            diagnostics_heartbeat = std::chrono::steady_clock::now();
        }
        if(!LGO_MODE){
            if(!dynamic_init->Data_processing_fished){
                if(sync_packages(Measures)){
                    if (flg_reset)
                    {
                        ROS_WARN("reset when rosbag play back.");
                        p_imu->Reset();
                        flg_reset = false;
                        continue;
                    }
                    if(dynamic_init->Data_processing(Measures))
                    {
                        dynamic_init->Data_processing_fished = true;
                        dynamic_init->lidar_frame_count = 0;
                        t_pass_begin = std::chrono::steady_clock::now();
                    }else{
                        continue;
                    }
                }
            }
            if(dynamic_init->Data_processing_fished && !dynamic_init->dynamic_init_fished){
                measures_num = measures_num + 1;
                if( measures_num == dynamic_init->data_accum_length)
                {
                    measures_num = 0;
                    t_pass_begin = std::chrono::steady_clock::now();
                }
                Measures = dynamic_init->Initialized_data[measures_num];
                data_alignment = 1;
            } else {
                data_alignment = sync_packages(Measures); 
            }
        } else {
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back.");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }
            data_alignment = sync_packages(Measures); 
        }        
        if(data_alignment) 
        {
            lidar_end_time = Measures.lidar_end_time;
            frame_id++;
            if (diagnostics_enabled) drift_diag::context().pass_id = pass_id;
            drift_diag::Frame diagnostic(diagnostics_enabled,
                                         state, frame_id, lidar_end_time,
                                         LIO_MODE ? "direct_lio" : (dynamic_init->dynamic_init_fished
                                         ? "post_dynamic" : (Iteration_begin ? "cache_reprocess" : "initial_lgo")));
            if (diagnostic.enabled) {
                diagnostics_processed = lidar_end_time;
                diagnostic.cache_index = dynamic_init->dynamic_init_fished ? -1 : measures_num;
                diagnostic.raw_points = Measures.lidar ? Measures.lidar->size() : 0;
            }
            data_alignment = 0;
            std::chrono::steady_clock::time_point t_imu_process_begin = std::chrono::steady_clock::now();

            /// 获得当前帧的Point3D点
            feats_points = Measures.points;
            /// 未下采样的Point3D点
            feats_points_full = Measures.points;

            // if(frame_id < dynamic_init->data_accum_length){
            //     pcl::PointCloud<PointType>::Ptr cloudyuanshi(new pcl::PointCloud<PointType>());
            //     *cloudyuanshi = *Measures.lidar;
            //     cloudyuanshi->height = 1;
            //     cloudyuanshi->width = Measures.lidar->size();
            //     std::string filenameyuanshi = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/yuanshi/yuanshi"+ std::to_string(frame_id)+ ".pcd";
            //     pcl::io::savePCDFile(filenameyuanshi, *(cloudyuanshi));
            // }

            // 前向传播与点云畸变校正
            // 初始化对比实验: 记录本帧传播/应用/协方差提交标志, Process 前复位
            cmp_applied = false;
            cmp_cov_committed = false;
            cmp_propagated = false;
            p_imu->Process(Measures, state, feats_undistort, dynamic_init->get_acc_bias(), dynamic_init->get_gyro_bias());
            cmp_propagated = p_imu->last_propagated;
            diagnostic.propagated = cmp_propagated;
            diagnostic.prediction(state);
            if (diagnostic.enabled) {
                diagnostic.undistorted_points = feats_undistort ? feats_undistort->size() : 0;
                diagnostic.imu_ms = drift_diag::elapsed(t_imu_process_begin);
            }
            diagnostic.reason = "empty_cloud";

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                ROS_WARN("First frame, no points stored.");
                if(!dynamic_init->dynamic_init_fished){
                    std::chrono::steady_clock::time_point t_frame_skip = std::chrono::steady_clock::now();
                    Li_D_init_frame_time << lidar_end_time << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_program_begin).count() * 1000 << " "
                        << frame_id << " " << pass_id << " " << measures_num << " " << 2 << " " << (int)imu_en << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_imu_process_begin).count() * 1000 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_imu_process_begin).count() * 1000 << " "
                        << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                        << -1 << " " << -1 << " " << -1 << " "
                        << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                        << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
                }
                diagnostic.finish(state);
                continue;
            }
            // if(frame_id < dynamic_init->data_accum_length){
            //     pcl::PointCloud<PointType>::Ptr cloudshowqujibian(new pcl::PointCloud<PointType>());
            //     *cloudshowqujibian = (*feats_undistort);
            //     cloudshowqujibian->height = 1;
            //     cloudshowqujibian->width = feats_undistort->size();
            //     std::string filenamequjibian = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/qujibian/qujibian"+ std::to_string(frame_id)+ ".pcd";
            //     pcl::io::savePCDFile(filenamequjibian, *(cloudshowqujibian));
            // }
            
            // 将畸变校正后的点云转换为Point3D格式
            feats_points = PCLtoPoint3D(feats_undistort, RAW);
            current_dt = p_imu->frame_dt;   // p_imu->frame_dt; 为当前帧雷达数据的时间间隔
            current_end_time = p_imu->frame_end_time;

            state_propagat = state;

            std::chrono::steady_clock::time_point t_imu_process_end = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t_downfilter_begin = std::chrono::steady_clock::now();
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
            if(imu_en){
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                feats_points_size = feats_down_body->points.size();
                feats_points_full_size = feats_undistort->points.size();
            }else{
                /// Point3D表达的下采样方式 梅森旋转随机排序 + 哈希体素下采样
                boost::mt19937_64 seed;
                std::shuffle(feats_points.begin(), feats_points.end(), seed);
                subSampleFrame(feats_points, filter_size_surf_min);
                std::shuffle(feats_points.begin(), feats_points.end(), seed);
                feats_points_size = feats_points.size();
                feats_points_full_size = feats_points_full.size();
            }
            std::chrono::steady_clock::time_point t_downfilter_end = std::chrono::steady_clock::now();
            /// 初始化第一帧点使用未下采样的点
            /*** initialize the map kdtree ***/
            diagnostic.reason = "map_seed";
            diagnostic.points = feats_points_size;
            if (ikdtree.Root_Node == nullptr )
            {
                if (feats_points_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_points_full_size);
                    for (int i = 0; i < feats_points_full_size; i++)
                    {
                        // 第一帧不去畸变
                        if(imu_en){
                            pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
                        }else{
                            feats_down_world->points[i] = point3DtoPCLPoint(feats_points_full[i], RAW);
                        }
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                if(!imu_en){
                    if(!Iteration_begin){
                        dynamic_init->Data_processing_lo(state.rot_end, state.pos_end, Measures.lidar_end_time, p_imu->tmp_pre_integration);
                    }else{
                        dynamic_init->system_state[measures_num].R = state.rot_end;
                        dynamic_init->system_state[measures_num].T = state.pos_end;
                    }
                }
                if(!dynamic_init->dynamic_init_fished){
                    std::chrono::steady_clock::time_point t_frame_skip = std::chrono::steady_clock::now();
                    Li_D_init_frame_time << lidar_end_time << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_program_begin).count() * 1000 << " "
                        << frame_id << " " << pass_id << " " << measures_num << " " << 1 << " " << (int)imu_en << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_imu_process_begin).count() * 1000 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_imu_process_end - t_imu_process_begin).count() * 1000 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_downfilter_end - t_downfilter_begin).count() * 1000 << " "
                        << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                        << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                        << feats_points_size << " " << feats_points_full_size << " "
                        << -1 << " " << -1 << " " << -1 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(t_frame_skip - t_downfilter_end).count() * 1000 << endl;
                }
                diagnostic.finish(state);
                continue;
            }
            int featsFromMapNum = ikdtree.validnum(); // 获取ikdtree中有效点数（地图点数）
            kdtree_size_st = ikdtree.size(); // 获取ikdtree总大小（含被删节点）
            /*** ICP and iterated Kalman filter update ***/
            normvec->resize(feats_points_size); // 法向量点云调整大小=降采样后点数
            feats_down_world->resize(feats_points_size); // 世界系下点云调整大小=降采样后点数
            euler_cur = RotMtoEuler(state.rot_end); // 当前旋转矩阵→欧拉角，用于后续发布

            pointSearchInd_surf.resize(feats_points_size); // 最近邻索引数组调整大小
            Nearest_Points.resize(feats_points_size); // 最近邻点集数组调整大小
            int rematch_num = 0; // 重匹配次数计数器（收敛后重新搜索近邻的次数）
            bool nearest_search_en = true; // 是否启用最近邻搜索标志，首轮必须搜索

            /*** iterated state estimation ***/
            std::vector<M3D> body_var; // 预分配：每个点在体坐标系下的协方差矩阵
            std::vector<M3D> crossmat_list; // 预分配：每个点的反对称矩阵列表
            body_var.reserve(feats_points_size); // 预留空间避免重复分配
            crossmat_list.reserve(feats_points_size); // 预留空间避免重复分配

            std::chrono::steady_clock::time_point icp_begin = std::chrono::steady_clock::now(); // ICP总计时起点

            double knn_time = 0; // KNN搜索累计耗时（秒）
            double solve_time = 0; // 状态求解累计耗时（秒）

            int knn_count = 0; // KNN搜索次数计数
            int iter_count = 0; // 总迭代次数计数
            StatesGroup last_converge_state = state; // 上一次收敛时的状态（用于LGO全局收敛判断）

            bool current_converge = true; // 当前迭代是否已收敛（初始true让第一轮distort_time++）
            int current_iter_count = 0; // 当前收敛轮内的迭代计数（用于LGO判断是否该重新去畸变）

            distort_time = 0; // 去畸变次数计数器（LGO模式下限制去畸变-ICP循环次数）

            int lio_converge_count = 0; // LIO模式收敛确认计数

            for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) // 迭代EKF主循环，最多NUM_MAX_ITERATIONS次（默认10）
            {
                laserCloudOri->clear(); // 清空：有效特征点的原始坐标（体坐标系）
                corr_normvect->clear(); // 清空：有效特征点对应的法向量+残差
                total_residual = 0.0; // 清零：总残差

                iter_count++; // 总迭代计数+1

                if (nearest_search_en) // 如果本轮需要重新搜索最近邻
                    knn_count++; // KNN搜索计数+1

                if (current_converge) // 如果上一轮已收敛（需要重新去畸变）
                    distort_time++; // 去畸变次数+1

                std::chrono::steady_clock::time_point knn_begin = std::chrono::steady_clock::now(); // KNN计时起点

                /** closest surface search and residual computation **/
                #ifdef MP_EN // 如果启用了OpenMP多线程
                    omp_set_num_threads(MP_PROC_NUM); // 设置线程数
                    #pragma omp parallel for // 并行化下面的for循环
                #endif
                for (int i = 0; i < feats_points_size; i++) // 遍历每个降采样后的点
                {
                    PointType point_body; // 体坐标系下的点（LiDAR系或IMU系）
                    PointType point_world; // 世界坐标系下的点
                    if(imu_en){ // LIO模式
                        point_body = feats_down_body->points[i]; // 取降采样后体坐标系点
                        point_world = feats_down_world->points[i]; // 取世界系点（上一轮的，会被覆盖）
                        /// transform to world frame
                        pointBodyToWorld(&point_body, &point_world); // 用当前状态将体坐标系点变换到世界系：p_w = R·(R_LI·p_L + T_LI) + T
                    } else { // LGO/LO模式
                        // 去畸变 计算世界坐标系下的点
                        iterCurrentPoint3D(feats_points[i], current_converge); // 用当前状态对Point3D做去畸变+世界变换（迭代中状态更新后需重新计算）
                        // 去畸变后的点
                        point_body = point3DtoPCLPoint(feats_points[i], UNDISTORT); // 取去畸变后的体坐标系点
                        point_world = point3DtoPCLPoint(feats_points[i], WORLD); // 取世界坐标系下的点
                    }
                    V3D p_body(point_body.x, point_body.y, point_body.z); // 转为Eigen向量，用于后续计算
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS); // 最近邻搜索距离平方结果（NUM_MATCH_POINTS=5）
                    auto &points_near = Nearest_Points[i]; // 引用：第i个点的最近邻点集
                    uint8_t search_flag = 0; // 搜索标志（未使用）

                    if (nearest_search_en) // 如果本轮需要重新搜索最近邻
                    {
                        /** Find the closest surfaces in the map **/
                        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 5); // 在ikdtree中搜索5个最近邻点
                        if (points_near.size() < NUM_MATCH_POINTS) // 如果找到的近邻不足5个
                            point_selected_surf[i] = false; // 标记该点无效（不参与配准）
                        else
                            point_selected_surf[i] = !(pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5); // 第5近邻距离>√5则标记无效（点离地图太远）
                    }

                    res_last[i] = -1000.0f; // 重置残差为无效值

                    if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS) // 如果该点无效或近邻不足
                    {
                        point_selected_surf[i] = false; // 确保标记为无效
                        continue; // 跳过该点
                    }

                    point_selected_surf[i] = false; // 先假设无效，下面验证通过再设为true
                    VD(4) pabcd; // 平面方程系数 [a,b,c,d]，ax+by+cz+d=0
                    pabcd.setZero(); // 初始化为零
                    if (esti_plane(pabcd, points_near, 0.1)) // 用5个近邻点拟合局部平面，0.1为平面拟合阈值（SVD最小奇异值比）
                    {
                        float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                                    pabcd(3); // 计算点到平面的有符号距离：pd2 = n·p_w + d
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); // 自适应权重：残差越小权重越大，p_body.norm()是点到原点距离

                        if (s > 0.9) { // 如果权重足够大（残差足够小，说明点是有效的平面特征）
                            point_selected_surf[i] = true; // 标记该点为有效特征点
                            normvec->points[i].x = pabcd(0); // 存储平面法向量x分量
                            normvec->points[i].y = pabcd(1); // 存储平面法向量y分量
                            normvec->points[i].z = pabcd(2); // 存储平面法向量z分量
                            normvec->points[i].intensity = pd2; // 存储点到面残差（有符号距离）
                            res_last[i] = abs(pd2); // 存储绝对残差，用于后续残差监控
                        }
                    }
                }

                std::chrono::steady_clock::time_point knn_end = std::chrono::steady_clock::now(); // KNN计时终点
                knn_time += std::chrono::duration_cast<std::chrono::duration<double> >(knn_end - knn_begin).count(); // 累加KNN耗时

                effect_feat_num = 0; // 有效特征点计数清零
                for (int i = 0; i < feats_points_size; i++) // 遍历所有降采样点
                {
                    if (point_selected_surf[i]) // 如果是有效特征点
                    {
                        if(imu_en){ // LIO模式
                            laserCloudOri->points[effect_feat_num] = feats_down_body->points[i]; // 存入体坐标系下的原始点
                        }else{ // LGO/LO模式
                            laserCloudOri->points[effect_feat_num] = point3DtoPCLPoint(feats_points[i], UNDISTORT); // 存入去畸变后的体坐标系点
                        }
                        corr_normvect->points[effect_feat_num] = normvec->points[i]; // 存入对应的法向量+残差
                        effect_feat_num++; // 有效特征点计数+1
                    }
                }

                res_mean_last = total_residual / effect_feat_num; // 计算平均残差（用于监控收敛）

                /*** Computation of Measurement Jacobian matrix H and measurents vector ***/

                MatrixXd Hsub(effect_feat_num, 12); // 测量Jacobian矩阵：effect_feat_num行×12列（12=[δR,δT,δR_LI,δT_LI]各3维）
                MatrixXd Hsub_T_R_inv(12, effect_feat_num); // H^T·R^{-1}：12行×effect_feat_num列，预计算加速后续矩阵运算
                VectorXd R_inv(effect_feat_num); // 测量噪声逆矩阵对角线（标量×单位阵）
                VectorXd meas_vec(effect_feat_num); // 测量残差向量：每个有效特征点的点到面距离

                Hsub.setZero(); // H矩阵清零
                Hsub_T_R_inv.setZero(); // H^T·R^{-1}清零
                meas_vec.setZero(); // 测量向量清零

                std::chrono::steady_clock::time_point solve_begin = std::chrono::steady_clock::now(); // 求解计时起点

                for (int i = 0; i < effect_feat_num; i++) // 遍历每个有效特征点，构建H和测量向量
                {
                    const PointType &laser_p = laserCloudOri->points[i]; // 取第i个有效点的体坐标系坐标
                    V3D point_this_L(laser_p.x, laser_p.y, laser_p.z); // 转为Eigen向量：LiDAR坐标系下的点p_L

                    V3D point_this = state.offset_R_L_I * point_this_L + state.offset_T_L_I; // 变换到IMU坐标系：p_I = R_L_I·p_L + T_L_I
                    M3D var; // 点在IMU体坐标系下的协方差
                    calcBodyVar(point_this, 0.02, 0.05, var); // 计算体坐标系协方差（基于点到原点距离的模型：近处0.02，远处0.05）
                    var = state.rot_end * var * state.rot_end.transpose(); // 旋转到世界系：var_w = R·var·R^T
                    M3D point_crossmat; // 点的反对称矩阵[p]_×
                    point_crossmat << SKEW_SYM_MATRX(point_this); // 构造反对称矩阵，用于叉积：p×v = [p]_×·v

                    /*** get the normal vector of closest surface/corner ***/
                    const PointType &norm_p = corr_normvect->points[i]; // 取该点对应的法向量+残差
                    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); // 提取法向量n（单位向量，指向平面）

                    R_inv(i) = 1000; // 测量噪声逆=1000，即σ²=0.001（cov_lidar），R^{-1}=1/σ²
                    laserCloudOri->points[i].intensity = sqrt(R_inv(i)); // 存储√(R^{-1})到intensity字段（用于可视化/调试）

                    /*** calculate the Measurement Jacobian matrix H ***/
                    if (imu_en) { // LIO模式：12维状态[δR,δT,δR_LI,δT_LI]
                        M3D point_this_L_cross; // LiDAR系下点的反对称矩阵[p_L]_×
                        point_this_L_cross << SKEW_SYM_MATRX(point_this_L); // 构造[p_L]_×
                        V3D H_R_LI = point_this_L_cross * state.offset_R_L_I.transpose() * state.rot_end.transpose() *
                                     norm_vec; // 对外参旋转R_L_I的Jacobian：∂z/∂δR_LI = [p_L]_×·R_L_I^T·R^T·n
                        V3D H_T_LI = state.rot_end.transpose() * norm_vec; // 对外参平移T_L_I的Jacobian：∂z/∂δT_LI = R^T·n
                        V3D A(point_crossmat * state.rot_end.transpose() * norm_vec); // 对旋转R的Jacobian主项：A = [p_I]_×·R^T·n
                        Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(
                                H_R_LI), VEC_FROM_ARRAY(H_T_LI); // 拼接第i行H：[A, n, H_R_LI, H_T_LI]，共12列
                    } else { // LGO/LO模式：仅6维状态[δR,δT]，外参Jacobian置零
                        V3D A(point_crossmat * state.rot_end.transpose() * norm_vec); // 对旋转R的Jacobian：A = [p_I]_×·R^T·n
                        Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0, 0, 0, 0, 0, 0; // 拼接第i行H：[A, n, 0,0,0,0,0,0]，后6列置零
                    }

                    Hsub_T_R_inv.col(i) = Hsub.row(i).transpose() / cov_lidar; // 预计算H^T·R^{-1}的第i列 = H_i^T / σ²
                    /*** Measurement: distance to the closest surface/corner ***/
                    meas_vec(i) = -norm_p.intensity; // 测量残差 = -(点到面距离)，取负号因为EKF更新公式中z = h(x)-z_meas
                }

                MatrixXd K(DIM_STATE, effect_feat_num); // 卡尔曼增益矩阵：DIM_STATE行×effect_feat_num列

                EKF_stop_flg = false; // 重置EKF停止标志
                flg_EKF_converged = false; // 重置EKF收敛标志
                current_converge = false; // 重置当前收敛标志
                /*** Iterative Kalman Filter Update ***/

                H_T_H.block<12, 12>(0, 0) = Hsub_T_R_inv * Hsub; // 计算 H^T·R^{-1}·H（12×12矩阵），只填充左上12×12块
                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + state.cov.inverse()).inverse(); // K_1 = (H^T·R^{-1}·H + P^{-1})^{-1}，即后验协方差
                K = K_1.block<DIM_STATE, 12>(0, 0) * Hsub_T_R_inv; // 卡尔曼增益 K = P_posterior · H^T · R^{-1}，取K_1左上DIM_STATE×12块
                auto vec = state_propagat - state; // IMU传播状态与当前EKF状态的差值（传播-更新），即"先验-后验"偏差
                solution = K * meas_vec + vec - K * Hsub * vec.block<12, 1>(0, 0); // 状态修正量 = K·z + (x_prop-x) - K·H·(x_prop-x)_前12维
                                                                                   // 第一项：测量更新；第二项：补偿传播偏差；第三项：减去测量已解释的偏差部分

                //state update
                state += solution; // 状态更新：state = state + δx（旋转用右乘更新，平移/速度等加法更新）
                cmp_applied = true; // 初始化对比实验: 本帧滤波解已应用到状态

                std::chrono::steady_clock::time_point solve_end = std::chrono::steady_clock::now(); // 求解计时终点

                solve_time += std::chrono::duration_cast<std::chrono::duration<double> >(solve_end - solve_begin).count(); // 累加求解耗时

                rot_add = solution.block<3, 1>(0, 0); // 提取旋转更新量δR（前3维）
                T_add = solution.block<3, 1>(3, 0); // 提取平移更新量δT（第4-6维）

                if ((rot_add.norm() * 57.3 < 0.057) && (T_add.norm() * 100 < 0.1)) // 收敛判断：旋转增量<0.057°且平移增量<0.1cm
                    flg_EKF_converged = true; // 标记EKF已收敛

                deltaR = rot_add.norm() * 57.3; // 旋转增量转角度（rad→deg），用于日志
                deltaT = T_add.norm() * 100; // 平移增量转厘米（m→cm），用于日志

                euler_cur = RotMtoEuler(state.rot_end); // 更新当前欧拉角

                current_iter_count++; // 当前收敛轮内迭代计数+1
                /*** Rematch Judgement ***/
                nearest_search_en = false; // 默认下一轮不重新搜索近邻（用上一轮的近邻结果）
                if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) // 如果EKF收敛，或者首次且倒数第2轮（保底重匹配）
                {
                    nearest_search_en = true; // 下一轮需要重新搜索最近邻（状态已更新，近邻可能改变）
                    rematch_num++; // 重匹配次数+1
                    if(!imu_en){ // LGO/LO模式特有的全局收敛判断
                        /// 全局收敛判断 两次匹配的收敛状态相差小于阈值 重新进行去畸变
                        auto delta_state = state - last_converge_state; // 当前状态与上次收敛状态的差

                        if (((delta_state.block<3, 1>(0, 0).norm() * 57.3 < 1) && (delta_state.block<3, 1>(3, 0).norm() * 100 < 1.0) && iterCount >= 4)
                                || current_iter_count >= 15) // 如果两次收敛间旋转差<1°且平移差<1cm且迭代≥4次，或迭代超过15次
                        {
                            current_converge = true; // 标记当前轮收敛，下一轮会触发重新去畸变
                            current_iter_count = 0; // 重置收敛轮内迭代计数
                            /// 如果去畸变次数达到预设值，那么也退出
                            if (distort_time >= NUM_MAX_UNDISTORT) // 如果去畸变-ICP循环次数达到上限（默认3次）
                            {
                                EKF_stop_flg = true; // 标记EKF停止（整个迭代退出）
                            }
                        }

                        last_converge_state = state; // 记录本次收敛状态，供下次比较
                    }
                }

                /*** LIO模式收敛确认提前退出 ***/
                if (imu_en) // LIO模式下原代码无退出通道，此处补上：连续2轮收敛即退出，避免空转
                {
                    if (flg_EKF_converged)
                    {
                        lio_converge_count++;
                        if (lio_converge_count >= 2)
                            EKF_stop_flg = true; // 连续第2轮收敛，退出迭代（走正常协方差更新后break）
                    }
                    else
                    {
                        lio_converge_count = 0; // 未收敛则重新计数（与FAST-LIO的t机制一致）
                    }
                }

                /*** Adjust const velocity model cov ***/
                // if (iterCount == 1 && adaptive_cov) adjustCVCov(); // 自适应协方差调整（当前未启用）


                /*** Convergence Judgements and Covariance Update ***/
                /// 不限制匹配次数
                if (EKF_stop_flg || ((iterCount == NUM_MAX_ITERATIONS - 1)) ) // 如果EKF停止或达到最大迭代次数
                {
                    if (flg_EKF_inited) // 如果EKF已初始化（非首帧）
                    {
                        /*** Covariance Update ***/
                        G.setZero(); // G矩阵清零
                        G.block<DIM_STATE, 12>(0, 0) = K * Hsub; // G = K·H（Joseph形式协方差更新的一部分）
                        state.cov = (I_STATE - G) * state.cov; // 协方差更新：P = (I - K·H)·P（简化形式）
                        cmp_cov_committed = true; // 初始化对比实验: 本帧协方差已提交
                        total_distance += (state.pos_end - position_last).norm(); // 累计行驶距离
                        position_last = state.pos_end; // 更新上一帧位置

                        if (!imu_en) { // LGO/LO模式：发布IMU系下的姿态
                            // M3D rot_cur_lidar = state.rot_end * state.offset_R_L_I;
                            // V3D euler_cur_lidar = RotMtoEuler(rot_cur_lidar);
                            // geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                            //         (euler_cur_lidar(0), euler_cur_lidar(1), euler_cur_lidar(2));
                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                    (euler_cur(0), euler_cur(1), euler_cur(2)); // 欧拉角→四元数（IMU系）
                        } else { // LIO模式：发布LiDAR系下的姿态
                            //Publish LiDAR's pose, instead of IMU's pose
                            M3D rot_cur_lidar = state.rot_end * state.offset_R_L_I; // LiDAR旋转 = R_IMU · R_L_I
                            V3D euler_cur_lidar = RotMtoEuler(rot_cur_lidar); // LiDAR欧拉角
                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                    (euler_cur_lidar(0), euler_cur_lidar(1), euler_cur_lidar(2)); // LiDAR欧拉角→四元数
                        }
                        VD(DIM_STATE) K_sum = K.rowwise().sum(); // K矩阵行求和（用于调试/分析卡尔曼增益总量）
                        VD(DIM_STATE) P_diag = state.cov.diagonal(); // 协方差对角线（用于调试/分析状态不确定性）
                    }
                    EKF_stop_flg = true; // 确保停止标志为true
                }

                if (EKF_stop_flg) break; // 如果EKF停止，退出迭代循环
            }

            std::chrono::steady_clock::time_point icp_end = std::chrono::steady_clock::now();
            if(!imu_en){
                *feats_down_body = point3DtoPCL(feats_points, UNDISTORT);
                last_state = state;
                // if(frame_id < dynamic_init->data_accum_length){
                //     pcl::PointCloud<PointType>::Ptr cloudshow(new pcl::PointCloud<PointType>());
                //     *cloudshow = *feats_down_body;
                //     cloudshow->height = 1;
                //     cloudshow->width = feats_down_body->size();
                //     std::string filenamequjibian_fin = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/qujibianfin/qujibianfin"+ std::to_string(frame_id)+ ".pcd";
                //     pcl::io::savePCDFile(filenamequjibian_fin, *(cloudshow));
                //     if (1) {
                //         int size = feats_down_body->points.size();
                //         PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
                //         for (int i = 0; i < size; i++) {
                //         pointBodyToWorld(&feats_down_body->points[i], &laserCloudWorld->points[i]);
                //         }
                //         *qujibian_map_save += *laserCloudWorld;
                //     }
                //     if (1) {
                //         int size = feats_undistort->points.size();
                //         PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
                //         for (int i = 0; i < size; i++) {
                //         pointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
                //         }
                //         *xuanzhaun_map_save += *laserCloudWorld;
                //     }
                //     if (1) {
                //         int size = Measures.lidar->points.size();
                //         PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
                //         for (int i = 0; i < size; i++) {
                //         pointBodyToWorld(&Measures.lidar->points[i], &laserCloudWorld->points[i]);
                //         }
                //         *yuanshi_map_save += *laserCloudWorld;
                //     }
                //     if(frame_id == dynamic_init->data_accum_length - 1){
                //         std::string filenamequjibian_map = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/qujibianfin/map"+ std::to_string(0000000)+ ".pcd";
                //         std::string filenameyuanshi_map = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/yuanshi/yuanshi_map"+ std::to_string(0000000)+ ".pcd";
                //         std::string filenamequxuanzhuan_map = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/qujibian/xuanzhuan_map"+ std::to_string(0000000)+ ".pcd";
                //         pcl::io::savePCDFile(filenamequjibian_map, *(qujibian_map_save));
                //         pcl::io::savePCDFile(filenamequxuanzhuan_map, *(xuanzhaun_map_save));
                //         pcl::io::savePCDFile(filenameyuanshi_map, *(yuanshi_map_save));
                //     }
                // }
            }
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            std::chrono::steady_clock::time_point map_increment_begin = std::chrono::steady_clock::now();

            /*** add the feature points to map kdtree ***/
            if(imu_en){
                map_incremental_imu();
            }else{
                map_incremental();
            }



            kdtree_size_end = ikdtree.size();

            std::chrono::steady_clock::time_point map_increment_end = std::chrono::steady_clock::now();

            /******* Publish points *******/
            if ((scan_pub_en || pcd_save_en) && dynamic_init->dynamic_init_fished == true) publish_frame_world(pubLaserCloudFullRes);
            last_odom = state.pos_end;
            last_rot = state.rot_end;
            publish_effect_world(pubLaserCloudEffect);
            if (path_en) publish_path(pubPath, measures_num);
            std::chrono::steady_clock::time_point frame_end = std::chrono::steady_clock::now();
            if (diagnostic.enabled) {
                diagnostic.applied = cmp_applied;
                diagnostic.committed = cmp_cov_committed;
                diagnostic.effective = iter_count > 0 ? effect_feat_num : 0;
                diagnostic.status = cmp_applied && cmp_cov_committed && diagnostic.effective > 0
                    ? "post" : (cmp_applied ? "partial_applied" : "predict");
                diagnostic.reason = diagnostic.effective > 0 ? "matched" : "no_effective_points";
                diagnostic.iterations = iter_count;
                diagnostic.searches = knn_count;
                // Diagnostic-only mean absolute point-to-plane residual at the last
                // linearization. Do not reuse the legacy, never-accumulated total_residual.
                if (diagnostic.effective > 0) {
                    double absolute_residual_sum = 0.0;
                    for (int i = 0; i < diagnostic.effective; ++i)
                        absolute_residual_sum += std::fabs(corr_normvect->points[i].intensity);
                    diagnostic.residual = absolute_residual_sum / diagnostic.effective;
                }
                diagnostic.noise = cov_lidar;
                auto ms = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                diagnostic.frame_ms = ms(t_imu_process_begin, frame_end);
                diagnostic.imu_ms = ms(t_imu_process_begin, t_imu_process_end);
                diagnostic.down_ms = ms(t_downfilter_begin, t_downfilter_end);
                diagnostic.icp_ms = ms(icp_begin, icp_end);
                diagnostic.map_ms = ms(map_increment_begin, map_increment_end);
                diagnostic.match_ms = knn_time * 1000;
                diagnostic.solve_ms = solve_time * 1000;
                if (diagnostic.effective > 0) diagnostic.information(H_T_H.block<6, 6>(0, 0));
            }
            diagnostic.update(state); // 即时冻结更新后状态, 先于后续批解与重置改写 state

            if (!dynamic_init->dynamic_init_fished)
            {
                double f_imu   = std::chrono::duration_cast<std::chrono::duration<double> >(t_imu_process_end - t_imu_process_begin).count() * 1000;
                double f_down  = std::chrono::duration_cast<std::chrono::duration<double> >(t_downfilter_end - t_downfilter_begin).count() * 1000;
                double f_icp   = std::chrono::duration_cast<std::chrono::duration<double> >(icp_end - icp_begin).count() * 1000;
                double f_pubod = std::chrono::duration_cast<std::chrono::duration<double> >(map_increment_begin - icp_end).count() * 1000;
                double f_map   = std::chrono::duration_cast<std::chrono::duration<double> >(map_increment_end - map_increment_begin).count() * 1000;
                double f_pubtl = std::chrono::duration_cast<std::chrono::duration<double> >(frame_end - map_increment_end).count() * 1000;
                double f_total = std::chrono::duration_cast<std::chrono::duration<double> >(frame_end - t_imu_process_begin).count() * 1000;
                Li_D_init_frame_time << lidar_end_time << " "
                    << std::chrono::duration_cast<std::chrono::duration<double> >(frame_end - t_program_begin).count() * 1000 << " "
                    << frame_id << " " << pass_id << " " << measures_num << " " << 0 << " " << (int)imu_en << " "
                    << f_total << " " << f_imu << " " << f_down << " " << f_icp << " "
                    << knn_time * 1000 << " " << solve_time * 1000 << " "
                    << f_pubod << " " << f_map << " " << f_pubtl << " "
                    << iter_count << " " << knn_count << " " << rematch_num << " "
                    << effect_feat_num << " " << feats_points_size << " " << feats_points_full_size << " "
                    << kdtree_size_st << " " << kdtree_size_end << " "
                    << res_mean_last << " " << -1 << endl;
            }
            // if(frame_num < 100 && imu_en){
            //     downSizeMap_fir.setInputCloud(feats_map);
            //     downSizeMap_fir.filter(*feats_map_down_fir);
            //     std::string filenamequjibian = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/map/map_fir"+ std::to_string(frame_num)+ ".pcd";
            //     pcl::io::savePCDFile(filenamequjibian, *(feats_map_down_fir));
            // }
            // if(frame_id > 400){
            //     downSizeMap.setInputCloud(feats_map);
            //     downSizeMap.filter(*feats_map_down);
            //     std::string filenamequjibian = "/home/myx/fighting/LGO_WS/src/LiDAR_DYNAMIC_INIT/pcb/map/map"+ std::to_string(0)+ ".pcd";
            //     pcl::io::savePCDFile(filenamequjibian, *(feats_map_down));
            // }

            if (dynamic_init->dynamic_init_fished == false)
            {
                time_result << "ID = " << measures_num <<
                    " imu_process = " << std::chrono::duration_cast<std::chrono::duration<double> >(t_imu_process_end - t_imu_process_begin).count() * 1000 << " ms" <<
                    " down filter = " << std::chrono::duration_cast<std::chrono::duration<double> >(t_downfilter_end - t_downfilter_begin).count() * 1000 << " ms" <<
                    " icp = "         << std::chrono::duration_cast<std::chrono::duration<double> >(icp_end - icp_begin).count() * 1000 << " ms" <<
                    " knn total cost = " << knn_time * 1000 << " ms" << " solve = " << solve_time * 1000 << " ms" <<
                    " map_increment = " << std::chrono::duration_cast<std::chrono::duration<double> >(map_increment_end - map_increment_begin).count() * 1000 << " ms" <<
                    " odometry time = " << std::chrono::duration_cast<std::chrono::duration<double> >(t_imu_process_end - t_imu_process_begin).count() * 1000
                            + std::chrono::duration_cast<std::chrono::duration<double> >(t_downfilter_end - t_downfilter_begin).count() * 1000 
                            + std::chrono::duration_cast<std::chrono::duration<double> >(icp_end - icp_begin).count() * 1000
                            + knn_time * 1000 + solve_time * 1000
                            + std::chrono::duration_cast<std::chrono::duration<double> >(map_increment_end - map_increment_begin).count() * 1000 << " ms" <<
                    " d_init_solve_time = " << d_init_solve_time << " ms" <<
                    " sum_d_init_solve_time = " << sum_d_init_solve_time << " ms" <<endl;
            }

            frame_num++;

            {
                std::string cmp_rtype = "predict";
                if (flg_EKF_inited && cmp_applied && cmp_cov_committed)      cmp_rtype = "post";
                else if (flg_EKF_inited && cmp_applied)                     cmp_rtype = "partial_applied";
                else if (flg_EKF_inited && !cmp_applied && cmp_propagated)  cmp_rtype = "invalid";
                cmp_states << lidar_end_time << " " << cmp_rtype << " "
                    << (int)cmp_applied << " " << (int)cmp_cov_committed << " " << (int)cmp_propagated << " "
                    << (int)flg_EKF_inited << " " << (int)imu_en << " " << effect_feat_num << " "
                    << state.vel_end.transpose() << " "
                    << state.bias_g.transpose() << " "
                    << state.gravity.transpose() << endl;
                if (!cmp_first_post_written && cmp_rtype == "post" && dynamic_init->dynamic_init_fished)
                {
                    // 初始化完成后的首个有效正式后验(时间协议与每帧行的 t_state 一致)
                    cmp_first_post_written = true;
                    cmp_events << "first_post " << lidar_end_time << " "
                        << state.vel_end.transpose() << " "
                        << state.bias_g.transpose() << " "
                        << state.gravity.transpose() << endl;
                }
            }


            double rtk_t = 0; V3D rtk_v = V3D::Zero();
            bool rtk_ok = get_rtk_at(lidar_end_time, rtk_t, rtk_v);
            if (diagnostic.enabled) {
                diagnostic.rtk_found = rtk_ok;
                if (rtk_ok) {
                    diagnostic.rtk_time = rtk_t;
                    diagnostic.rtk_velocity = rtk_v;
                }
            }
            Li_D_init_fastlio <<  std::fixed << std::setprecision(9) 
            <<lidar_end_time << " " 
            <<state.vel_end.transpose() << " " 
            << state.bias_g.transpose() << " " 
            << state.gravity.transpose() << " " 
            << init_frame_flag << " " 
            << (rtk_ok ? rtk_t : std::nan("")) << " " 
            << (rtk_ok ? rtk_v(0) : std::nan("")) << " " 
            << (rtk_ok ? rtk_v(1) : std::nan("")) << " " 
            << (rtk_ok ? rtk_v(2) : std::nan("")) << " " << endl;

            if (dynamic_init->dynamic_init_fished)
            {
                double imu_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t_imu_process_end - t_imu_process_begin).count() * 1000;
                double downfilter_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t_downfilter_end - t_downfilter_begin).count() * 1000;
                double icp_ms = std::chrono::duration_cast<std::chrono::duration<double> >(icp_end - icp_begin).count() * 1000;
                double map_inc_ms = std::chrono::duration_cast<std::chrono::duration<double> >(map_increment_end - map_increment_begin).count() * 1000;
                double frame_total_ms = std::chrono::duration_cast<std::chrono::duration<double> >(frame_end - t_imu_process_begin).count() * 1000;
                Li_D_init_lio_time << std::fixed << std::setprecision(6)
                    << lidar_end_time << " "
                    << frame_total_ms << " "
                    << imu_ms << " "
                    << downfilter_ms << " "
                    << icp_ms << " "
                    << knn_time * 1000 << " "
                    << solve_time * 1000 << " "
                    << map_inc_ms << " "
                    << std::chrono::duration_cast<std::chrono::duration<double> >(frame_end - t_program_begin).count() * 1000 << " "
                    << effect_feat_num << " " << feats_points_size << " " << feats_points_full_size << endl;
            }
            if(!dynamic_init->dynamic_init_fished){
                if(!Iteration_begin){
                    dynamic_init->Data_processing_lo(state.rot_end, state.pos_end, Measures.lidar_end_time, p_imu->tmp_pre_integration);
                }else{
                    // dynamic_init->system_state[measures_num].R = Lidar_R_wrt_IMU.transpose() * state.rot_end;
                    dynamic_init->system_state[measures_num].R = state.rot_end * Lidar_R_wrt_IMU;
                    dynamic_init->system_state[measures_num].T = state.rot_end * Lidar_T_wrt_IMU + state.pos_end;
                    // dynamic_init->system_state[measures_num].T = state.pos_end - Lidar_T_wrt_IMU;
                }
            }
            diagnostic.finish(state); // Freeze the frame before any batch solve/reset.
            if(!LGO_MODE && measures_num == dynamic_init->data_accum_length - 1 && !dynamic_init->dynamic_init_fished){
                if(!dynamic_init->dynamic_init_fished){
                    std::chrono::steady_clock::time_point t_solve_begin = std::chrono::steady_clock::now();
                    // 只读诊断: 批解观察者仅在求解期间借用, 不写任何估计器状态
                    std::unique_ptr<drift_diag::Batch> batch_diagnostic;
                    if (diagnostics_enabled) {
                        drift_diag::context().batch_id = pass_id;
                        batch_diagnostic.reset(new drift_diag::Batch(*dynamic_init, Measures.lidar_end_time, false));
                    }
                    dynamic_init->solve_Rot_bias_gyro();
                    VectorXd x_;
                    dynamic_init->LinearAlignment_withoutba(state, x_);
                    std::chrono::steady_clock::time_point t_solve_end = std::chrono::steady_clock::now();
                    d_init_solve_time = std::chrono::duration_cast<std::chrono::duration<double> >(t_solve_end - t_solve_begin).count();
                    sum_d_init_solve_time += std::chrono::duration_cast<std::chrono::duration<double> >(t_solve_end - t_solve_begin).count();
                    current_g = dynamic_init->get_g();
                    current_v = dynamic_init->get_V_0();
                    if (batch_diagnostic) {
                        // 相邻批解初速度差: 首轮无上一轮解记 null
                        const double batch_delta_v = Iteration_begin ? (current_v - Last_v).norm()
                                                                    : drift_diag::missing();
                        batch_diagnostic->finish(x_, current_v, dynamic_init->get_Grav_L0(),
                                                 dynamic_init->get_gyro_bias(), Zero3d,
                                                 batch_delta_v, Iteration_v,
                                                 d_init_solve_time * 1000.0,
                                                 dynamic_init->solve_diagnostic_ms,
                                                 std::chrono::duration<double, std::milli>(
                                                     std::chrono::steady_clock::now() - t_pass_begin).count());
                        batch_diagnostic.reset(); // 状态重置前先解绑观察者
                    }
                    if(!Iteration_begin){
                        Iteration_begin = true;
                        cout<<"kaishidiedai!!!"<<endl;
                        Last_g = dynamic_init->get_g();
                        Last_v = dynamic_init->get_V_0();
                        state.bias_a = Zero3d;
                        state.offset_R_L_I = Lidar_R_wrt_IMU;
                        state.offset_T_L_I = Lidar_T_wrt_IMU;

                        state.pos_end = Zero3d;
                        state.rot_end = Eye3d;

                        state.bias_g = dynamic_init->get_gyro_bias();
                        state.gravity = -(Lidar_R_wrt_IMU.transpose() * dynamic_init->get_Grav_L0());
                        state.vel_end = Lidar_R_wrt_IMU.transpose() * dynamic_init->get_V_0();

                        state.cov(6, 6) = state.cov(7, 7) = state.cov(8, 8) = 0.00001;  //offset_R_L_I
                        state.cov(9, 9) = state.cov(10, 10) = state.cov(11, 11) = 0.00001;   //offset_T_L_I
                        state.cov(12, 12) = state.cov(13, 13) = state.cov(14, 14) = 0.001;  //vel
                        state.cov(15, 15) = state.cov(16, 16) = state.cov(17, 17) = 0.001;  //bg
                        state.cov(18, 18) = state.cov(19, 19) = state.cov(20, 20) = 0.001;  //ba
                        state.cov(21, 21) = state.cov(22, 22) = state.cov(23, 23) = 0.001;  //g
                        p_imu->Dynamic_init_done = true;
                        imu_en = true;
                        p_imu->imu_en = imu_en;
                        p_imu->Reset();
                        p_imu->set_gyr_cov(V3D(0.1, 0.1, 0.1));
                        p_imu->set_acc_cov(V3D(0.1, 0.1, 0.1));
                        // p_imu->set_gyr_cov(V3D(0.01, 0.01, 0.01));
                        // p_imu->set_acc_cov(V3D(5, 5, 5));
                        p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
                        p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
                        p_imu->imu_need_init_ = true;
                        // flg_reset = true;
                        ikdtree.delete_tree_nodes(&ikdtree.Root_Node);
                        feats_undistort == NULL;
                        // 数据分析: 首轮(预处理+首次求解)迭代耗时, 首轮无上一轮解, delta_v 写-1
                        Li_D_init_iter_time << std::fixed << std::setprecision(6)
                            << pass_id << " "
                            << d_init_solve_time * 1000 << " "
                            << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_pass_begin).count() * 1000 << " "
                            << -1.0 << " " << 0 << endl;
                        pass_id++;
                        continue;
                    }
                }
                // 采用相邻两次迭代初速度向量差范数作为收敛判据
                if((current_v - Last_v).norm() < Iteration_v || iteration_num == Iteration_NUM)
                {
                    dynamic_init->dynamic_init_fished = true;
                    if (diagnostics_enabled) {
                        // 完成时钟统一: 后续 LIO 帧的 relative_time 以交接帧数据时间为锚
                        drift_diag::context().handoff_time = Measures.lidar_end_time;
                        drift_diag::event("init_done", lidar_end_time,
                            (current_v - Last_v).norm() < Iteration_v ? "converged" : "max_iter");
                    }
                    {
                        double t_sim_now = -1.0, lag_ms = -1.0;
                        if (ros::Time::isSimTime())
                        {
                            t_sim_now = ros::Time::now().toSec();
                            // now() 在收到首条 /clock 前为 0, 此时不可用
                            if (t_sim_now > 0.0 && Measures.lidar_end_time > 0.0)
                                lag_ms = (t_sim_now - Measures.lidar_end_time) * 1000.0;
                        }
                        bool lag_converged = ((current_v - Last_v).norm() < Iteration_v);
                        if (lag_ms >= 0.0)
                            ROS_INFO("[System delay] After Dynamic initialization: time=%.6f s, "
                                     "sim time=%.6f s, system delay=%.1f ms (%.3f s, %s)",
                                     Measures.lidar_end_time, t_sim_now, lag_ms, lag_ms / 1000.0,
                                     lag_converged ? "速度收敛" : "达最大迭代");
                        else
                            ROS_WARN("[System delay] After Dynamic initialization: use_sim_time = false, cannot compute system delay");
                        Li_D_init_lag << Measures.lidar_end_time << " " << t_sim_now << " "
                                      << lag_ms << " " << pass_id << " "
                                      << (lag_converged ? 1 : 0) << endl;
                    }
                    // ==== 初始化对比实验: 批量解参照行 + init_done 事件 ====
                    {
                        bool cmp_converged = ((current_v - Last_v).norm() < Iteration_v);
                        double cmp_t_first = dynamic_init->Initialized_data.empty() ? std::nan("")
                            : dynamic_init->Initialized_data.front().lidar_end_time;
                        cmp_events << "batch_solution " << cmp_t_first << " "
                            << dynamic_init->get_V_0().transpose() << " "
                            << (Lidar_R_wrt_IMU.transpose() * dynamic_init->get_V_0()).transpose() << " "
                            << dynamic_init->get_gyro_bias().transpose() << " "
                            << dynamic_init->get_Grav_L0().transpose() << " "
                            << (-(Lidar_R_wrt_IMU.transpose() * dynamic_init->get_Grav_L0())).transpose() << endl;
                        cmp_events << "init_done " << Measures.lidar_end_time << " "
                            << (cmp_converged ? "converged" : "max_iter") << " "
                            << state.vel_end.transpose() << " "
                            << state.bias_g.transpose() << " "
                            << state.gravity.transpose() << " "
                            << state.pos_end.transpose() << " "
                            << state_propagat.vel_end.transpose() << " "
                            << state_propagat.bias_g.transpose() << " "
                            << state_propagat.gravity.transpose() << endl;
                    }

                    // 速度初值=R_LI^T*V_0, 零偏=gyro_bias, 重力=-(R_LI^T*Grav_L0), 与状态重置公式一致
                    double rtk_t2 = 0; V3D rtk_v2 = V3D::Zero();
                    bool rtk_ok2 = get_rtk_at(Measures.lidar_end_time, rtk_t2, rtk_v2);
                    Li_D_init_fastlio << std::fixed << std::setprecision(9)
                    << Measures.lidar_end_time << " "
                    << (Lidar_R_wrt_IMU.transpose() * dynamic_init->get_V_0()).transpose() << " "
                    << dynamic_init->get_gyro_bias().transpose() << " "
                    << (-(Lidar_R_wrt_IMU.transpose() * dynamic_init->get_Grav_L0())).transpose() << " "
                    << 1 << " "
                    << (rtk_ok2 ? rtk_t2 : std::nan("")) << " "
                    << (rtk_ok2 ? rtk_v2(0) : std::nan("")) << " "
                    << (rtk_ok2 ? rtk_v2(1) : std::nan("")) << " "
                    << (rtk_ok2 ? rtk_v2(2) : std::nan("")) << " " << endl;
                    Li_D_init_iter_time << std::fixed << std::setprecision(6)
                        << pass_id << " "
                        << d_init_solve_time * 1000 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_pass_begin).count() * 1000 << " "
                        << (current_v - Last_v).norm() << " " << 1 << endl;
                    pass_id++;
                    init_frame_flag = 2;
                }else{
                    iteration_num ++;
                    Li_D_init_iter_time << std::fixed << std::setprecision(6)
                        << pass_id << " "
                        << d_init_solve_time * 1000 << " "
                        << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_pass_begin).count() * 1000 << " "
                        << (current_v - Last_v).norm() << " " << 0 << endl;
                    pass_id++;
                    Last_g = dynamic_init->get_g();
                    Last_v = current_v; // 供下一轮判据使用
                    state.bias_a = Zero3d;
                    state.offset_R_L_I = Lidar_R_wrt_IMU;
                    state.offset_T_L_I = Lidar_T_wrt_IMU;

                    state.pos_end = Zero3d;
                    state.rot_end = Eye3d;

                    state.bias_g = dynamic_init->get_gyro_bias();
                    state.gravity = -(Lidar_R_wrt_IMU.transpose() * dynamic_init->get_Grav_L0());
                    state.vel_end = Lidar_R_wrt_IMU.transpose() * dynamic_init->get_V_0();

                    state.cov(6, 6) = state.cov(7, 7) = state.cov(8, 8) = 0.00001;  //offset_R_L_I
                    state.cov(9, 9) = state.cov(10, 10) = state.cov(11, 11) = 0.00001;   //offset_T_L_I
                    state.cov(12, 12) = state.cov(13, 13) = state.cov(14, 14) = 0.001;  //vel
                    state.cov(15, 15) = state.cov(16, 16) = state.cov(17, 17) = 0.001;  //bg
                    state.cov(18, 18) = state.cov(19, 19) = state.cov(20, 20) = 0.001;  //ba
                    state.cov(21, 21) = state.cov(22, 22) = state.cov(23, 23) = 0.001;  //g
                    p_imu->Dynamic_init_done = true;
                    p_imu->Reset();
                    p_imu->set_gyr_cov(V3D(0.1, 0.1, 0.1));
                    p_imu->set_acc_cov(V3D(0.1, 0.1, 0.1));
                    // p_imu->set_gyr_cov(V3D(0.01, 0.01, 0.01));
                    // p_imu->set_acc_cov(V3D(5, 5, 5));
                    p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
                    p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
                    p_imu->imu_need_init_ = true;
                    ikdtree.delete_tree_nodes(&ikdtree.Root_Node);
                    feats_undistort == NULL;
                    // return 0;
                }
            }
        }
        status = ros::ok();
        if (diagnostics_enabled) diagnostics_rate.sleep();
        else rate.sleep();
    }

    if (diagnostics_enabled) {
        drift_diag::event("shutdown", diagnostics_processed, "normal");
        google::ShutdownGoogleLogging();
    }
    return 0;
}