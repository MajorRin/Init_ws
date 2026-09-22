#include "Dynamic_init.h"
#include <chrono>
#include <limits>

double last_lidar_end_time_ = 0;   
V3D angvel_last;
sensor_msgs::ImuConstPtr last_imu_;
const bool time_(PointType &x, PointType &y) {return (x.curvature < y.curvature);};
std::ostringstream oss;

void voxel_filter(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<PointType>::Ptr &cloud_filtered,
                float leaf_size = 1.0f) {
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.setInputCloud(cloud);
    voxel.filter(*cloud_filtered);
}

bool Dynamic_init::Data_processing(MeasureGroup& meas)
{
    Initialized_data.push_back(meas);
    if (lidar_frame_count < data_accum_length)
    {
        lidar_frame_count++;
        return false;
    }
    return true; 
}
bool Dynamic_init::Data_processing_lo(M3D rot, V3D t, double time, IntegrationBase *pre_integration){
    CalibState calibState(rot, t, time);
    calibState.pre_integration = pre_integration;
    system_state.push_back(calibState);
    if (lidar_frame_count < data_accum_length)
    {
        lidar_frame_count++;
        return false;
    }
    return true; 
}

Dynamic_init::Dynamic_init(){
    fout_LiDAR_meas.open(FILE_DIR("LiDAR_meas.txt"), ios::out);
    fout_IMU_meas.open(FILE_DIR("IMU_meas.txt"), ios::out);
    data_accum_length = 20;
    lidar_frame_count = 0;
    gyro_bias = Zero3d;
    acc_bias = Zero3d;
    Grav_L0 = Zero3d;
    V_0 = Zero3d;
    first_point = true;
    second_point = true;
}

Dynamic_init::~Dynamic_init() = default;


void Dynamic_init::clear() {
    // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
    cout << "\x1B[2J\x1B[H";
}


void Dynamic_init::Data_propagate(){
    //Calculate icp odometer as well as IMU preintegration
    
    return;
    
}

void Dynamic_init::Dynamic_Initialization(int &orig_odom_freq, int &cut_frame_num, double &timediff_imu_wrt_lidar,
                                const double &move_start_time) {

    TimeConsuming time("Batch optimization");


 
    printf(BOLDBLUE"============================================================ \n\n" RESET);
    // print_initialization_result(time_L_I, Rot_Lidar_wrt_IMU, Trans_Lidar_wrt_IMU, gyro_bias, acc_bias, Grav_L0);
    printf(BOLDBLUE"============================================================ \n\n" RESET);
    printf(BOLDCYAN "[Initialization] Lidar IMU initialization done.\n");
    printf("" RESET);
}

void Dynamic_init::solve_Rot_bias_gyro() {
    M3D A;
    V3D b;
    V3D delta_bg;
    A.setZero();
    b.setZero();
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++)
    {
        auto frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->R.transpose() * frame_j->R);
        tmp_A = frame_j->pre_integration->jacobian.template block<3, 3>(3, 12);
        tmp_b = 2 * (frame_j->pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    auto decomposition = A.ldlt();
    const auto decomposition_status = decomposition.info();
    delta_bg = decomposition.solve(b);
    if (solve_observer) {
        const auto diagnostic_begin = std::chrono::steady_clock::now();
        solve_observer->solved("gyro_bias", -1, A, b,
            delta_bg, V3D::Constant(std::numeric_limits<double>::quiet_NaN()),
            decomposition_status, decomposition.info());
        solve_diagnostic_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - diagnostic_begin).count();
    }
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    gyro_bias += delta_bg;

    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end( ); frame_i++)
    {
        auto frame_j = next(frame_i);
        frame_j->pre_integration->repropagate(Vector3d::Zero(), gyro_bias);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void Dynamic_init::RefineGravity(StatesGroup icp_state, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = system_state.size();
    int n_state = all_frame_count * 3 + 2 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();


    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
        {
            auto frame_j = next(frame_i);

            MatrixXd  tmp_A(6, 11);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 3>(0, 8) = -frame_j->pre_integration->jacobian.template block<3, 3>(0, 9);     
            tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p- icp_state.offset_T_L_I\
                        - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I)\
                        - frame_i->R.transpose() * dt * dt / 2 * g0;
            //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
            tmp_A.block<3, 2>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 3>(3, 8) = -frame_j->pre_integration->jacobian.template block<3, 3>(6, 9);
            tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v\
                        - frame_i->R.transpose() * dt * Matrix3d::Identity() * g0;
            //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            cov_inv.setIdentity();
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<5, 5>() += r_A.bottomRightCorner<5, 5>();
            b.tail<5>() += r_b.tail<5>();

            A.block<6, 5>(i * 3, n_state - 5) += r_A.topRightCorner<6, 5>();
            A.block<5, 6>(n_state - 5, i * 3) += r_A.bottomLeftCorner<5, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        auto decomposition = A.ldlt();
        const auto decomposition_status = decomposition.info();
        x = decomposition.solve(b);
        const auto solve_status = decomposition.info();
            VectorXd dg = x.segment<2>(n_state - 5);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
        if (solve_observer) {
            const auto diagnostic_begin = std::chrono::steady_clock::now();
            solve_observer->solved("gravity_refine", k, A, b,
                x, g0, decomposition_status, solve_status);
            solve_diagnostic_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - diagnostic_begin).count();
        }
            //double s = x(n_state - 1);
    }   
    g = g0;
}

void Dynamic_init::LinearAlignment(StatesGroup icp_state, VectorXd &x){
    int all_pose_count = system_state.size();
    int n_state = all_pose_count * 3 + 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    int i = 0;
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
    {
        auto frame_j = next(frame_i);

        MatrixXd  tmp_A(6, 12);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 9) = -frame_j->pre_integration->jacobian.template block<3, 3>(0, 9);     
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p - icp_state.offset_T_L_I\
                    - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I);
        
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
        tmp_A.block<3, 3>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 9) = -frame_j->pre_integration->jacobian.template block<3, 3>(6, 9);
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<6, 6>() += r_A.bottomRightCorner<6, 6>();
        b.tail<6>() += r_b.tail<6>();

        A.block<6, 6>(i * 3, n_state - 6) += r_A.topRightCorner<6, 6>();
        A.block<6, 6>(n_state - 6, i * 3) += r_A.bottomLeftCorner<6, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    auto decomposition = A.ldlt();
    const auto decomposition_status = decomposition.info();
    x = decomposition.solve(b);
    const auto solve_status = decomposition.info();
    if (debug_dump_matrices) {
        std::ofstream outFile(DEBUG_FILE_DIR("A_ba.txt"));
        outFile << A;
    }
    Vector3d g = x.segment<3>(n_state - 6);
    if (solve_observer) {
        const auto diagnostic_begin = std::chrono::steady_clock::now();
        solve_observer->solved("linear_initial", -1, A, b,
            x, g, decomposition_status, solve_status);
        solve_diagnostic_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - diagnostic_begin).count();
    }
    auto ba = x.segment<3>(n_state - 3);
    auto v_0 = x.segment<3>(0);
    cout<<"size: "<<x.size()<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" ba     " <<  ba.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0.transpose());
    RefineGravity(icp_state, g, x);
    auto ba_ = x.segment<3>(n_state - 4);
    cout<<"size: "<<x.size()<<endl;
    auto v_0_ = x.segment<3>(0);
    cout<<"---------------------------------"<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" ba     " <<  ba.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0_.transpose());
}


void Dynamic_init::RefineGravity_withoutba(StatesGroup icp_state, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = system_state.size();
    int n_state = all_frame_count * 3 + 2;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();


    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
        {
            auto frame_j = next(frame_i);

            MatrixXd  tmp_A(6, 8);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;   
            tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p- icp_state.offset_T_L_I\
                        - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I)\
                        - frame_i->R.transpose() * dt * dt / 2 * g0;
            //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
            tmp_A.block<3, 2>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v\
                        - frame_i->R.transpose() * dt * Matrix3d::Identity() * g0;
            //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            cov_inv.setIdentity();
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
            b.tail<2>() += r_b.tail<2>();

            A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
            A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        auto decomposition = A.ldlt();
        const auto decomposition_status = decomposition.info();
        x = decomposition.solve(b);
        const auto solve_status = decomposition.info();
            VectorXd dg = x.segment<2>(n_state - 2);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
        if (solve_observer) {
            const auto diagnostic_begin = std::chrono::steady_clock::now();
            solve_observer->solved("gravity_refine", k, A, b,
                x, g0, decomposition_status, solve_status);
            solve_diagnostic_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - diagnostic_begin).count();
        }
            //double s = x(n_state - 1);
    }   
    g = g0;
}

void Dynamic_init::LinearAlignment_withoutba(StatesGroup icp_state, VectorXd &x){
    int all_pose_count = system_state.size();
    int n_state = all_pose_count * 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    int i = 0;
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
    {
        auto frame_j = next(frame_i);

        MatrixXd  tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->pre_integration->sum_dt;
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p - icp_state.offset_T_L_I\
                    - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I);
        // cout << "delta_p   " << frame_j->pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
        tmp_A.block<3, 3>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v;
        // cout << "delta_v   " << frame_j->pre_integration->delta_v.transpose() << endl;
        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    auto decomposition = A.ldlt();
    const auto decomposition_status = decomposition.info();
    x = decomposition.solve(b);
    const auto solve_status = decomposition.info();
    Vector3d g = x.segment<3>(n_state - 3);
    if (solve_observer) {
        const auto diagnostic_begin = std::chrono::steady_clock::now();
        solve_observer->solved("linear_initial", -1, A, b,
            x, g, decomposition_status, solve_status);
        solve_diagnostic_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - diagnostic_begin).count();
    }
    auto v_0 = x.segment<3>(0);
    if (debug_dump_matrices) {
        std::ofstream outFile(DEBUG_FILE_DIR("A.txt"));
        outFile << A;
    }
    // cout<<"size: "<<x.size()<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0.transpose());
    g_ = g.norm();
    RefineGravity_withoutba(icp_state, g, x);
    auto v_0_ = x.segment<3>(0);
    cout<<"------------------------- After RefineGravity ---------------------------"<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0_.transpose());
    Grav_L0 = g;
    V_0 = v_0_;
}
void Dynamic_init::print_initialization_result(V3D &bias_g, V3D &bias_a, V3D gravity, V3D V_0){
    cout.setf(ios::fixed);
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Bias of Gyroscope        = " << bias_g.transpose() << " rad/s" << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Bias of Accelerometer    = " << bias_a.transpose() << " m/s^2" << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Gravity in World Frame   = " << gravity.transpose() << " m/s^2" << endl << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "V_0  = " << V_0.transpose() << " m/s" << endl << endl;
}
