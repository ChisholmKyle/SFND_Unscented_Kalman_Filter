#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 2.0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // generated sigma points matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  // Sigma point spreading parameter
  lambda_ = alpha_ * alpha_ * (n_aug_ + kappa_) - n_aug_;

  // time
  time_us_ = 0;

  // initialize
  is_initialized_ = false;

  // weights of mean sigma points
  weights_mean_ = VectorXd(2 * n_aug_ + 1);
  weights_mean_(0) = lambda_ / (n_aug_ + lambda_);
  for (int k = 1; k < 2 * n_aug_ + 1; ++k)
  {
    weights_mean_(k) = 1.0 / (2.0 * (n_aug_ + lambda_));
  }

  // weights of covariance sigma points
  weights_cov_ = weights_mean_;
  weights_cov_(0) = lambda_ / (n_aug_ + lambda_) + (1 - alpha_ * alpha_ + beta_);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(0, 0) = std_laspx_ * std_laspx_;
  P_(1, 1) = std_laspy_ * std_laspy_;
  P_(2, 2) = std_radrd_ * std_radrd_;
  P_(3, 3) = std_radphi_ * std_radphi_;

  // create augmented mean vector
  x_aug_ = VectorXd(n_aug_);

  // create augmented state covariance
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(n_x_, n_x_) = std_a_ * std_a_;
  P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  // update time
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == meas_package.LASER)
    {
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
    }
    else if (meas_package.sensor_type_ == meas_package.RADAR)
    {
      double r = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_(0) = r * std::cos(phi);
      x_(1) = r * std::sin(phi);
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }
  const double delta_time = static_cast<double>(meas_package.timestamp_ - time_us_) / 1.e6;
  time_us_ = meas_package.timestamp_;

  // prediction
  Prediction(delta_time);

  // update
  if (use_laser_ && meas_package.sensor_type_ == meas_package.LASER)
  {
    // lidar
    UpdateLidar(meas_package);
  }
  else if (use_laser_ && meas_package.sensor_type_ == meas_package.RADAR)
  {
    // radar
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t)
{

  // Prediction Unscented Transform //
  // ============================== //

  // update augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  // update augmented covariance matrix
  P_aug_.topLeftCorner(5, 5) = P_;

  // create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // set first column of sigma point matrix
  Xsig_aug_.col(0) = x_aug_;

  // set remaining sigma points
  for (int k = 0; k < n_aug_; ++k)
  {
    Xsig_aug_.col(k + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(k);
    Xsig_aug_.col(k + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(k);
  }

  // predict sigma points
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  {
    // extract values for better readability
    double p_x = Xsig_aug_(0, k);
    double p_y = Xsig_aug_(1, k);
    double v = Xsig_aug_(2, k);
    double yaw = Xsig_aug_(3, k);
    double yawd = Xsig_aug_(4, k);
    double nu_a = Xsig_aug_(5, k);
    double nu_yawdd = Xsig_aug_(6, k);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (std::fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (std::sin(yaw + yawd * delta_t) - std::sin(yaw));
      py_p = p_y + v / yawd * (std::cos(yaw) - std::cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * std::cos(yaw);
      py_p = p_y + v * delta_t * std::sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * std::cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * std::sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, k) = px_p;
    Xsig_pred_(1, k) = py_p;
    Xsig_pred_(2, k) = v_p;
    Xsig_pred_(3, k) = yaw_p;
    Xsig_pred_(4, k) = yawd_p;
  }

  // predicted state mean
  x_.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  { // iterate over sigma points
    x_ = x_ + weights_mean_(k) * Xsig_pred_.col(k);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(k) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_cov_(k) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{

  // Measurements Unscented Transform //
  // ================================ //

  // set measurement dimension, lidar can measure x, y
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  {
    // 2n+1 sigma points
    // extract values for better readability
    double p_x = Xsig_pred_(0, k);
    double p_y = Xsig_pred_(1, k);

    // measurement model
    Zsig(0, k) = p_x; // x
    Zsig(1, k) = p_y; // y
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  {
    z_pred = z_pred + weights_mean_(k) * Zsig.col(k);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  { // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(k) - z_pred;

    S = S + weights_cov_(k) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;
  S = S + R;

  // Update State //
  // ============ //

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_cov_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{

  // Measurements Unscented Transform //
  // ================================ //

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  {
    // 2n+1 sigma points
    // extract values for better readability
    double p_x = Xsig_pred_(0, k);
    double p_y = Xsig_pred_(1, k);
    double v = Xsig_pred_(2, k);
    double yaw = Xsig_pred_(3, k);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, k) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig(1, k) = atan2(p_y, p_x);                                     // phi
    Zsig(2, k) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  {
    z_pred = z_pred + weights_mean_(k) * Zsig.col(k);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int k = 0; k < 2 * n_aug_ + 1; ++k)
  { // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(k) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_cov_(k) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  S = S + R;

  // Update State //
  // ============ //

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_cov_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
