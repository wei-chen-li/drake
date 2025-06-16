#include "drake/multibody/der/contact_energy.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

namespace {

double casadi_sq(double x) {
  return x * x;
}

/* jacobian:(i0[12])->(o0[12]) */
void casadi_f0(const double** arg, double** res) {
  double a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11;
  double a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23;
  double a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35;
  a00 = arg[0] ? arg[0][9] : 0;
  a01 = arg[0] ? arg[0][6] : 0;
  a00 = (a00 - a01);
  a02 = arg[0] ? arg[0][3] : 0;
  a03 = arg[0] ? arg[0][0] : 0;
  a02 = (a02 - a03);
  a04 = casadi_sq(a02);
  a05 = arg[0] ? arg[0][4] : 0;
  a06 = arg[0] ? arg[0][1] : 0;
  a05 = (a05 - a06);
  a07 = casadi_sq(a05);
  a04 = (a04 + a07);
  a07 = arg[0] ? arg[0][5] : 0;
  a08 = arg[0] ? arg[0][2] : 0;
  a07 = (a07 - a08);
  a09 = casadi_sq(a07);
  a04 = (a04 + a09);
  a09 = casadi_sq(a00);
  a10 = arg[0] ? arg[0][10] : 0;
  a11 = arg[0] ? arg[0][7] : 0;
  a10 = (a10 - a11);
  a12 = casadi_sq(a10);
  a09 = (a09 + a12);
  a12 = arg[0] ? arg[0][11] : 0;
  a13 = arg[0] ? arg[0][8] : 0;
  a12 = (a12 - a13);
  a14 = casadi_sq(a12);
  a09 = (a09 + a14);
  a14 = (a04 * a09);
  a15 = (a02 * a00);
  a16 = (a05 * a10);
  a15 = (a15 + a16);
  a16 = (a07 * a12);
  a15 = (a15 + a16);
  a16 = casadi_sq(a15);
  a14 = (a14 - a16);
  a16 = 0.;
  a17 = (a14 == a16);
  a17 = (!a17);
  a01 = (a01 - a03);
  a03 = (a02 * a01);
  a11 = (a11 - a06);
  a06 = (a05 * a11);
  a03 = (a03 + a06);
  a13 = (a13 - a08);
  a08 = (a07 * a13);
  a03 = (a03 + a08);
  a08 = (a03 * a09);
  a06 = (a00 * a01);
  a18 = (a10 * a11);
  a06 = (a06 + a18);
  a18 = (a12 * a13);
  a06 = (a06 + a18);
  a18 = (a06 * a15);
  a08 = (a08 - a18);
  a08 = (a08 / a14);
  a18 = (a17 ? a08 : 0);
  a19 = (a16 <= a18);
  a20 = (a18 <= a16);
  a20 = (a19 + a20);
  a19 = (a19 / a20);
  a18 = fmax(a18, a16);
  a20 = 1.;
  a21 = (a18 <= a20);
  a22 = (a20 <= a18);
  a22 = (a21 + a22);
  a21 = (a21 / a22);
  a18 = fmin(a18, a20);
  a22 = (a15 * a18);
  a22 = (a22 - a06);
  a22 = (a22 / a09);
  a23 = (a16 <= a22);
  a24 = (a22 <= a16);
  a24 = (a23 + a24);
  a23 = (a23 / a24);
  a24 = fmax(a22, a16);
  a25 = (a24 <= a20);
  a26 = (a20 <= a24);
  a26 = (a25 + a26);
  a25 = (a25 / a26);
  a24 = fmin(a24, a20);
  a26 = (a15 * a24);
  a26 = (a26 + a03);
  a26 = (a26 / a04);
  a27 = (a16 <= a26);
  a28 = (a26 <= a16);
  a28 = (a27 + a28);
  a27 = (a27 / a28);
  a16 = fmax(a26, a16);
  a28 = (a16 <= a20);
  a29 = (a20 <= a16);
  a29 = (a28 + a29);
  a28 = (a28 / a29);
  a16 = fmin(a16, a20);
  a20 = (a07 * a16);
  a29 = (a12 * a24);
  a20 = (a20 - a29);
  a20 = (a20 - a13);
  a29 = (a02 * a16);
  a30 = (a00 * a24);
  a29 = (a29 - a30);
  a29 = (a29 - a01);
  a30 = casadi_sq(a29);
  a31 = (a05 * a16);
  a32 = (a10 * a24);
  a31 = (a31 - a32);
  a31 = (a31 - a11);
  a32 = casadi_sq(a31);
  a30 = (a30 + a32);
  a32 = casadi_sq(a20);
  a30 = (a30 + a32);
  a30 = sqrt(a30);
  a20 = (a20 / a30);
  a32 = (a07 * a20);
  a31 = (a31 / a30);
  a33 = (a05 * a31);
  a32 = (a32 + a33);
  a29 = (a29 / a30);
  a30 = (a02 * a29);
  a32 = (a32 + a30);
  a28 = (a28 * a32);
  a27 = (a27 * a28);
  a28 = (a27 / a04);
  a32 = (a15 * a28);
  a30 = (a12 * a20);
  a33 = (a10 * a31);
  a30 = (a30 + a33);
  a33 = (a00 * a29);
  a30 = (a30 + a33);
  a32 = (a32 - a30);
  a25 = (a25 * a32);
  a23 = (a23 * a25);
  a25 = (a23 / a09);
  a32 = (a15 * a25);
  a21 = (a21 * a32);
  a19 = (a19 * a21);
  a21 = (a19 / a14);
  a32 = (a15 * a21);
  a32 = (-a32);
  a32 = (a17 ? a32 : 0);
  a32 = (a32 - a25);
  a30 = (a00 * a32);
  a30 = (a30 - a29);
  a33 = (a09 * a21);
  a33 = (a17 ? a33 : 0);
  a33 = (a28 + a33);
  a34 = (a02 * a33);
  a30 = (a30 + a34);
  a34 = (a16 * a29);
  a35 = (a01 * a33);
  a34 = (a34 + a35);
  a35 = (a02 + a02);
  a08 = (a08 / a14);
  a08 = (a08 * a19);
  a19 = (a09 * a08);
  a19 = (-a19);
  a19 = (a17 ? a19 : 0);
  a26 = (a26 / a04);
  a26 = (a26 * a27);
  a19 = (a19 - a26);
  a35 = (a35 * a19);
  a34 = (a34 + a35);
  a28 = (a24 * a28);
  a18 = (a18 * a25);
  a28 = (a28 + a18);
  a06 = (a06 * a21);
  a06 = (-a06);
  a06 = (a17 ? a06 : 0);
  a28 = (a28 + a06);
  a15 = (a15 + a15);
  a15 = (a15 * a08);
  a15 = (a17 ? a15 : 0);
  a28 = (a28 + a15);
  a15 = (a00 * a28);
  a34 = (a34 + a15);
  a15 = (a30 + a34);
  a15 = (-a15);
  if (res[0] != 0) res[0][0] = a15;
  a15 = (a10 * a32);
  a15 = (a15 - a31);
  a06 = (a05 * a33);
  a15 = (a15 + a06);
  a06 = (a16 * a31);
  a18 = (a11 * a33);
  a06 = (a06 + a18);
  a18 = (a05 + a05);
  a18 = (a18 * a19);
  a06 = (a06 + a18);
  a18 = (a10 * a28);
  a06 = (a06 + a18);
  a18 = (a15 + a06);
  a18 = (-a18);
  if (res[0] != 0) res[0][1] = a18;
  a18 = (a12 * a32);
  a18 = (a18 - a20);
  a25 = (a07 * a33);
  a18 = (a18 + a25);
  a16 = (a16 * a20);
  a33 = (a13 * a33);
  a16 = (a16 + a33);
  a33 = (a07 + a07);
  a33 = (a33 * a19);
  a16 = (a16 + a33);
  a33 = (a12 * a28);
  a16 = (a16 + a33);
  a33 = (a18 + a16);
  a33 = (-a33);
  if (res[0] != 0) res[0][2] = a33;
  if (res[0] != 0) res[0][3] = a34;
  if (res[0] != 0) res[0][4] = a06;
  if (res[0] != 0) res[0][5] = a16;
  a01 = (a01 * a32);
  a29 = (a24 * a29);
  a01 = (a01 - a29);
  a00 = (a00 + a00);
  a03 = (a03 * a21);
  a03 = (a17 ? a03 : 0);
  a22 = (a22 / a09);
  a22 = (a22 * a23);
  a03 = (a03 - a22);
  a04 = (a04 * a08);
  a04 = (-a04);
  a17 = (a17 ? a04 : 0);
  a03 = (a03 + a17);
  a00 = (a00 * a03);
  a01 = (a01 + a00);
  a02 = (a02 * a28);
  a01 = (a01 + a02);
  a30 = (a30 - a01);
  if (res[0] != 0) res[0][6] = a30;
  a11 = (a11 * a32);
  a31 = (a24 * a31);
  a11 = (a11 - a31);
  a10 = (a10 + a10);
  a10 = (a10 * a03);
  a11 = (a11 + a10);
  a05 = (a05 * a28);
  a11 = (a11 + a05);
  a15 = (a15 - a11);
  if (res[0] != 0) res[0][7] = a15;
  a13 = (a13 * a32);
  a24 = (a24 * a20);
  a13 = (a13 - a24);
  a12 = (a12 + a12);
  a12 = (a12 * a03);
  a13 = (a13 + a12);
  a07 = (a07 * a28);
  a13 = (a13 + a07);
  a18 = (a18 - a13);
  if (res[0] != 0) res[0][8] = a18;
  if (res[0] != 0) res[0][9] = a01;
  if (res[0] != 0) res[0][10] = a11;
  if (res[0] != 0) res[0][11] = a13;
}

template <typename T>
double ExtractDoubleOrThrow(const T& in) {
  return drake::ExtractDoubleOrThrow(in);
}
template <>
double ExtractDoubleOrThrow<AutoDiffXAutoDiffXd>(
    const AutoDiffXAutoDiffXd& in) {
  return in.value().value();
}

}  // namespace

template <typename T>
T ComputeDistanceBetweenLineSegments(const Eigen::Ref<const Vector3<T>>& x1,
                                     const Eigen::Ref<const Vector3<T>>& x2,
                                     const Eigen::Ref<const Vector3<T>>& x3,
                                     const Eigen::Ref<const Vector3<T>>& x4) {
  /* e₁₂ = x₂ - x₁, e₃₄ = x₄ - x₃, e₁₃ = x₃ - x₁ */
  const Vector3<T> e12 = x2 - x1;
  const Vector3<T> e34 = x4 - x3;
  const Vector3<T> e13 = x3 - x1;
  /* D₁ u - R  v = S₁
     R  u - D₂ v = S₂ */
  const T D1 = e12.dot(e12);
  const T D2 = e34.dot(e34);
  const T R = e12.dot(e34);
  const T S1 = e12.dot(e13);
  const T S2 = e34.dot(e13);

  constexpr auto clamp = [](T* val) {
    if (ExtractDoubleOrThrow(*val) <= 0)
      *val = 0;
    else if (ExtractDoubleOrThrow(*val) >= 1)
      *val = 1;
  };

  const T denom = D1 * D2 - R * R;
  T u =
      (ExtractDoubleOrThrow(denom) != 0.0) ? (S1 * D2 - S2 * R) / denom : T(0);
  clamp(&u);
  T v = (R * u - S2) / D2;
  clamp(&v);
  u = (R * v + S1) / D1;
  clamp(&u);

  const T distance = (e12 * u - e34 * v - e13).norm();
  return distance;
}

template <typename T>
Vector<T, 12> ComputeLineSegmentsDistanceJacobian(
    const Eigen::Ref<const Vector3<T>>& x1,
    const Eigen::Ref<const Vector3<T>>& x2,
    const Eigen::Ref<const Vector3<T>>& x3,
    const Eigen::Ref<const Vector3<T>>& x4) {
  Vector<double, 12> xs;
  xs.template segment<3>(0) = x1;
  xs.template segment<3>(3) = x2;
  xs.template segment<3>(6) = x3;
  xs.template segment<3>(9) = x4;
  Vector<double, 12> jacobian;
  const double* arg[1] = {xs.data()};
  double* res[1] = {jacobian.data()};
  casadi_f0(arg, res);
  return jacobian;
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&ComputeDistanceBetweenLineSegments<T>));

template AutoDiffXAutoDiffXd
ComputeDistanceBetweenLineSegments<AutoDiffXAutoDiffXd>(
    const Eigen::Ref<const Vector3<AutoDiffXAutoDiffXd>>& x1,
    const Eigen::Ref<const Vector3<AutoDiffXAutoDiffXd>>& x2,
    const Eigen::Ref<const Vector3<AutoDiffXAutoDiffXd>>& x3,
    const Eigen::Ref<const Vector3<AutoDiffXAutoDiffXd>>& x4);

template Vector<double, 12> ComputeLineSegmentsDistanceJacobian<double>(
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
