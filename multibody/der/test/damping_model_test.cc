#include "drake/multibody/der/damping_model.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

GTEST_TEST(DampingModelTest, Getters) {
  const double mass_coeff_alpha = 1.0;
  const double stiffness_coeff_beta = 2.0;
  const DampingModel<double> model(mass_coeff_alpha, stiffness_coeff_beta);
  EXPECT_EQ(model.mass_coeff_alpha(), mass_coeff_alpha);
  EXPECT_EQ(model.stiffness_coeff_beta(), stiffness_coeff_beta);
}

GTEST_TEST(DampingModelTest, InvalidModel) {
  /* Negative coefficients are not allowed. */
  EXPECT_THROW(DampingModel<double>(1.0, -1.0), std::exception);
  EXPECT_THROW(DampingModel<double>(-1.0, 1.0), std::exception);
  /* Zero coefficients are OK. */
  EXPECT_NO_THROW(DampingModel<double>(0, 0));
}

GTEST_TEST(DampingModelTest, ScalarConversion) {
  DampingModel<AutoDiffXd> damp1(0.1, 0.2);
  DampingModel<double> damp2 = damp1.ToScalarType<double>();

  EXPECT_EQ(damp1.mass_coeff_alpha().value(), damp2.mass_coeff_alpha());
  EXPECT_EQ(damp1.stiffness_coeff_beta().value(), damp2.stiffness_coeff_beta());
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
