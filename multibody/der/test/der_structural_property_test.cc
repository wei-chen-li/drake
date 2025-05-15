#include "drake/multibody/der/der_structural_property.h"

#include <math.h>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using std::pow;

class DerStructuralPropertyTest : public ::testing::Test {
 protected:
  const double E_ = 3e9;
  const double G_ = 0.8e9;
  const double rho_ = 910;
  constexpr static double kTol = 1e-16;
};

TEST_F(DerStructuralPropertyTest, RectangularCrossSection) {
  const double w = 1.0e-3;
  const double h = 2.5e-3;
  const auto prop = DerStructuralProperty<double>::FromRectangularCrossSection(
      w, h, E_, G_, rho_);

  const double A = w * h;
  const double I1 = h * pow(w, 3) / 12;
  const double I2 = w * pow(h, 3) / 12;
  const double J = (h * pow(w, 3) + w * pow(h, 3)) / 12;

  EXPECT_NEAR(prop.A(), A, kTol);
  EXPECT_NEAR(prop.EA(), E_ * A, kTol);
  EXPECT_NEAR(prop.EI1(), E_ * I1, kTol);
  EXPECT_NEAR(prop.EI2(), E_ * I2, kTol);
  EXPECT_NEAR(prop.GJ(), G_ * J, kTol);
  EXPECT_NEAR(prop.rhoA(), rho_ * A, kTol);
  EXPECT_NEAR(prop.rhoJ(), rho_ * J, kTol);
}

TEST_F(DerStructuralPropertyTest, EllipticalCrossSection) {
  const double a = 1.0e-3;
  const double b = 2.5e-3;
  const auto prop = DerStructuralProperty<double>::FromEllipticalCrossSection(
      a, b, E_, G_, rho_);

  const double pi = M_PI;
  const double A = pi * a * b;
  const double I1 = pi * b * pow(a, 3) / 4;
  const double I2 = pi * a * pow(b, 3) / 4;
  const double J = pi * (b * pow(a, 3) + a * pow(b, 3)) / 4;

  EXPECT_NEAR(prop.A(), A, kTol);
  EXPECT_NEAR(prop.EA(), E_ * A, kTol);
  EXPECT_NEAR(prop.EI1(), E_ * I1, kTol);
  EXPECT_NEAR(prop.EI2(), E_ * I2, kTol);
  EXPECT_NEAR(prop.GJ(), G_ * J, kTol);
  EXPECT_NEAR(prop.rhoA(), rho_ * A, kTol);
  EXPECT_NEAR(prop.rhoJ(), rho_ * J, kTol);
}

TEST_F(DerStructuralPropertyTest, CircularCrossSection) {
  const double r = 1e-3;
  const auto prop =
      DerStructuralProperty<double>::FromCircularCrossSection(r, E_, G_, rho_);

  const double pi = M_PI;
  const double A = pi * pow(r, 2);
  const double I = pi * pow(r, 4) / 4;
  const double J = pi * pow(r, 4) / 2;

  EXPECT_NEAR(prop.A(), A, kTol);
  EXPECT_NEAR(prop.EA(), E_ * A, kTol);
  EXPECT_NEAR(prop.EI1(), E_ * I, kTol);
  EXPECT_NEAR(prop.EI2(), E_ * I, kTol);
  EXPECT_NEAR(prop.GJ(), G_ * J, kTol);
  EXPECT_NEAR(prop.rhoA(), rho_ * A, kTol);
  EXPECT_NEAR(prop.rhoJ(), rho_ * J, kTol);
}

TEST_F(DerStructuralPropertyTest, ScalarConversion) {
  const double r = 1e-3;
  const auto prop1 =
      DerStructuralProperty<AutoDiffXd>::FromCircularCrossSection(r, E_, G_,
                                                                  rho_);

  const auto prop2 = prop1.ToScalarType<double>();

  EXPECT_EQ(prop1.EA().value(), prop2.EA());
  EXPECT_EQ(prop1.EI1().value(), prop2.EI1());
  EXPECT_EQ(prop1.EI2().value(), prop2.EI2());
  EXPECT_EQ(prop1.GJ().value(), prop2.GJ());
  EXPECT_EQ(prop1.rhoA().value(), prop2.rhoA());
  EXPECT_EQ(prop1.rhoJ().value(), prop2.rhoJ());
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
