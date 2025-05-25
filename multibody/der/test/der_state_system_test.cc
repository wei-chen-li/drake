#include "drake/multibody/der/der_state_system.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* The tests here only cover a subset of the functionalities of DerStateSystem,
 the rest are covered in der_state_test since DerState is effectively a
 DerStateSystem plus an owned context. */

class DerStateSystemTester {
 public:
  DerStateSystemTester() = delete;

  template <typename T>
  static std::vector<CacheIndex> get_all_cache_indexes(
      const DerStateSystem<T>& system) {
    return {system.edge_vector_index_,
            system.edge_length_index_,
            system.tangent_index_,
            system.reference_frame_d1_index_,
            system.reference_frame_d2_index_,
            system.material_frame_m1_index_,
            system.material_frame_m2_index_,
            system.discrete_integrated_curvature_index_,
            system.curvature_kappa1_index_,
            system.curvature_kappa2_index_,
            system.reference_twist_index_,
            system.twist_index_};
  }
};

namespace {

using Eigen::Vector3;
using Eigen::VectorX;

template <typename T>
class DerStateSystemTest : public ::testing::Test {
 protected:
  void SetUp() {
    bool has_closed_ends = false;
    std::vector<Vector3<T>> node_positions = {
        Vector3<T>(0, 0, 0), Vector3<T>(1, 0, 0), Vector3<T>(2, 1, 0)};
    std::vector<T> edge_angles = {M_PI / 6, M_PI / 6};
    der_state_system_ = std::make_unique<DerStateSystem<T>>(
        has_closed_ends, node_positions, edge_angles, std::nullopt);
  }

  std::unique_ptr<const DerStateSystem<T>> der_state_system_;
};

using DefaultScalarTypes =
    ::testing::Types<double, AutoDiffXd, symbolic::Expression>;
TYPED_TEST_SUITE(DerStateSystemTest, DefaultScalarTypes);

/* Test that copying the context results in the same q, q̇, and q̈ vectors and
 quantity calculation results. */
TYPED_TEST(DerStateSystemTest, CopyContext) {
  using T = TypeParam;
  const DerStateSystem<T>& system = *this->der_state_system_;
  const int num_dofs = system.num_dofs();

  auto context1 = system.CreateDefaultContext();
  system.SetVelocity(context1.get(), VectorX<T>::Constant(num_dofs, 1.2));
  system.SetAcceleration(context1.get(), VectorX<T>::Constant(num_dofs, 3.4));

  /* Set a random q vector. */
  system.AdvancePositionToNextStep(context1.get(),
                                   VectorX<T>::LinSpaced(num_dofs, 0.0, 1.0));
  /* Set a new random q vector. */
  system.AdvancePositionToNextStep(context1.get(),
                                   VectorX<T>::LinSpaced(num_dofs, 1.0, 2.0));

  auto context2 = system.CreateDefaultContext();
  system.CopyContext(*context1, context2.get());

  /* If the abstract state holding PrevStep is not copied, some of these
   EXPECT_EQ will fail. */
  EXPECT_EQ(system.get_position(*context1), system.get_position(*context2));
  EXPECT_EQ(system.get_velocity(*context1), system.get_velocity(*context2));
  EXPECT_EQ(system.get_acceleration(*context1),
            system.get_acceleration(*context2));
  EXPECT_EQ(system.get_edge_vector(*context1),
            system.get_edge_vector(*context2));
  EXPECT_EQ(system.get_edge_length(*context1),
            system.get_edge_length(*context2));
  EXPECT_EQ(system.get_tangent(*context1),  //
            system.get_tangent(*context2));
  EXPECT_EQ(system.get_reference_frame_d1(*context1),
            system.get_reference_frame_d1(*context2));
  EXPECT_EQ(system.get_reference_frame_d2(*context1),
            system.get_reference_frame_d2(*context2));
  EXPECT_EQ(system.get_material_frame_m1(*context1),
            system.get_material_frame_m1(*context2));
  EXPECT_EQ(system.get_material_frame_m2(*context1),
            system.get_material_frame_m2(*context2));
  EXPECT_EQ(system.get_discrete_integrated_curvature(*context1),
            system.get_discrete_integrated_curvature(*context2));
  EXPECT_EQ(system.get_curvature_kappa1(*context1),
            system.get_curvature_kappa1(*context2));
  EXPECT_EQ(system.get_curvature_kappa2(*context1),
            system.get_curvature_kappa2(*context2));
  EXPECT_EQ(system.get_reference_twist(*context1),
            system.get_reference_twist(*context2));
  EXPECT_EQ(system.get_twist(*context1),  //
            system.get_twist(*context2));

  /* Cache entries in `context1` are now all up-to-date. We copy it to
   `context3` and test if `context3` cache entries are also up-to-date. */
  auto context3 = system.CreateDefaultContext();
  system.CopyContext(*context1, context3.get());

  using TypeA = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  using TypeB = Eigen::Matrix<T, 3, Eigen::Dynamic>;
  for (CacheIndex index : DerStateSystemTester::get_all_cache_indexes(system)) {
    const AbstractValue& value1 =
        system.get_cache_entry(index).GetKnownUpToDateAbstract(*context1);
    // This checks if the `context3` cache entry is up-to-date.
    const AbstractValue& value3 =
        system.get_cache_entry(index).GetKnownUpToDateAbstract(*context3);

    if (value1.type_info() == typeid(TypeA))
      EXPECT_EQ(value1.get_value<TypeA>(), value3.get_value<TypeA>());
    else
      EXPECT_EQ(value1.get_value<TypeB>(), value3.get_value<TypeB>());
  }
}

/* Test that serializing and deseralizing the context results in the same q, q̇,
 and q̈ vectors and quantity calculation results. */
TYPED_TEST(DerStateSystemTest, Serialize) {
  using T = TypeParam;
  const DerStateSystem<T>& system = *this->der_state_system_;
  const int num_dofs = system.num_dofs();

  auto context1 = system.CreateDefaultContext();
  system.SetVelocity(context1.get(), VectorX<T>::Constant(num_dofs, 1.2));
  system.SetAcceleration(context1.get(), VectorX<T>::Constant(num_dofs, 3.4));

  /* Set a random q vector. */
  system.AdvancePositionToNextStep(context1.get(),
                                   VectorX<T>::LinSpaced(num_dofs, 0.0, 1.0));
  /* Set a new random q vector. */
  system.AdvancePositionToNextStep(context1.get(),
                                   VectorX<T>::LinSpaced(num_dofs, 1.0, 2.0));

  /* Perform serialization. */
  const VectorX<T> serialized = system.Serialize(*context1);
  EXPECT_EQ(serialized.head(num_dofs), system.get_position(*context1));
  EXPECT_EQ(serialized.segment(num_dofs, num_dofs),
            system.get_velocity(*context1));
  EXPECT_EQ(serialized.segment(num_dofs * 2, num_dofs),
            system.get_acceleration(*context1));

  /* Perform the deserialization. */
  auto context2 = system.CreateDefaultContext();
  system.Deserialize(context2.get(), serialized);

  /* If the abstract state holding PrevStep is not serialized, some of these
   EXPECT_EQ will fail. */
  EXPECT_EQ(system.get_position(*context1), system.get_position(*context2));
  EXPECT_EQ(system.get_velocity(*context1), system.get_velocity(*context2));
  EXPECT_EQ(system.get_acceleration(*context1),
            system.get_acceleration(*context2));
  EXPECT_EQ(system.get_reference_frame_d1(*context1),
            system.get_reference_frame_d1(*context2));
  EXPECT_EQ(system.get_reference_twist(*context1),
            system.get_reference_twist(*context2));
}

TYPED_TEST(DerStateSystemTest, ScalarConversion) {
  using T = TypeParam;
  const DerStateSystem<T>& system = *this->der_state_system_;
  using U = std::conditional_t<std::is_same_v<T, double>, AutoDiffXd, double>;
  EXPECT_NO_THROW(system.template ToScalarType<U>());
  EXPECT_NO_THROW(system.Clone());
}

TYPED_TEST(DerStateSystemTest, SerialNumber) {
  using T = TypeParam;
  const DerStateSystem<T>& system = *this->der_state_system_;
  const VectorX<T> dummy_vector = VectorX<T>::Ones(system.num_dofs());
  auto ptr = system.CreateDefaultContext();
  Context<T>& context = *ptr;

  /* All mothods that modify the context or grant mutable access into the
   context should increment the serial number. */
  EXPECT_EQ(system.serial_number(context), 0);
  system.AdvancePositionToNextStep(&context, dummy_vector);
  EXPECT_EQ(system.serial_number(context), 1);
  system.AdjustPositionWithinStep(&context, dummy_vector);
  EXPECT_EQ(system.serial_number(context), 2);
  system.SetVelocity(&context, dummy_vector);
  EXPECT_EQ(system.serial_number(context), 3);
  system.SetAcceleration(&context, dummy_vector);
  EXPECT_EQ(system.serial_number(context), 4);
  system.get_mutable_position_within_step(&context);
  EXPECT_EQ(system.serial_number(context), 5);
  system.get_mutable_velocity(&context);
  EXPECT_EQ(system.serial_number(context), 6);
  system.get_mutable_acceleration(&context);
  EXPECT_EQ(system.serial_number(context), 7);
  system.Deserialize(&context, system.Serialize(context));
  EXPECT_EQ(system.serial_number(context), 8);
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    system.FixReferenceFrameDuringAutoDiff(&context);
    EXPECT_EQ(system.serial_number(context), 9);
  }

  auto context2 = system.CreateDefaultContext();
  system.CopyContext(context, context2.get());
  EXPECT_EQ(system.serial_number(*context2), system.serial_number(context));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
