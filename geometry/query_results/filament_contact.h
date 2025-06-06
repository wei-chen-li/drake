#pragma once

#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/ssize.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {

/* A discrete representation of the intersection of two geometries A and B, at
 least one of which is a filament. We maintain the convention that geometry A is
 always a filament and geometry B may be a filament. When both geometries are
 filaments, we maintain the convention that the GeometryId of geometry A is less
 than or equal to (in the case of self contact) the GeometryId of geometry B.
 @tparam_double_only */
template <typename T>
class FilamentContactGeometryPair {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentContactGeometryPair);

  /* Constructs a filament-filament contact geometry pair with the given data.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_coordinates_A
      Tuples (𝑤₁, r, 𝑤₂) so that the velocity of a point Ac coincident with
      the contact point C and affixed to geometry A, denoted as v_WAc, equals
      𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @param[in] contact_edge_indexes_B
      The indexes of filament B edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_coordinates_B
      Tuples (𝑤₁, r, 𝑤₂) so that the velocity of a point Bc coincident with
      the contact point C and affixed to geometry B, denoted as v_WBc, equals
      𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @pre `id_A < id_B`.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`,
      `kinematic_coordinates_A`, `contact_edge_indexes_B`, and
      `kinematic_coordinates_B` have the same size and not empty. */
  FilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A,
      std::vector<int> contact_edge_indexes_B,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_B);

  /* Constructs a filament-rigid contact geometry pair with the given data.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the rigid geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_coordinates_A
      Tuples (𝑤₁, r, 𝑤₂) so that the velocity of a point Ac coincident with
      the contact point C and affixed to geometry A, denoted as v_WAc, equals
      𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`, and
      `kinematic_coordinates_A` have the same size and not empty. */
  FilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A);

  /* Returns the GeometryId of geometry A. If `is_B_filament()` is true, this
   is guaranteed to be less than or equal to id_B(). */
  GeometryId id_A() const { return id_A_; }

  /* Returns the GeometryId of geometry B. If `is_B_filament()` is true, this
   is guaranteed to be greater than or equal to id_A(). */
  GeometryId id_B() const { return id_B_; }

  /* Returns true if geometry B is a filament, false if geometry B is rigid. */
  bool is_B_filament() const { return contact_edge_indexes_B_.has_value(); }

  /* Returns the total number of contacts between this geometry pair. */
  int num_contacts() const { return ssize(p_WCs_); }

  /* Returns the contact points expressed in the world frame. */
  const std::vector<Vector3<T>>& p_WCs() const { return p_WCs_; }

  /* Returns the contact normals pointing from geometry B into geometry A
   expressed in the world frame. */
  const std::vector<Vector3<T>>& nhats_BA_W() const { return nhats_BA_W_; }

  /* Returns rotation matrices that transform the basis of frame W into the
   basis of an arbitrary frame C. In this transformation, the z-axis of frame,
   Cz, is aligned with the vector n̂. The vector n̂ represents the normal as
   opposite to the one reported in `nhats_BA_W()`. Cx and Cy are arbitrary but
   sufficient to form the right-handed basis. The ordering is same as p_WCs() or
   nhats_BA_W(). */
  const std::vector<math::RotationMatrix<T>>& R_WCs() const { return R_WCs_; }

  /* Returns the signed distances of the contacts. The ordering is same as
   p_WCs() or nhats_BA_W(). */
  const std::vector<T>& signed_distances() const { return signed_distances_; }

  /* Returns the indexes of filament A edges participating in contact. The
   ordering is same as p_WCs() or nhats_BA_W(). */
  const std::vector<int>& contact_edge_indexes_A() const {
    return contact_edge_indexes_A_;
  }

  /* Returns tuples (𝑤₁, r, 𝑤₂) so that the velocity of a point Ac coincident
    with the contact point C and affixed to geometry A, denoted as v_WAc, equals
    𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁. */
  const std::vector<std::tuple<T, Vector3<T>, T>>& kinematic_coordinates_A()
      const {
    return kinematic_coordinates_A_;
  }

  /* Returns the indexes of filament B edges participating in contact. The
   ordering is same as p_WCs() or nhats_BA_W().
   @pre `is_B_filament()`. */
  const std::vector<int>& contact_edge_indexes_B() const {
    DRAKE_THROW_UNLESS(is_B_filament());
    return *contact_edge_indexes_B_;
  }

  /* Returns tuples (𝑤₁, r, 𝑤₂) so that the velocity of a point Bc coincident
    with the contact point C and affixed to geometry B, denoted as v_WBc, equals
    𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
    @pre `is_B_filament()`. */
  const std::vector<std::tuple<T, Vector3<T>, T>>& kinematic_coordinates_B()
      const {
    DRAKE_THROW_UNLESS(is_B_filament());
    return *kinematic_coordinates_B_;
  }

 private:
  GeometryId id_A_;
  GeometryId id_B_;
  std::vector<Vector3<T>> p_WCs_;
  std::vector<Vector3<T>> nhats_BA_W_;
  std::vector<math::RotationMatrix<T>> R_WCs_;
  std::vector<T> signed_distances_;
  std::vector<int> contact_edge_indexes_A_;
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A_;
  std::optional<std::vector<int>> contact_edge_indexes_B_;
  std::optional<std::vector<std::tuple<T, Vector3<T>, T>>>
      kinematic_coordinates_B_;
};

/* Data structure to hold contact information about filaments.
 @tparam_double_only */
template <typename T>
class FilamentContact {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentContact);

  FilamentContact() = default;

  /* Adds a filament-filament contact geometry pair to this FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges under contact for each contact point.
   @param[in] contact_edge_indexes_B
      The indexes of filament B edges under contact for each contact point.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is used to compute the
      kinematic coordinates.
   @param[in] node_positions_B
      The position of the nodes of filament B. This is used to compute the
      kinematic coordinates.
   @pre `id_A < id_B`.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`, and
      `contact_edge_indexes_B` have the same size and not empty. */
  void AddFilamentFilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<int> contact_edge_indexes_B,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B);

  /* Adds a filament-rigid contact geometry pair to this FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the rigid geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges under contact for each contact point.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is used to compute the
      kinematic coordinates.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, and `contact_edge_indexes_A`
      have the same size and not empty. */
  void AddFilamentRigidContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A);

  /* Returns a list of geometry pairs that are in contact, where at least one of
   the geometries is a filament. Each pair includes associated contact
   information. */
  const std::vector<FilamentContactGeometryPair<T>>& contact_geometry_pairs()
      const {
    return contact_geometry_pairs_;
  }

  /* Returns the set of edge indices for the filament geometry with `id` that
   are under contact. */
  const std::unordered_set<int>& contact_edges(GeometryId id) const;

 private:
  std::unordered_map<GeometryId, std::unordered_set<int>> id_to_contact_edges_;
  std::vector<FilamentContactGeometryPair<T>> contact_geometry_pairs_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
