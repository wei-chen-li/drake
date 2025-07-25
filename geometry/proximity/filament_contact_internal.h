#pragma once

#include <memory>
#include <vector>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/proximity/collision_filter.h"
#include "drake/geometry/query_results/filament_contact.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/* Forward declaration. */
namespace hydroelastic {
class Geometries;
}

namespace filament {

class Geometries {
 public:
  Geometries();

  Geometries(const Geometries& other);
  Geometries& operator=(const Geometries& other);
  Geometries(Geometries&& other);
  Geometries& operator=(Geometries&& other);

  ~Geometries();

  // clang-format off
  /* Adds a filament geometry
   @param id        The unique identifier for the geometry.
   @param filament  The filament with information on number of nodes and edges,
                    and cross-section.
   @param props     The proximity properties for the filament. See table below.
   @pre There is no previous geometry associated with `id`.

   |  Group name  |          Property Name          |  Property Type   | Property Description                                            |
   | :----------: | :-----------------------------: | :--------------: | :-------------------------------------------------------------: |
   | hydroelastic | compliance_type                 | HydroelasticType | Must be HydroelasticType::kCompliant (default kCompliant).      |
   | hydroelastic | hydroelastic_modulus            | double           | Hydroelastic modulus.                                           |
   | hydroelastic | circumferential_resolution_hint | double           | Resolution hint in the circumferential direction.               |
   | hydroelastic | longitudinal_resolution_hint    | double           | Resolution hint in the longitudinal direction.                  |
   | hydroelastic | resolution_hint                 | double           | Both Resolution hints (used if the above two aren't specified). |
   | hydroelastic | margin                          | double           | Hydroelastic margin (default 0).                                |
   | collision    | self_contact                    | bool             | Is self contact enabled (default true).                         |
   */
  // clang-format on
  void AddFilamentGeometry(GeometryId id, const Filament& filament,
                           const ProximityProperties& props);

  /* Removes the geometry specified by `id`. No-op if no geometry exist with the
   provided `id`. */
  void RemoveGeometry(GeometryId id);

  /* Updates the node positions and edge m1 directors of the filament geometry
   associated with `id`.
   @param id    The identifier for the filament geometry.
   @param q_WG  The vector holding the node positions and the edge m1
                directors. Hence, the vector should have size
                (3 * num_nodes + 3 * num_edges).
   @pre The `id` is associated with a filament geometry.
   @pre `q_WG` has the correct size. */
  void UpdateFilamentConfigurationVector(
      GeometryId id, const Eigen::Ref<const Eigen::VectorXd>& q_WG);

  /* For each registered filament geometry, computes the contact data including
   self-contact, contact with other filament geometries and, and contact with
   rigid body geometries. Assumes the configuration vector of all filament
   geometries are up to date.
   @param collision_filter  The collision filter used to determine if two
                            geometries can collide, with the exception that
                            whether filament self-contact is enabled/disabled
                            is determined by the AddFilamentGeometry() proximity
                            property ('collision','self_contact').
   @param rigid_body_trees  Pointers to fcl::DynamicAABBTreeCollisionManagerd
                            trees containing rigid body geometries.
   @pre `tree != nullptr` for all `tree` in `rigid_body_trees`.
   @pre `hydroelastic_geometries != nullptr`. */
  FilamentContact<double> ComputeFilamentContact(
      const CollisionFilter& collision_filter,
      const std::vector<const void*>& rigid_body_trees,
      const hydroelastic::Geometries* hydroelastic_geometries) const;

  /* Returns true if a filament geometry with the given `id` exists. */
  bool is_filament(GeometryId id) const;

 private:
  /* Pointer to Impl. */
  class Impl;
  struct ImplDeleter {
    void operator()(Impl*);
  };
  std::unique_ptr<Impl, ImplDeleter> impl_;
};

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
