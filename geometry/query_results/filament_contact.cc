#include "drake/geometry/query_results/filament_contact.h"

namespace drake {
namespace geometry {
namespace internal {

namespace {

/* Compute the tuples (𝑤₁, r, 𝑤₂) so that the velocity of the point Gc
 coincident with the contact point C and affixed to geometry G, denoted as
 v_WGc, equals 𝑤₁ ẋᵢ + r γ̇ⁱ + 𝑤₂ ẋᵢ₊₁. */
template <typename T>
std::vector<std::tuple<T, Vector3<T>, T>> ComputeKinematicCoordinates(
    const std::vector<Vector3<T>>& p_WCs,
    const std::vector<int>& contact_edge_indexes,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions) {
  DRAKE_THROW_UNLESS(p_WCs.size() == contact_edge_indexes.size());
  const int num_nodes = node_positions.cols();
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates;
  for (int k = 0; k < ssize(p_WCs); ++k) {
    const int i = contact_edge_indexes[k];
    const int ip1 = (i + 1) % num_nodes;
    const Vector3<T> node_i = node_positions.col(i);
    const Vector3<T> node_ip1 = node_positions.col(ip1);
    const T l = (node_ip1 - node_i).norm();
    const Vector3<T> t = (node_ip1 - node_i) / l;
    const T w2 = (p_WCs[k] - node_i).dot(t) / l;
    const T w1 = 1.0 - w2;
    const Vector3<T> r = t.cross(p_WCs[k] - node_i);
    kinematic_coordinates.emplace_back(w1, r, w2);
  }
  return kinematic_coordinates;
}

}  // namespace

template <typename T>
FilamentContactGeometryPair<T>::FilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A,
    std::vector<int> contact_edge_indexes_B,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_B)
    : id_A_(id_A),
      id_B_(id_B),
      p_WCs_(std::move(p_WCs)),
      nhats_BA_W_(std::move(nhats_BA_W)),
      signed_distances_(std::move(signed_distances)),
      contact_edge_indexes_A_(std::move(contact_edge_indexes_A)),
      kinematic_coordinates_A_(std::move(kinematic_coordinates_A)),
      contact_edge_indexes_B_(std::move(contact_edge_indexes_B)),
      kinematic_coordinates_B_(std::move(kinematic_coordinates_B)) {
  DRAKE_THROW_UNLESS(id_A.get_value() <= id_B.get_value());
  DRAKE_THROW_UNLESS(p_WCs_.size() == nhats_BA_W_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == signed_distances_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == contact_edge_indexes_A_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == kinematic_coordinates_A_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == contact_edge_indexes_B_->size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == kinematic_coordinates_B_->size());
  DRAKE_THROW_UNLESS(!p_WCs_.empty());
}

template <typename T>
FilamentContactGeometryPair<T>::FilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A)
    : id_A_(id_A),
      id_B_(id_B),
      p_WCs_(std::move(p_WCs)),
      nhats_BA_W_(std::move(nhats_BA_W)),
      signed_distances_(std::move(signed_distances)),
      contact_edge_indexes_A_(std::move(contact_edge_indexes_A)),
      kinematic_coordinates_A_(std::move(kinematic_coordinates_A)) {
  DRAKE_THROW_UNLESS(p_WCs_.size() == nhats_BA_W_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == signed_distances_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == contact_edge_indexes_A_.size());
  DRAKE_THROW_UNLESS(p_WCs_.size() == kinematic_coordinates_A_.size());
  DRAKE_THROW_UNLESS(!p_WCs_.empty());
}

template <typename T>
void FilamentContact<T>::AddFilamentFilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<int> contact_edge_indexes_B,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B) {
  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A.begin(),
                                    contact_edge_indexes_A.end());
  id_to_contact_edges_[id_B].insert(contact_edge_indexes_B.begin(),
                                    contact_edge_indexes_B.end());

  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A =
      ComputeKinematicCoordinates(p_WCs, contact_edge_indexes_A,
                                  node_positions_A);
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_B =
      ComputeKinematicCoordinates(p_WCs, contact_edge_indexes_B,
                                  node_positions_B);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_coordinates_A), std::move(contact_edge_indexes_B),
      std::move(kinematic_coordinates_B));
}

template <typename T>
void FilamentContact<T>::AddFilamentRigidContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A) {
  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A.begin(),
                                    contact_edge_indexes_A.end());

  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_coordinates_A =
      ComputeKinematicCoordinates(p_WCs, contact_edge_indexes_A,
                                  node_positions_A);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_coordinates_A));
}

template <typename T>
const std::unordered_set<int>& FilamentContact<T>::contact_edges(
    GeometryId id) const {
  const auto iter = id_to_contact_edges_.find(id);
  if (iter != id_to_contact_edges_.end()) {
    return iter->second;
  } else {
    static const std::unordered_set<int> empty_set;
    return empty_set;
  }
}

template class FilamentContactGeometryPair<double>;
template class FilamentContact<double>;

}  // namespace internal
}  // namespace geometry
}  // namespace drake
