#include "drake/geometry/meshcat.h"

using drake::geometry::Filament;
using drake::geometry::Meshcat;
using drake::geometry::Rgba;
using Eigen::Matrix3Xd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

void DrawTwistedBeam(Meshcat* meshcat) {
  const bool closed = false;
  const int num_nodes = 301;
  const int num_edges = num_nodes - 1;
  const double length = 1.0;  // m

  Matrix3Xd node_pos(3, num_nodes);
  node_pos.row(0) = VectorXd::LinSpaced(num_nodes, 0.0, length);
  node_pos.row(1) = VectorXd::Zero(num_nodes);
  node_pos.row(2) = VectorXd::Zero(num_nodes);

  auto theta = VectorXd::LinSpaced(num_edges, 0, 2 * M_PI).array();
  Matrix3Xd edge_m1(3, num_edges);
  edge_m1.row(0) = VectorXd::Zero(num_edges);
  edge_m1.row(1) = cos(theta);
  edge_m1.row(2) = sin(theta);

  Filament filament(
      closed, node_pos, edge_m1,
      Filament::CrossSection{
          .type = Filament::kRectangular, .width = 0.04, .height = 0.01});

  meshcat->SetObject("/main/twisted beam", filament, Rgba(0.7, 0.5, 0.4, 0.8));
}

void DrawCircularLoop(Meshcat* meshcat) {
  const bool closed = true;
  const int num_nodes = 301;
  const double loop_radius = 0.5;  // m

  auto theta =
      VectorXd::LinSpaced(num_nodes + 1, 0, 2 * M_PI).head(num_nodes).array();
  Matrix3Xd node_pos(3, num_nodes);
  node_pos.row(0) = loop_radius * cos(theta);
  node_pos.row(1) = loop_radius * sin(theta);
  node_pos.row(2) = VectorXd::Constant(num_nodes, loop_radius);

  Filament filament(
      closed, node_pos, Vector3d(0, 0, 1),
      Filament::CrossSection{
          .type = Filament::kElliptical, .width = 0.025, .height = 0.035});

  meshcat->SetObject("/main/circular loop", filament, Rgba(0.7, 0.5, 0.4, 0.8));
}

}  // namespace

int main(int argc, char* argv[]) {
  Meshcat meshcat;
  DrawTwistedBeam(&meshcat);
  DrawCircularLoop(&meshcat);
  while (true) {
  }
}
