#pragma once

#include "drake/common/type_safe_index.h"

namespace drake {
namespace multibody {
namespace der {

/** Type used to index DER nodes. */
using DerNodeIndex = TypeSafeIndex<class DerNodeTag>;

/** Type used to index DER edges. */
using DerEdgeIndex = TypeSafeIndex<class DerEdgeTag>;

}  // namespace der
}  // namespace multibody
}  // namespace drake
