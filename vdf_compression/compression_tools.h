#pragma once
/*
 * This file is part of Vlasiator.
 * Copyright 2010-2024 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

// These tools  are fwd declared here and implemented at the end of the file for
// better clarity. They are not for external usage and as such they do not go
// into the header file

#include "../definitions.h"
#include "../logger.h"
#include "../mpiconversion.h"
#include "../object_wrapper.h"
#include "../spatial_cell_wrapper.hpp"
#include "../velocity_blocks.h"
#include "stdlib.h"
#include <array>

namespace ASTERIX {
struct VCoords {
   Real vx, vy, vz;
};

struct OrderedVDF {
   std::vector<Realf> vdf_vals;
   std::array<Real, 6> v_limits;     // vx_min,vy_min,vz_min,vx_max,vy_max,vz_max
   std::array<std::size_t, 3> shape; // x,y,z
   std::size_t index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
      return i * (shape[1] * shape[2]) + j * shape[0] + k;
   }

   Realf& at(std::size_t i, std::size_t j, std::size_t k) noexcept { return vdf_vals.at(index(i, j, k)); }

   const Realf& at(std::size_t i, std::size_t j, std::size_t k) const noexcept { return vdf_vals.at(index(i, j, k)); }
};

struct UnorderedVDF {
   std::vector<Realf> vdf_vals;
   std::vector<std::array<Real, 3>> vdf_coords;
   std::array<Real, 6> v_limits; // vx_min,vy_min,vz_min,vx_max,vy_max,vz_max
};

auto overwrite_pop_spatial_cell_vdf(SpatialCell* sc, uint popID, const std::vector<Realf>& new_vspace) -> void;

auto extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID) -> UnorderedVDF;

auto extract_union_pop_vdfs_from_cids(const std::vector<CellID>& cids, uint popID,
                                      const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                      std::vector<std::array<Real, 3>>& vcoords, std::vector<Realf>& vspace)
    -> std::tuple<std::size_t, std::array<Real, 6>, std::unordered_map<vmesh::LocalID, std::size_t>>;


auto extract_pop_vdf_from_spatial_cell_ordered_min_bbox_zoomed(SpatialCell* sc, uint popID, int zoom)->OrderedVDF;

constexpr auto isPow2(std::unsigned_integral auto val) -> bool { return (val & (val - 1)) == 0; };

auto overwrite_cellids_vdfs(const std::vector<CellID>& cids, uint popID,
                            dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                            const std::vector<std::array<Real, 3>>& vcoords, const std::vector<Realf>& vspace_union,
                            const std::unordered_map<vmesh::LocalID, std::size_t>& map_exists_id) -> void;

} // namespace ASTERIX
