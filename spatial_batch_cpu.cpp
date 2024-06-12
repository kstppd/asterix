/*
 * This file is part of Vlasiator.
 * Copyright 2010-2024 Finnish Meteorological Institute and University of Helsinki
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

#include "spatial_cell_cpu.hpp"
#include "spatial_batch_cpu.hpp"

namespace spatial_cell {
/** Bulk call over listed cells of spatial grid
    Prepares the content / no-content velocity block lists
    for all requested cells, for the requested popID
**/
   void update_velocity_block_content_lists(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const vector<CellID>& cells,
      const uint popID=0) {

#ifdef DEBUG_SPATIAL_CELL
      if (popID >= populations.size()) {
         std::cerr << "ERROR, popID " << popID << " exceeds populations.size() " << populations.size() << " in ";
         std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
         exit(1);
      }
#endif

      int computeId {phiprof::initializeTimer("Compute with_content_list")};
#pragma omp parallel
      {
         phiprof::Timer timer {computeId};
#pragma omp for schedule(dynamic,1)
         for (uint i=0; i<cells.size(); ++i) {
            mpiGrid[cells[i]]->updateSparseMinValue(popID);
            mpiGrid[cells[i]]->update_velocity_block_content_lists(popID);
         }
         timer.stop();
      } // end parallel region
   }

   void adjust_velocity_blocks_in_cells(
      dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const vector<CellID>& cellsToAdjust,
      const uint popID=0,
      bool includeNeighbours=true
      ) {
      int adjustId {phiprof::initializeTimer("Adjusting blocks")};
#pragma omp parallel
      {
         phiprof::Timer timer {adjustId};
#pragma omp for schedule(dynamic,1)
         for (size_t i=0; i<cellsToAdjust.size(); ++i) {
            Real density_pre_adjust=0.0;
            Real density_post_adjust=0.0;
            CellID cell_id=cellsToAdjust[i];
            SpatialCell* cell = mpiGrid[cell_id];
            vector<SpatialCell*> neighbor_ptrs;
            if (includeNeighbours) {
               // gather spatial neighbor list and gather vector with pointers to cells
               // If we are within an acceleration substep prior to the last one,
               // it's enough to adjust blocks based on local data only, and in
               // that case we simply pass an empty list of pointers.
               const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, NEAREST_NEIGHBORHOOD_ID);
               // Note: at AMR refinement boundaries this can cause blocks to propagate further
               // than absolutely required. Face neighbours, however, are not enough as we must
               // account for diagonal propagation.
               neighbor_ptrs.reserve(neighbors->size());
               uint reservationSize = cell->getReservation(popID);
               for ( const auto& [neighbor_id, dir] : *neighbors) {
                  if ((neighbor_id != 0) && (neighbor_id != cell_id)) {
                     neighbor_ptrs.push_back(mpiGrid[neighbor_id]);
                  }
                  // Ensure cell has sufficient reservation
                  reservationSize = (mpiGrid[neighbor_id]->velocity_block_with_content_list_size > reservationSize) ?
                     mpiGrid[neighbor_id]->velocity_block_with_content_list_size : reservationSize;
               }
               cell->setReservation(popID,reservationSize);
            }

            if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
               for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
                  density_pre_adjust += cell->get_data(popID)[i];
               }
            }

            cell->adjust_velocity_blocks(neighbor_ptrs,popID);

            if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
               for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
                  density_post_adjust += cell->get_data(popID)[i];
               }
               if (density_post_adjust != 0.0) {
                  for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
                     cell->get_data(popID)[i] *= density_pre_adjust/density_post_adjust;
                  }
               }
            }
         }
         timer.stop();
      } // end parallel region
   }

} //namespace
