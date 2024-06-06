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

#ifdef DEBUG_SPATIAL_BATCH
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
      }
   }
}
