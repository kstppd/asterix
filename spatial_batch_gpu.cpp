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

#include "spatial_batch_gpu.hpp"
#include "spatial_cell_kernels.hpp"
#include "arch/gpu_base.hpp"
#include "object_wrapper.h"
#include "velocity_mesh_parameters.h"

using namespace std;

namespace spatial_cell {

/** Bulk call over listed cells of spatial grid
    Prepares the content / no-content velocity block lists
    for all requested cells, for the requested popID
**/
void update_velocity_block_content_lists(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const vector<CellID>& cells,
   const uint popID) {

   const uint nCells = cells.size();
#ifdef DEBUG_SPATIAL_CELL
   if (popID >= populations.size()) {
      std::cerr << "ERROR, popID " << popID << " exceeds populations.size() " << populations.size() << " in ";
      std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
      exit(1);
   }
#endif

   const gpuStream_t baseStream = gpu_getStream();

   // Allocate buffers for GPU operations
   phiprof::Timer mallocTimer {"allocate buffers for content list analysis"};
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** host_allMaps, **dev_allMaps;
   split::SplitVector<vmesh::GlobalID> ** host_vbwcl_vec, **dev_vbwcl_vec;
   Real* host_minValues, *dev_minValues;
   vmesh::VelocityMesh** host_vmeshes, **dev_vmeshes;
   vmesh::VelocityBlockContainer** host_VBCs, **dev_VBCs;
   vmesh::LocalID* host_contentSizes, *dev_contentSizes;
   CHK_ERR( gpuMallocHost((void**)&host_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*)) );
   CHK_ERR( gpuMallocHost((void**)&host_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*)) );
   CHK_ERR( gpuMallocHost((void**)&host_minValues, nCells*sizeof(Real)) );
   CHK_ERR( gpuMallocHost((void**)&host_vmeshes,nCells*sizeof(vmesh::VelocityMesh*)) );
   CHK_ERR( gpuMallocHost((void**)&host_VBCs,nCells*sizeof(vmesh::VelocityBlockContainer*)) );
   CHK_ERR( gpuMallocHost((void**)&host_contentSizes,nCells*sizeof(vmesh::LocalID)) );
   CHK_ERR( gpuMallocAsync((void**)&dev_contentSizes,nCells*sizeof(vmesh::LocalID),baseStream) );
   CHK_ERR( gpuMallocAsync((void**)&dev_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*),baseStream) );
   CHK_ERR( gpuMallocAsync((void**)&dev_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*),baseStream) );
   CHK_ERR( gpuMallocAsync((void**)&dev_minValues,nCells*sizeof(Real),baseStream) );
   CHK_ERR( gpuMallocAsync((void**)&dev_vmeshes,nCells*sizeof(vmesh::VelocityMesh*),baseStream) );
   CHK_ERR( gpuMallocAsync((void**)&dev_VBCs,nCells*sizeof(vmesh::VelocityBlockContainer*),baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   mallocTimer.stop();

   phiprof::Timer sparsityTimer {"update Sparsity values"};
   size_t largestSizePower = 0;
   size_t largestVelMesh = 0;
#pragma omp parallel
   {
      size_t threadLargestVelMesh = 0;
      size_t threadLargestSizePower = 0;
      SpatialCell *SC;
#pragma omp for
      for (uint i=0; i<nCells; ++i) {
         SC = mpiGrid[cells[i]];
         SC->velocity_block_with_content_list_size = 0;
         SC->updateSparseMinValue(popID);
         SC->applyReservation(popID);
         // might be better to apply reservation *after* clearing maps, but pointers might change.
         // Store values and pointers
         host_vmeshes[i] = SC->dev_get_velocity_mesh(popID);
         host_VBCs[i] = SC->dev_get_velocity_blocks(popID);
         host_minValues[i] = SC->getVelocityBlockMinValue(popID);
         host_allMaps[i] = SC->dev_velocity_block_with_content_map;
         host_allMaps[nCells+i] = SC->dev_velocity_block_with_no_content_map;
         host_vbwcl_vec[i] = SC->dev_velocity_block_with_content_list;

         // Gather largest values
         vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         threadLargestVelMesh = threadLargestVelMesh > vmesh->size() ? threadLargestVelMesh : vmesh->size();
         threadLargestSizePower = threadLargestSizePower > SC->vbwcl_sizePower ? threadLargestSizePower : SC->vbwcl_sizePower;
         threadLargestSizePower = threadLargestSizePower > SC->vbwncl_sizePower ? threadLargestSizePower : SC->vbwncl_sizePower;
      }
#pragma omp critical
      {
         largestVelMesh = threadLargestVelMesh > largestVelMesh ? threadLargestVelMesh : largestVelMesh;
         largestSizePower = threadLargestSizePower > largestSizePower ? threadLargestSizePower : largestSizePower;
      }
   }
   // Expose map GID identifiers
   const vmesh::GlobalID emptybucket = host_allMaps[0]->expose_emptybucket();
   const vmesh::GlobalID tombstone = host_allMaps[0]->expose_tombstone();
   sparsityTimer.stop();

   phiprof::Timer copyTimer {"copy values to device"};
   // Copy pointers and counters over to device
   CHK_ERR( gpuMemcpyAsync(dev_allMaps, host_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vbwcl_vec, host_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_minValues, host_minValues, nCells*sizeof(Real), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vmeshes, host_vmeshes, nCells*sizeof(vmesh::VelocityMesh*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_VBCs, host_VBCs, nCells*sizeof(vmesh::VelocityBlockContainer*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   copyTimer.stop();

   // Batch clear all hash maps
   phiprof::Timer clearTimer {"clear all content maps"};
   const size_t largestMapSize = std::pow(2,largestSizePower);
   // fast ceil for positive ints
   const size_t blocksNeeded = largestMapSize / Hashinator::defaults::MAX_BLOCKSIZE + (largestMapSize % Hashinator::defaults::MAX_BLOCKSIZE != 0);
   dim3 grid1(2*nCells,blocksNeeded,1);
   batch_reset_all_to_empty<<<grid1, Hashinator::defaults::MAX_BLOCKSIZE, 0, baseStream>>>(
      dev_allMaps,
      emptybucket
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   clearTimer.stop();

   // Batch gather GID-LID-pairs into two maps (one with content, one without)
   phiprof::Timer blockKernelTimer {"update content lists kernel"};
   const uint vlasiBlocksPerWorkUnit = 1;
   // ceil int division
   const uint launchBlocks = 1 + ((largestVelMesh - 1) / vlasiBlocksPerWorkUnit);
   dim3 grid2(nCells,launchBlocks,1);
   batch_update_velocity_block_content_lists_kernel<<<grid2, (vlasiBlocksPerWorkUnit * WID3), 0, baseStream>>> (
      dev_vmeshes,
      dev_VBCs,
      dev_allMaps,
      dev_minValues
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   blockKernelTimer.stop();

   // Extract all keys from content maps into content list
   phiprof::Timer extractKeysTimer {"extract content keys"};
   auto rule = [emptybucket, tombstone] __host__ __device__(const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval) -> bool {
                  return kval.first != emptybucket && kval.first != tombstone;
               };
   // Go via launcher due to templating
   extract_all_content_blocks_launcher(
      dev_allMaps,
      dev_vbwcl_vec,
      dev_contentSizes,
      rule,
      nCells,
      baseStream
      );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   extractKeysTimer.stop();

   // Update host-side size values
   phiprof::Timer blocklistTimer {"update content lists extract"};
   CHK_ERR( gpuMemcpyAsync(host_contentSizes, dev_contentSizes, nCells*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
#pragma omp parallel for
   for (uint i=0; i<nCells; ++i) {
      mpiGrid[cells[i]]->velocity_block_with_content_list_size = host_contentSizes[i];
   }
   blocklistTimer.stop();

   CHK_ERR( gpuFreeHost(host_allMaps));
   CHK_ERR( gpuFreeHost(host_vbwcl_vec));
   CHK_ERR( gpuFreeHost(host_minValues));
   CHK_ERR( gpuFreeHost(host_vmeshes));
   CHK_ERR( gpuFreeHost(host_VBCs));
   CHK_ERR( gpuFreeHost(host_contentSizes));
   CHK_ERR( gpuFree(dev_contentSizes));
   CHK_ERR( gpuFree(dev_allMaps));
   CHK_ERR( gpuFree(dev_vbwcl_vec));
   CHK_ERR( gpuFree(dev_minValues));
   CHK_ERR( gpuFree(dev_vmeshes));
   CHK_ERR( gpuFree(dev_VBCs));

}

void adjust_velocity_blocks_in_cells(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const vector<CellID>& cellsToAdjust,
   const uint popID,
   bool includeNeighbours
   ) {

   int adjustPreId {phiprof::initializeTimer("Adjusting blocks Pre")};
   int adjustId {phiprof::initializeTimer("Adjusting blocks")};
   int adjustPostId {phiprof::initializeTimer("Adjusting blocks Post")};

#pragma omp parallel
   {
      phiprof::Timer timer {adjustPreId};
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<cellsToAdjust.size(); ++i) {
         CellID cell_id=cellsToAdjust[i];
         SpatialCell* cell = mpiGrid[cell_id];
         if (cell->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }
         cell->neighbor_ptrs.clear();
         cell->density_pre_adjust=0.0;
         cell->density_post_adjust=0.0;
         if (includeNeighbours) {
            // gather spatial neighbor list and gather vector with pointers to cells
            // If we are within an acceleration substep prior to the last one,
            // it's enough to adjust blocks based on local data only, and in
            // that case we simply pass an empty list of pointers.
            const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, NEAREST_NEIGHBORHOOD_ID);
            // Note: at AMR refinement boundaries this can cause blocks to propagate further
            // than absolutely required. Face neighbours, however, are not enough as we must
            // account for diagonal propagation.
            cell->neighbor_ptrs.reserve(neighbors->size());
            uint reservationSize = cell->getReservation(popID);
            for ( const auto& [neighbor_id, dir] : *neighbors) {
               if ((neighbor_id != 0) && (neighbor_id != cell_id)) {
                  cell->neighbor_ptrs.push_back(mpiGrid[neighbor_id]);
               }
               // Ensure cell has sufficient reservation
               reservationSize = (mpiGrid[neighbor_id]->velocity_block_with_content_list_size > reservationSize) ?
                  mpiGrid[neighbor_id]->velocity_block_with_content_list_size : reservationSize;
            }
            cell->setReservation(popID,reservationSize);
         }

         //GPUTODO make kernel. Or Perhaps gather total mass in content block gathering?
         if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
            for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
               cell->density_pre_adjust += cell->get_data(popID)[i];
            }
         }
      }
      timer.stop();
   } // end parallel region

   // Actual block adjustment
#pragma omp parallel
   {
      phiprof::Timer timer {adjustId};
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<cellsToAdjust.size(); ++i) {
         SpatialCell* cell = mpiGrid[cellsToAdjust[i]];
         if (cell->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }
         cell->adjust_velocity_blocks(popID);
      }
      timer.stop();
   } // end parallel region

   if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
#pragma omp parallel
      {
         phiprof::Timer timer {adjustPostId};
#pragma omp for schedule(dynamic,1)
         for (size_t i=0; i<cellsToAdjust.size(); ++i) {
            SpatialCell* cell = mpiGrid[cellsToAdjust[i]];
            if (cell->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
               continue;
            }
            for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
               cell->density_post_adjust += cell->get_data(popID)[i];
            }
            if (cell->density_post_adjust != 0.0) {
               // GPUTODO use population scaling function here
               for (size_t i=0; i<cell->get_number_of_velocity_blocks(popID)*WID3; ++i) {
                  cell->get_data(popID)[i] *= cell->density_pre_adjust/cell->density_post_adjust;
               }
            }
         } // end cell loop
         timer.stop();
      } // end parallel region
   } // end if conserve mass
}

} // namespace
