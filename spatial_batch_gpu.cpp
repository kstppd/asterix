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
   if (nCells == 0) {
      return;
   }

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
   gpu_batch_allocate(nCells,0);
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

         vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         // Make sure local vectors are large enough
         size_t mySize = vmesh->size();
         SC->setReservation(popID,mySize);
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
         threadLargestVelMesh = threadLargestVelMesh > mySize ? threadLargestVelMesh : mySize;
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
   auto rule = [emptybucket, tombstone]
      __device__(const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval, vmesh::LocalID threshold) -> bool {
                  // This rule does not use the threshold value
                  return kval.first != emptybucket && kval.first != tombstone;
               };
   // Go via launcher due to templating
   extract_GIDs_kernel_launcher(
      dev_allMaps, // points to has_content maps
      dev_vbwcl_vec, // content list vectors, output value
      dev_contentSizes, // content list vector sizes, output value
      rule,
      dev_vmeshes, // rule_meshes, not used in this call
      dev_allMaps, // rule_maps, not used in this call
      dev_vbwcl_vec, // rule_vectors, not used in this call
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

}

void adjust_velocity_blocks_in_cells(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const vector<CellID>& cellsToAdjust,
   const uint popID,
   bool includeNeighbors
   ) {

   int adjustPreId {phiprof::initializeTimer("Adjusting blocks Pre")};
   int adjustId {phiprof::initializeTimer("Adjusting blocks")};
   int adjustPostId {phiprof::initializeTimer("Adjusting blocks Post")};
   const gpuStream_t baseStream = gpu_getStream();
   const uint nCells = cellsToAdjust.size();
            // If we are within an acceleration substep prior to the last one,
            // it's enough to adjust blocks based on local data only, and in
            // that case we simply pass an empty list of pointers.

   // Allocate buffers for GPU operations
   phiprof::Timer mallocTimer {"allocate buffers for content list analysis"};
   gpu_batch_allocate(nCells,0);
   mallocTimer.stop();

   size_t maxNeighbors = 0;
   if (includeNeighbors) {
      // Count maximum number of neighbors
#pragma omp parallel
      {
         size_t threadMaxNeighbors = 0;
#pragma omp for
         for (size_t i=0; i<cellsToAdjust.size(); ++i) {
            CellID cell_id = cellsToAdjust[i];
            SpatialCell* SC = mpiGrid[cell_id];
            if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
               continue;
            }
            uint reservationSize = SC->getReservation(popID);
            const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, NEAREST_NEIGHBORHOOD_ID);
            const uint nNeighbors = neighbors->size();
            threadMaxNeighbors = threadMaxNeighbors > nNeighbors ? threadMaxNeighbors : nNeighbors;
            for ( const auto& [neighbor_id, dir] : *neighbors) {
               reservationSize = (mpiGrid[neighbor_id]->velocity_block_with_content_list_size > reservationSize) ?
                  mpiGrid[neighbor_id]->velocity_block_with_content_list_size : reservationSize;
            }
            SC->setReservation(popID,reservationSize);
            SC->applyReservation(popID);
         }
#pragma omp critical
         {
            maxNeighbors = maxNeighbors > threadMaxNeighbors ? maxNeighbors : threadMaxNeighbors;
         }
      } // end parallel region
      gpu_batch_allocate(nCells,maxNeighbors);
   } // end if includeNeighbors

   size_t largestVelMesh = 0;
#pragma omp parallel
   {
      phiprof::Timer timer {adjustPreId};
      size_t threadLargestVelMesh = 0;
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<cellsToAdjust.size(); ++i) {
         CellID cell_id=cellsToAdjust[i];
         SpatialCell* SC = mpiGrid[cell_id];
         if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            host_vmeshes[i]=0;
            host_allMaps[i]=0;
            host_allMaps[nCells+i]=0;
            host_vbwcl_vec[i]=0;
            host_list_with_replace_new[i]=0;
            continue;
         }

         // Gather largest mesh size for launch parameters
         vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         threadLargestVelMesh = threadLargestVelMesh > vmesh->size() ? threadLargestVelMesh : vmesh->size();

         SC->density_pre_adjust=0.0;
         SC->density_post_adjust=0.0;
         if (includeNeighbors) {
            // gather vector with pointers to spatial neighbor lists
            const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, NEAREST_NEIGHBORHOOD_ID);
            // Note: at AMR refinement boundaries this can cause blocks to propagate further
            // than absolutely required. Face neighbors, however, are not enough as we must
            // account for diagonal propagation.
            const uint nNeighbors = neighbors->size();
            for (uint iN = 0; iN < maxNeighbors; ++iN) {
               if (iN >= nNeighbors) {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = 0; // no neighbor at this index
                  continue;
               }
               auto [neighbor_id, dir] = neighbors->at(iN);
               // store pointer to neighbor content list
               SpatialCell* NC = mpiGrid[neighbor_id];
               if (NC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = 0;
               } else {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = mpiGrid[neighbor_id]->dev_velocity_block_with_content_list;
               }
            }
         }

         // Store values and pointers
         host_vmeshes[i] = SC->dev_get_velocity_mesh(popID);
         host_allMaps[i] = SC->dev_velocity_block_with_content_map;
         host_allMaps[nCells+i] = SC->dev_velocity_block_with_no_content_map;
         host_vbwcl_vec[i] = SC->dev_velocity_block_with_content_list;
         host_list_with_replace_new[i] = SC->dev_list_with_replace_new;
         // host_list_delete[i] = SC->dev_list_delete;
         // host_list_to_replace[i] = SC->dev_list_to_replace;
         // host_list_with_replace_old[i] = SC->dev_list_with_replace_old;

         //GPUTODO make kernel. Or Perhaps gather total mass in content block gathering?
         if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
            for (size_t i=0; i<SC->get_number_of_velocity_blocks(popID)*WID3; ++i) {
               SC->density_pre_adjust += SC->get_data(popID)[i];
            }
         }
      }
      timer.stop();
#pragma omp critical
      {
         largestVelMesh = threadLargestVelMesh > largestVelMesh ? threadLargestVelMesh : largestVelMesh;
      }
   } // end parallel region

   /*
    * Perform block adjustment via batch operations
    * */

   phiprof::Timer copyTimer {"copy values to device"};
   // Copy pointers and counters over to device
   CHK_ERR( gpuMemcpyAsync(dev_allMaps, host_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vbwcl_vec, host_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vmeshes, host_vmeshes, nCells*sizeof(vmesh::VelocityMesh*), gpuMemcpyHostToDevice, baseStream) );
   if (includeNeighbors) {
      CHK_ERR( gpuMemcpyAsync(dev_vbwcl_neigh, host_vbwcl_neigh, nCells*maxNeighbors*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   }
   CHK_ERR( gpuMemsetAsync(dev_contentSizes, 0, nCells*sizeof(vmesh::LocalID), baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_list_with_replace_new, host_list_with_replace_new, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   //CHK_ERR( gpuMemcpyAsync(dev_VBCs, host_VBCs, nCells*sizeof(vmesh::VelocityBlockContainer*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   copyTimer.stop();

   // Evaluate velocity halo for local content blocks
   phiprof::Timer blockHaloTimer {"Block halo batch kernel"};
   const int addWidthV = getObjectWrapper().particleSpecies[popID].sparseBlockAddWidthV;
   if (addWidthV!=1) {
      std::cerr<<"Warning! "<<__FILE__<<":"<<__LINE__<<" Halo extent is not 1, unsupported size."<<std::endl;
   }
   // Halo of 1 in each direction adds up to 26 velocity neighbors.
   // For NVIDIA/CUDA, we dan do 26 neighbors and 32 threads per warp in a single block.
   // For AMD/HIP, we dan do 13 neighbors and 64 threads per warp in a single block, meaning two loops per cell.
   // In either case, we launch blocks equal to velocity_block_with_content_list_size
   dim3 grid_vel_halo(nCells,largestVelMesh,1);
      batch_update_velocity_halo_kernel<<<grid_vel_halo, 26*32, 0, baseStream>>> (
      dev_vmeshes,
      dev_vbwcl_vec,
      dev_allMaps
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   blockHaloTimer.stop();

   if (includeNeighbors) {
      phiprof::Timer neighHaloTimer {"Neighbor halo kernel"};
      // ceil int division
      const uint NeighLaunchBlocks = 1 + ((largestVelMesh - 1) / WARPSPERBLOCK);
      dim3 grid_neigh_halo(nCells,NeighLaunchBlocks,maxNeighbors);
      // For NVIDIA/CUDA, we dan do 32 neighbor GIDs and 32 threads per warp in a single block.
      // For AMD/HIP, we dan do 16 neighbor GIDs and 64 threads per warp in a single block
      // This is managed in-kernel.
      batch_update_neighbour_halo_kernel<<<grid_neigh_halo, WARPSPERBLOCK*GPUTHREADS, 0, baseStream>>> (
         dev_vmeshes,
         dev_allMaps,
         dev_vbwcl_neigh
         );
      CHK_ERR( gpuStreamSynchronize(baseStream) );
      neighHaloTimer.stop();
   }

   // Now extract vectors to be used in actual block adjustment
   // Previous kernels may have added dummy (to be added) entries to
   // velocity_block_with_content_map with LID=vmesh->invalidLocalID()
   // Or non-content blocks which should be retained (with correct LIDs).

   /** Rules used in extracting keys or elements from hashmaps
       Now these include passing pointers to GPU memory in order to evaluate
       nBlocksAfterAdjust without going via host. Pointers are copied by value.
   */
   phiprof::Timer extractKeysTimer {"extract content keys"};
   const vmesh::GlobalID emptybucket = host_allMaps[0]->expose_emptybucket();
   const vmesh::GlobalID tombstone   = host_allMaps[0]->expose_tombstone();
   const vmesh::GlobalID invalidGID  = host_vmeshes[0]->invalidGlobalID();
   const vmesh::LocalID  invalidLID  = host_vmeshes[0]->invalidLocalID();


// Required GIDs which do not yet exist in vmesh were stored in velocity_block_with_content_map with invalidLID
   auto rule_add = [emptybucket, tombstone, invalidGID, invalidLID]
      __device__(const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval, vmesh::LocalID threshold) -> bool {
                      // This rule does not use the threshold value
                      return kval.first != emptybucket &&
                         kval.first != tombstone &&
                         kval.first != invalidGID &&
                         kval.second == invalidLID; };

   // Go via launcher due to templating
   extract_GIDs_kernel_launcher(
      dev_allMaps, // uses first nCells entries, i.e. content blocks
      dev_list_with_replace_new, // output vecs
      dev_contentSizes, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      rule_add,
      dev_vmeshes, // rule_meshes, not used in this call
      dev_allMaps, // rule_maps, not used in this call
      dev_vbwcl_vec, // rule_vectors, not used in this call
      nCells,
      baseStream
      );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   extractKeysTimer.stop();

   // Remaining unconverted tasks of block adjustment
#pragma omp parallel
   {
      phiprof::Timer timer {adjustId};
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<cellsToAdjust.size(); ++i) {
         SpatialCell* cell = mpiGrid[cellsToAdjust[i]];
         if (cell->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }
         cell->adjust_velocity_blocks(popID, true, true); // true1 = doDeleteEmptyBlocks; true2 = batch mode
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
