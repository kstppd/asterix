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

#ifndef VLASIATOR_SPATIAL_CELL_KERNELS_HPP
#define VLASIATOR_SPATIAL_CELL_KERNELS_HPP

/** GPU kernel for identifying which blocks have relevant content */
__global__ void batch_update_velocity_block_content_lists_kernel (
   vmesh::VelocityMesh **vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** allMaps,
   Real* velocity_block_min_values
   ) {
   // launch griddim3 grid(nCells,launchBlocks,1);
   const uint nCells = gridDim.x;
   const int cellIndex = blockIdx.x;
   const int blocki = blockIdx.y;
   const uint ti = threadIdx.x;

   vmesh::VelocityMesh* vmesh = vmeshes[cellIndex];
   vmesh::VelocityBlockContainer* blockContainer = blockContainers[cellIndex];
   Real velocity_block_min_value = velocity_block_min_values[cellIndex];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwcl_map = allMaps[cellIndex];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwncl_map = allMaps[nCells+cellIndex];

   // Each GPU block / workunit can theoretically manage several Vlasiator velocity blocks at once.
   const uint vlasiBlocksPerWorkUnit = 1;
   const uint workUnitIndex = 0; // [0,vlasiBlocksPerWorkUnit)
   // const uint vlasiBlocksPerWorkUnit = WARPSPERBLOCK * GPUTHREADS / WID3;
   // const uint workUnitIndex = ti / WID3; // [0,vlasiBlocksPerWorkUnit)
   const uint b_tid = ti % WID3; // [0,WID3)
   const uint blockLID = blocki * vlasiBlocksPerWorkUnit + workUnitIndex; // [0,nBlocksToChange)

   __shared__ int has_content[WARPSPERBLOCK * GPUTHREADS];
   const uint nBlocks = vmesh->size();
   //for (uint blockLID=blocki; blockLID<nBlocks; blockLID += gpuBlocks) {
   if (blockLID < nBlocks) {
      const vmesh::GlobalID blockGID = vmesh->getGlobalID(blockLID);
#ifdef DEBUG_SPATIAL_CELL
      if (blockGID == vmesh->invalidGlobalID()) {
         if (b_tid==0) printf("Invalid GID encountered in update_velocity_block_content_lists_kernel!\n");
         return;
      }
      if (blockLID == vmesh->invalidLocalID()) {
         if (b_tid==0) printf("Invalid LID encountered in update_velocity_block_content_lists_kernel!\n");
         return;
      }
#endif
      // Check each velocity cell if it is above the threshold
      const Realf* avgs = blockContainer->getData(blockLID);
      has_content[ti] = avgs[b_tid] >= velocity_block_min_value ? 1 : 0;
      __syncthreads(); // THIS SYNC IS CRUCIAL!
      // Implemented just a simple non-optimized thread OR
      // GPUTODO reductions via warp voting
      for (unsigned int s=WID3/2; s>0; s>>=1) {
         if (b_tid < s) {
            has_content[ti] = has_content[ti] || has_content[ti + s];
         }
         __syncthreads();
      }
      // Insert into map only from threads 0...WARPSIZE
      if (b_tid < GPUTHREADS) {
         if (has_content[0]) {
            vbwcl_map->warpInsert(blockGID,blockLID,b_tid);
         } else {
            vbwncl_map->warpInsert(blockGID,blockLID,b_tid);
         }
      }
      __syncthreads();
   }
}

/*
 * Resets all elements in all provided hashmaps to EMPTY, VAL_TYPE()
 */
__global__ void batch_reset_all_to_empty(
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>**maps,
   const vmesh::GlobalID emptybucket
   ) {
   //launch parameters: dim3 grid(nMaps,blocksNeeded,1);
   const size_t hashmapIndex = blockIdx.x;
   const size_t tid = threadIdx.x + blockIdx.y * blockDim.x;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = maps[hashmapIndex];
   const size_t len = thisMap->bucket_count();
   Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>* dst = thisMap->expose_bucketdata<false>();

   // Early exit here
   if (tid >= len) {
      return;
   }
   if (dst[tid].first != emptybucket) {
      dst[tid].first = emptybucket;
   }

   //Thread 0 resets fill
   if (tid==0) {
      Hashinator::Info *info = thisMap->expose_mapinfo<false>();
      info->fill=0;
   }
   return;
}

/*
 * Extracts keys from all provided hashmaps to provided splitvectors, and stores the vector size in an array.
 */
template <typename Rule>
__global__ void extract_all_content_blocks(
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>**maps, //we're going to access only every second entry (content, not no-content)
   split::SplitVector<vmesh::GlobalID> **outputVecs,
   vmesh::LocalID* dev_contentSizes,
   Rule rule
   ) {
   //launch parameters: dim3 grid(nMaps,1,1); // As this is a looping reduction
   const size_t hashmapIndex = blockIdx.x;
   if (maps[hashmapIndex]==0) {
      return; // Early return for invalid cells
   }
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = maps[hashmapIndex];
   split::SplitVector<vmesh::GlobalID> *outputVec = outputVecs[hashmapIndex];

   // This must be equal to at least both WARPLENGTH and MAX_BLOCKSIZE/WARPLENGTH
   __shared__ uint32_t warpSums[WARPLENGTH];
   __shared__ uint32_t outputCount;
   // blockIdx.x is always 0 for this kernel
   const size_t tid = threadIdx.x; // + blockIdx.x * blockDim.x;
   const size_t wid = tid / WARPLENGTH;
   const size_t w_tid = tid % WARPLENGTH;
   //const uint warpsPerBlock = BLOCKSIZE / WARPLENGTH;
   const uint warpsPerBlock = blockDim.x / WARPLENGTH;
   // zero init shared buffer
   if (wid == 0) {
      warpSums[w_tid] = 0;
   }
   __syncthreads();
   // full warp votes for rule-> mask = [01010101010101010101010101010101]
   int64_t remaining = thisMap->bucket_count();
   const uint capacity = outputVec->capacity();
   uint32_t outputSize = 0;
   // Initial pointers into data
   Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID> *input = thisMap->expose_bucketdata<false>();
   vmesh::GlobalID* output = outputVec->data();
   // Start loop
   while (remaining > 0) {
      int current = remaining > blockDim.x ? blockDim.x : remaining;
      __syncthreads();
      const int active = (tid < current) ? rule(input[tid]) : false;
      //const int active = (tid < current) ? (input[tid].first != tombstone && input[tid].first != emptybucket) : false;
      const auto mask = split::s_warpVote(active == 1, SPLIT_VOTING_MASK);
      const auto warpCount = split::s_pop_count(mask);
      if (w_tid == 0) {
         warpSums[wid] = warpCount;
      }
      __syncthreads();
      // Figure out the total here because we overwrite shared mem later
      if (wid == 0) {
         // ceil int division
         int activeWARPS = nextPow2(1 + ((current - 1) / WARPLENGTH));
         auto reduceCounts = [activeWARPS](int localCount) -> int {
                                for (int i = activeWARPS / 2; i > 0; i = i / 2) {
                                   localCount += split::s_shuffle_down(localCount, i, SPLIT_VOTING_MASK);
                                }
                                return localCount;
                             };
         auto localCount = warpSums[w_tid];
         int totalCount = reduceCounts(localCount);
         if (w_tid == 0) {
            outputCount = totalCount;
            outputSize += totalCount;
            assert((outputSize <= capacity) && "loop_compact ran out of capacity!");
            outputVec->device_resize(outputSize);
         }
      }
      // Prefix scan WarpSums on the first warp
      if (wid == 0) {
         auto value = warpSums[w_tid];
         for (int d = 1; d < warpsPerBlock; d = 2 * d) {
            int res = split::s_shuffle_up(value, d, SPLIT_VOTING_MASK);
            if (tid % warpsPerBlock >= d) {
               value += res;
            }
         }
         warpSums[w_tid] = value;
      }
      __syncthreads();
      auto offset = (wid == 0) ? 0 : warpSums[wid - 1];
      auto pp = split::s_pop_count(mask & ((ONE << w_tid) - ONE));
      const auto warpTidWriteIndex = offset + pp;
      if (active) {
         output[warpTidWriteIndex] = input[tid].first;
      }
      // Next loop iteration:
      input += current;
      output += outputCount;
      remaining -= current;
   }
   __syncthreads();
   if (tid == 0) {
      // Resize to final correct output size.
      outputVec->device_resize(outputSize);
      dev_contentSizes[hashmapIndex] = outputSize;
   }
}

template <typename Rule>
void extract_all_content_blocks_launcher(
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** dev_allMaps,
   split::SplitVector<vmesh::GlobalID> **dev_vbwcl_vec,
   vmesh::LocalID* dev_contentSizes,
   Rule rule,
   const uint nCells,
   gpuStream_t stream
   ) {
   extract_all_content_blocks<<<nCells, Hashinator::defaults::MAX_BLOCKSIZE, 0, stream>>>(
      dev_allMaps,
      dev_vbwcl_vec,
      dev_contentSizes,
      rule
      );
   CHK_ERR( gpuPeekAtLastError() );
}

/** Gpu Kernel to quickly gather the v-space halo of local content blocks
    Halo of 1 in each direction adds up to 26 neighbours.
    For NVIDIA/CUDA, we dan do 26 neighbours and 32 threads per warp in a single block.
    For AMD/HIP, we dan do 13 neighbours and 64 threads per warp in a single block, meaning two loops per cell.
    In either case, we launch blocks equal to nCells * max_velocity_block_with_content_list_size
*/
__global__ void batch_update_velocity_halo_kernel (
   vmesh::VelocityMesh **vmeshes,
   split::SplitVector<vmesh::GlobalID> **velocity_block_with_content_lists,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** allMaps
   ) {
   // launch grid dim3 grid(nCells,launchBlocks,1);
   const uint nCells = gridDim.x;
   const int cellIndex = blockIdx.x;
   //const int gpuBlocks = gridDim.y; // At least VB with content list size
   const int blocki = blockIdx.y;
   //const int blockSize = blockDim.x; // should be 26*32 or 13*64
   const uint ti = threadIdx.x;

   // Cells such as DO_NOT_COMPUTE are identified with a zero in the vmeshes pointer buffer
   if (vmeshes[cellIndex] == 0) {
      return;
   }
   vmesh::VelocityMesh* vmesh = vmeshes[cellIndex];
   split::SplitVector<vmesh::GlobalID> *velocity_block_with_content_list = velocity_block_with_content_lists[cellIndex];
   vmesh::GlobalID* velocity_block_with_content_list_data = velocity_block_with_content_list->data();
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwcl_map = allMaps[cellIndex];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwncl_map = allMaps[nCells+cellIndex];

   if (blocki >= velocity_block_with_content_list->size()) {
      return;
   } // Early return if we are beyond the size of the list for this cell

   const int offsetIndex1 = ti / GPUTHREADS; // [0,26) (NVIDIA) or [0,13) (AMD)
   const int w_tid = ti % GPUTHREADS; // [0,WARPSIZE)
   // Assumes addWidthV = 1
   #ifdef __CUDACC__
   const int max_i=1;
   #endif
   #ifdef __HIP_PLATFORM_HCC___
   const int max_i=2;
   #endif
   for (int i=0; i<max_i; i++) {
      int offsetIndex = offsetIndex1 + 13*i;
      // nudge latter half in order to exclude self
      if (offsetIndex > 12) {
         offsetIndex++;
      }
      const int offset_vx = (offsetIndex % 3) - 1;
      const int offset_vy = ((offsetIndex / 3) % 3) - 1;
      const int offset_vz = (offsetIndex / 9) - 1;
      // Offsets verified in python
      const vmesh::GlobalID GID = velocity_block_with_content_list_data[blocki];
      vmesh::LocalID ind0,ind1,ind2;
      vmesh->getIndices(GID,ind0,ind1,ind2);
      const int nind0 = ind0 + offset_vx;
      const int nind1 = ind1 + offset_vy;
      const int nind2 = ind2 + offset_vz;
      const vmesh::GlobalID nGID
         = vmesh->getGlobalID(nind0,nind1,nind2);
      // Does block already exist in mesh?
      const vmesh::LocalID LID = vmesh->warpGetLocalID(nGID, w_tid);
      // Try adding this nGID to velocity_block_with_content_map. If it exists, do not overwrite.
      const bool newlyadded = vbwcl_map->warpInsert_V<true>(nGID,LID, w_tid);
      if (newlyadded) {
         // Block did not previously exist in velocity_block_with_content_map
         if ( LID != vmesh->invalidLocalID()) {
            // Block exists in mesh, ensure it won't get deleted:
            // try deleting from no_content map
            vbwncl_map->warpErase(nGID, w_tid);
         }
         // else:
         // Block does not yet exist in mesh at all. Needs adding!
         // Identified as invalidLID entries in velocity_block_with_content_map.
      }
      __syncthreads();
   }
}

/** Gpu Kernel to quickly gather the spatial halo of neighbour content blocks
*/
__global__ void batch_update_neighbour_halo_kernel (
   vmesh::VelocityMesh **vmeshes,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** allMaps,
   split::SplitVector<vmesh::GlobalID> **neigh_velocity_block_with_content_lists
   ) {

   // const uint NeighLaunchBlocks = 1 + ((largestVelMesh - 1) / WARPSPERBLOCK);
   // launch grid dim3 grid_neigh_halo(nCells,NeighLaunchBlocks,maxNeighbours);

   const uint nCells = gridDim.x;
   const uint maxNeighbours = gridDim.z;
   const int cellIndex = blockIdx.x;
   const int neighIndex = blockIdx.x * maxNeighbours + blockIdx.z;

   //const int blockSize = blockDim.x; // should be 32*32 or 16*64
   //const int neighBlocks = gridDim.y; // Equal to count of neighbour content blocks divided by (warps/block)
   const int ti = threadIdx.x; // [0,blockSize)
   const int w_tid = ti % GPUTHREADS; // [0,WARPSIZE)
   const int myindex = blockIdx.y * WARPSPERBLOCK + ti / GPUTHREADS;

   // Cells such as DO_NOT_COMPUTE are identified with a zero in the vmeshes pointer buffer
   if (vmeshes[cellIndex] == 0) {
      return;
   }
   // Early return for non-existing neighbour indexes
   if (neigh_velocity_block_with_content_lists[neighIndex] == 0) {
      return;
   }

   split::SplitVector<vmesh::GlobalID> *velocity_block_with_content_list = neigh_velocity_block_with_content_lists[neighIndex];
   const uint len = velocity_block_with_content_list->size();
   if (myindex >= len) {
      return;
   } // Early return if we are beyond the size of the list for this cell

   vmesh::VelocityMesh* vmesh = vmeshes[cellIndex];
   vmesh::GlobalID* velocity_block_with_content_list_data = velocity_block_with_content_list->data();
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwcl_map = allMaps[cellIndex];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwncl_map = allMaps[nCells+cellIndex];

   const vmesh::GlobalID nGID = velocity_block_with_content_list_data[myindex];
   // Does block already exist in mesh?
   const vmesh::LocalID LID = vmesh->warpGetLocalID(nGID, w_tid);
   // Try adding this nGID to velocity_block_with_content_map. If it exists, do not overwrite.
   const bool newlyadded = vbwcl_map->warpInsert_V<true>(nGID,LID, w_tid);
   if (newlyadded) {
      // Block did not previously exist in velocity_block_with_content_map
      if ( LID != vmesh->invalidLocalID()) {
         // Block exists in mesh, ensure it won't get deleted:
         // try deleting from no_content map
         vbwncl_map->warpErase(nGID, w_tid);
      }
      // else:
      // Block does not yet exist in mesh at all. Needs adding!
      // Identified as invalidLID entries in velocity_block_with_content_map.
   }
}



#endif
