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

#ifndef VLASIATOR_SPATIAL_BATCH_KERNELS_HPP
#define VLASIATOR_SPATIAL_BATCH_KERNELS_HPP

#ifdef USE_WARPACCESSORS
 #define USE_BATCH_WARPACCESSORS
#endif
/** GPU kernel for identifying which blocks have relevant content */
__global__ void batch_update_velocity_block_content_lists_kernel (
   vmesh::VelocityMesh **vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** allMaps,
   Real* velocity_block_min_values
   ) {
   // launch griddim3 grid(launchBlocks,nCells,1);
   const uint nCells = gridDim.y;
   const int cellIndex = blockIdx.y;
   const int blocki = blockIdx.x;
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
   if (blockLID < nBlocks) {
      const vmesh::GlobalID blockGID = vmesh->getGlobalID(blockLID);
#ifdef DEBUG_SPATIAL_CELL
      if (blockGID == vmesh->invalidGlobalID()) {
         if (b_tid==0) printf("Invalid GID encountered in batch_update_velocity_block_content_lists_kernel!\n");
         return;
      }
      if (blockLID == vmesh->invalidLocalID()) {
         if (b_tid==0) printf("Invalid LID encountered in batch_update_velocity_block_content_lists_kernel!\n");
         return;
      }
#endif
      // Check each velocity cell if it is above the threshold
      const Realf* avgs = blockContainer->getData(blockLID);
      has_content[ti] = avgs[b_tid] >= velocity_block_min_value ? 1 : 0;
      __syncthreads(); // THIS SYNC IS CRUCIAL!
      // Implemented just a simple non-optimized thread OR
      // GPUTODO reductions via warp voting

      // Perform loop only until first value fulfills condition
      for (unsigned int s=WID3/2; s>0; s>>=1) {
         if (has_content[0]) {
            break;
         }
         if (b_tid < s) {
            has_content[ti] = has_content[ti] || has_content[ti + s];
         }
         __syncthreads();
      }
      __syncthreads();
      #ifdef USE_BATCH_WARPACCESSORS
      // Insert into map only from threads 0...WARPSIZE
      if (b_tid < GPUTHREADS) {
         if (has_content[0]) {
            vbwcl_map->warpInsert(blockGID,blockLID,b_tid);
         } else {
            vbwncl_map->warpInsert(blockGID,blockLID,b_tid);
         }
      }
      #else
      // Insert into map only from thread 0
      if (b_tid == 0) {
         if (has_content[0]) {
            vbwcl_map->set_element(blockGID,blockLID);
         } else {
            vbwncl_map->set_element(blockGID,blockLID);
         }
      }
      #endif
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
   //launch parameters: dim3 grid(blocksNeeded,nMaps,1);
   const size_t hashmapIndex = blockIdx.y;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t stride = gridDim.x * blockDim.x;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = maps[hashmapIndex];
   const size_t len = thisMap->bucket_count();
   Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>* dst = thisMap->expose_bucketdata<false>();

   for (size_t bucketIndex = tid; bucketIndex < len; bucketIndex += stride) {
      dst[bucketIndex].first = emptybucket;
   }

   //Thread 0 resets fill
   if (tid==0) {
      Hashinator::Info *info = thisMap->expose_mapinfo<false>();
      info->fill=0;
   }
   return;
}

/*
 * Extracts keys (GIDs, if firstonly is true) or key-value pairs (GID-LID pairs)
 * from all provided hashmaps to provided splitvectors, and stores the vector size in an array.
 */
template <typename Rule, typename ELEMENT, bool FIRSTONLY=false>
__global__ void extract_GIDs_kernel(
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> **input_maps, // buffer of pointers to source maps
   split::SplitVector<ELEMENT> **output_vecs,
   vmesh::LocalID* output_sizes,
   Rule rule,
   vmesh::VelocityMesh **rule_meshes, // buffer of pointers to vmeshes, sizes used by rules
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> **rule_maps,
   split::SplitVector<vmesh::GlobalID> **rule_vectors
   ) {
   //launch parameters: dim3 grid(nMaps,1,1); // As this is a looping reduction
   const size_t hashmapIndex = blockIdx.x;
   if (input_maps[hashmapIndex]==0) {
      return; // Early return for invalid cells
   }
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = input_maps[hashmapIndex];
   split::SplitVector<ELEMENT> *outputVec = output_vecs[hashmapIndex];

   // Threshold value used by some rules
   vmesh::LocalID threshold = rule_meshes[hashmapIndex]->size()
      + rule_vectors[hashmapIndex]->size() - rule_maps[hashmapIndex]->size();

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
   ELEMENT* output = outputVec->data();
   // Start loop
   while (remaining > 0) {
      int current = remaining > blockDim.x ? blockDim.x : remaining;
      __syncthreads();
      const int active = (tid < current) ? rule(thisMap, input[tid], threshold) : false;
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
            assert((outputSize <= capacity) && "extract_GIDs_kernel ran out of capacity!");
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
         if constexpr (FIRSTONLY) {
            output[warpTidWriteIndex] = input[tid].first;
         } else {
            output[warpTidWriteIndex] = input[tid];
         }
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
      output_sizes[hashmapIndex] = outputSize;
   }
}

template <typename Rule, typename ELEMENT, bool FIRSTONLY=false>
void extract_GIDs_kernel_launcher(
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>** input_maps,
   split::SplitVector<ELEMENT> **output_vecs,
   vmesh::LocalID* output_sizes,
   Rule rule,
   vmesh::VelocityMesh** rule_meshes,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> **rule_maps,
   split::SplitVector<vmesh::GlobalID> **rule_vectors,
   const uint nCells,
   gpuStream_t stream
   ) {
   extract_GIDs_kernel<Rule,ELEMENT,FIRSTONLY><<<nCells, Hashinator::defaults::MAX_BLOCKSIZE, 0, stream>>>(
      input_maps,
      output_vecs,
      output_sizes,
      rule,
      rule_meshes,
      rule_maps,
      rule_vectors
      );
   CHK_ERR( gpuPeekAtLastError() );
}

/*
 * Extracts key-value (GID-LID) pairs matching the given rule
 * from the hashmaps of all provided velocity meshes,
 * stores them in provided splitvectors, and
 * clears all tombstones and matched elements.
 */
template <typename Rule>
__global__ void extract_overflown_kernel(
   vmesh::VelocityMesh **vmeshes, // buffer of pointers to vmeshes, contain hashmaps
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> **output_vecs,
   vmesh::LocalID* output_sizes,
   Rule rule
   ) {
   //launch parameters: dim3 grid(nMaps,1,1); // As this is a looping reduction
   const size_t vmeshIndex = blockIdx.x;
   if (vmeshes[vmeshIndex]==0) {
      return; // Early return for invalid cells
   }
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = vmeshes[vmeshIndex]->gpu_expose_map();
   Hashinator::Info *info = thisMap->expose_mapinfo<false>();
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *outputVec = output_vecs[vmeshIndex];

   if (info->tombstoneCounter == 0) {
      // If there are no tombstones, then also any overflown elements will be minimally overflown.
      outputVec->device_resize(0);
      return;
   }
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
   Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>* output = outputVec->data();
   const vmesh::GlobalID emptybucket = thisMap->get_emptybucket();
   // Start loop
   while (remaining > 0) {
      int current = remaining > blockDim.x ? blockDim.x : remaining;
      __syncthreads();
      const int active = (tid < current) ? rule(thisMap, input[tid]) : false;
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
            assert((outputSize <= capacity) && "extract_overflown_kernel ran out of capacity!");
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
         output[warpTidWriteIndex] = input[tid];
         // Now also delete this entry. Must edit fill count at end of kernel.
         input[tid].first = emptybucket;
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
      output_sizes[vmeshIndex] = outputSize;
      // Update mapInfo
      info->currentMaxBucketOverflow = Hashinator::defaults::BUCKET_OVERFLOW;
      info->fill -= outputSize; // subtract deleted (overflown) elements
      info->tombstoneCounter = 0;
   }
}

template <typename Rule>
void clean_tombstones_launcher(
   vmesh::VelocityMesh** vmeshes,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> **overflown_elements,
   vmesh::LocalID* output_sizes,
   Rule rule,
   const uint nCells,
   gpuStream_t stream
   ) {
   // Extract overflown elements into temporary vector
   extract_overflown_kernel<Rule><<<nCells, Hashinator::defaults::MAX_BLOCKSIZE, 0, stream>>>(
      vmeshes,
      overflown_elements,
      output_sizes,
      rule
      );
   CHK_ERR( gpuPeekAtLastError() );
}

/*
 * Mini-kernel for inserting previously extracted overflown elements
 */
__global__ void batch_insert_kernel(
   vmesh::VelocityMesh **vmeshes, // buffer of pointers to vmeshes, contain hashmaps
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> **input_vecs
   ) {
   //launch parameters: dim3 grid(largestOverflow,nCells,1);
   const uint ti = threadIdx.x; // [0,blockSize)
   const int b_tid = ti % GPUTHREADS; // [0,GPUTHREADS)
   // GPUTODO: several entries in parallel per block
   const size_t vmeshIndex = blockIdx.y;
   const size_t blockIndex = blockIdx.x;
   if (vmeshes[vmeshIndex]==0) {
      return; // Early return for invalid cells
   }
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* thisMap = vmeshes[vmeshIndex]->gpu_expose_map();
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *inputVec = input_vecs[vmeshIndex];

   size_t inputVecSize = inputVec->size();
   if (inputVecSize == 0 || blockIndex >= inputVecSize) {
      // No elements to insert
      return;
   }

   #ifdef USE_BATCH_WARPACCESSORS
   // Insert into map only from threads 0...WARPSIZE
   if (b_tid < GPUTHREADS) {
      thisMap->warpInsert((inputVec->at(blockIndex)).first,(inputVec->at(blockIndex)).second,b_tid);
   }
   #else
   // Insert into map only from thread 0
   if (b_tid == 0) {
      thisMap->set_element((inputVec->at(blockIndex)).first,(inputVec->at(blockIndex)).second);
   }
   #endif
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
   // launch grid dim3 grid(launchBlocks,nCells,1);
   // Each block manages a single GID, all velocity neighbours
   const uint nCells = gridDim.y;
   const int cellIndex = blockIdx.y;
   //const int gpuBlocks = gridDim.x; // At least VB with content list size
   const int blockistart = blockIdx.x;
   //const int blockSize = blockDim.x; // should be 26*32 or 13*64
   const uint ti = threadIdx.x;
   const uint stride = gridDim.x;

   // Cells such as DO_NOT_COMPUTE are identified with a zero in the vmeshes pointer buffer
   if (vmeshes[cellIndex] == 0) {
      return;
   }
   vmesh::VelocityMesh* vmesh = vmeshes[cellIndex];
   split::SplitVector<vmesh::GlobalID> *velocity_block_with_content_list = velocity_block_with_content_lists[cellIndex];
   vmesh::GlobalID* velocity_block_with_content_list_data = velocity_block_with_content_list->data();
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwcl_map = allMaps[cellIndex];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwncl_map = allMaps[nCells+cellIndex];
   const vmesh::LocalID nBlocks = velocity_block_with_content_list->size();

   for (int blocki=blockistart; blocki<nBlocks; blocki += stride) {
      // Return if we are beyond the size of the list for this cell

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
         if (nGID != vmesh->invalidGlobalID()) {
            #ifdef USE_BATCH_WARPACCESSORS
            // Does block already exist in mesh?
            const vmesh::LocalID LID = vmesh->warpGetLocalID(nGID, w_tid);
            // Try adding this nGID to velocity_block_with_content_map. If it exists, do not overwrite.
            const bool newlyadded = vbwcl_map->warpInsert_V<true>(nGID,LID, w_tid);
            if (newlyadded) {
               // Block did not previously exist in velocity_block_with_content_map
               if ( LID != vmesh->invalidLocalID()) {
                  // Block exists in mesh, ensure it won't get deleted:
                  vbwncl_map->warpErase(nGID, w_tid);
               }
               // else:
               // Block does not yet exist in mesh at all. Needs adding!
               // Identified as invalidLID entries in velocity_block_with_content_map.
            }
            #else
            if (w_tid==0) {
               // Does block already exist in mesh?
               const vmesh::LocalID LID = vmesh->getLocalID(nGID);
               // Add this nGID to velocity_block_with_content_map.
               const bool newlyadded = vbwcl_map->set_element<true>(nGID,LID);
               if (newlyadded) {
                  // Block did not previously exist in velocity_block_with_content_map
                  if ( LID != vmesh->invalidLocalID()) {
                     // Block exists in mesh, ensure it won't get deleted:
                     vbwncl_map->device_erase(nGID);
                  }
               }
            }
            #endif
         }
         __syncthreads();
      }
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
   // launch grid dim3 grid_neigh_halo(NeighLaunchBlocks,nCells,maxNeighbours);

   const uint nCells = gridDim.y;
   const uint maxNeighbours = gridDim.z;
   const int cellIndex = blockIdx.y;
   const int neighIndex = blockIdx.y * maxNeighbours + blockIdx.z;
   const uint stride = gridDim.x * WARPSPERBLOCK;

   // const int blockSize = blockDim.x; // should be 32*32 or 16*64
   const int ti = threadIdx.x; // [0,blockSize)
   const int w_tid = ti % GPUTHREADS; // [0,WARPSIZE)
   const int blockistart = blockIdx.x * WARPSPERBLOCK + ti / GPUTHREADS;

   // Cells such as DO_NOT_COMPUTE are identified with a zero in the vmeshes pointer buffer
   if (vmeshes[cellIndex] == 0) {
      return;
   }
   // Early return for non-existing neighbour indexes
   if (neigh_velocity_block_with_content_lists[neighIndex] == 0) {
      return;
   }

   split::SplitVector<vmesh::GlobalID> *velocity_block_with_content_list = neigh_velocity_block_with_content_lists[neighIndex];
   const uint nBlocks = velocity_block_with_content_list->size();

   for (int blocki=blockistart; blocki<nBlocks; blocki += stride) {
      // Return if we are beyond the size of the list for this cell

      vmesh::VelocityMesh* vmesh = vmeshes[cellIndex];
      vmesh::GlobalID* velocity_block_with_content_list_data = velocity_block_with_content_list->data();
      Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwcl_map = allMaps[cellIndex];
      Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* vbwncl_map = allMaps[nCells+cellIndex];

      const vmesh::GlobalID nGID = velocity_block_with_content_list_data[blocki];
      #ifdef USE_BATCH_WARPACCESSORS
      // Does block already exist in mesh?
      const vmesh::LocalID LID = vmesh->warpGetLocalID(nGID, w_tid);
      // Try adding this nGID to velocity_block_with_content_map. If it exists, do not overwrite.
      const bool newlyadded = vbwcl_map->warpInsert_V<true>(nGID,LID, w_tid);
      if (newlyadded) {
         // Block did not previously exist in velocity_block_with_content_map
         if ( LID != vmesh->invalidLocalID()) {
            // Block exists in mesh, ensure it won't get deleted:
            vbwncl_map->warpErase(nGID, w_tid);
         }
         // else:
         // Block does not yet exist in mesh at all. Needs adding!
         // Identified as invalidLID entries in velocity_block_with_content_map.
      }
      #else
      if (w_tid==0) {
         // Does block already exist in mesh?
         const vmesh::LocalID LID = vmesh->getLocalID(nGID);
         // Add this nGID to velocity_block_with_content_map.
         const bool newlyadded = vbwcl_map->set_element<true>(nGID,LID);
         if (newlyadded) {
            // Block did not previously exist in velocity_block_with_content_map
            if ( LID != vmesh->invalidLocalID()) {
               // Block exists in mesh, ensure it won't get deleted:
               vbwncl_map->device_erase(nGID);
            }
         }
      }
      #endif
   }
}

/** Mini-kernel for checking list sizes and attempting to adjust vmesh and VBC size on-device */
__global__ void batch_resize_vbc_kernel_pre(
   vmesh::VelocityMesh **vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   split::SplitVector<vmesh::GlobalID>** dev_list_with_replace_new,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_delete,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_to_replace,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_with_replace_old,
   vmesh::LocalID* contentSizes_all, // return values: nbefore, nafter, nblockstochange, resize success
   Realf* gpu_rhoLossAdjust // mass loss, set to zero
   ) {
   const size_t cellIndex = blockIdx.x;
   if (vmeshes[cellIndex]==0) {
      return; // Early return for invalid cells
   }
   vmesh::VelocityMesh *vmesh = vmeshes[cellIndex];
   vmesh::VelocityBlockContainer *blockContainer = blockContainers[cellIndex];
   split::SplitVector<vmesh::GlobalID>* list_with_replace_new = dev_list_with_replace_new[cellIndex];
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_delete = dev_list_delete[cellIndex];
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_to_replace = dev_list_to_replace[cellIndex];
   //split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_with_replace_old = dev_list_with_replace_old[cellIndex];
   vmesh::LocalID* contentSizes = contentSizes_all + cellIndex * 4; // pointer into large return value array for this cell

   const vmesh::LocalID nBlocksBeforeAdjust = vmesh->size();

   // const vmesh::LocalID n_to_replace = list_to_replace->size(); // replace these blocks
   // const vmesh::LocalID n_with_replace_new = list_with_replace_new->size(); // use to replace, or add at end
   // const vmesh::LocalID n_with_replace_old = list_with_replace_old->size(); // use to replace
   // const vmesh::LocalID n_to_delete = list_delete->size(); // delete from end
   // const vmesh::LocalID nBlocksToChange = n_with_replace_new + n_with_replace_old + n_to_delete;
   // const vmesh::LocalID nBlocksAfterAdjust = nBlocksBeforeAdjust + n_with_replace_new - n_to_delete;
   // const vmesh::LocalID nBlocksToChange = n_with_replace_new + n_with_replace_old + n_to_delete;

   const vmesh::LocalID nToAdd = list_with_replace_new->size();
   const vmesh::LocalID nToRemove = list_delete->size() + list_to_replace->size();
   const vmesh::LocalID nBlocksAfterAdjust = nBlocksBeforeAdjust + nToAdd - nToRemove;
   const vmesh::LocalID nBlocksToChange = nToAdd > nToRemove ? nToAdd : nToRemove;

   gpu_rhoLossAdjust[cellIndex] = 0.0;
   contentSizes[0] = nBlocksBeforeAdjust;
   contentSizes[1] = nBlocksAfterAdjust;
   contentSizes[2] = nBlocksToChange;
   // Should we grow the size?
   if (nBlocksAfterAdjust > nBlocksBeforeAdjust) {
      if ((nBlocksAfterAdjust <= vmesh->capacity()) && (nBlocksAfterAdjust <= blockContainer->capacity())) {
         contentSizes[3] = 1; // Resize on-device will work.
         vmesh->device_setNewSize(nBlocksAfterAdjust);
         blockContainer->setNewSize(nBlocksAfterAdjust);
      } else {
         contentSizes[3] = 0; // Need to recapacitate and resize from host
      }
   } else {
      // No error as no resize.
      contentSizes[3] = 2;
   }
}

/** Mini-kernel for adjusting vmesh and VBC size on-device aftewards (shrink only) */
__global__ void batch_resize_vbc_kernel_post(
   vmesh::VelocityMesh **vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   vmesh::LocalID* sizes // nbefore, nafter, nblockstochange, previous resize success
   ) {
   const size_t cellIndex = blockIdx.x;
   if (vmeshes[cellIndex]==0) {
      return; // Early return for invalid cells
   }
   vmesh::VelocityMesh *vmesh = vmeshes[cellIndex];
   vmesh::VelocityBlockContainer *blockContainer = blockContainers[cellIndex];
   vmesh::LocalID nBlocksAfterAdjust = sizes[4*cellIndex + 1];
   vmesh->device_setNewSize(nBlocksAfterAdjust);
   blockContainer->setNewSize(nBlocksAfterAdjust);
}


/** GPU kernel for updating blocks based on generated lists */
__global__ void batch_update_velocity_blocks_kernel(
   vmesh::VelocityMesh **vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   split::SplitVector<vmesh::GlobalID>** dev_list_with_replace_new,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_delete,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_to_replace,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>** dev_list_with_replace_old,
   vmesh::LocalID* sizes,  // nbefore, nafter, nblockstochange, previous resize success
   Realf* gpu_rhoLossAdjust // mass loss, gather from deleted blocks
   ) {
   // launch griddim3 grid(launchBlocks,nCells,1);
   const size_t cellIndex = blockIdx.y;
   if (vmeshes[cellIndex]==0) {
      return; // Early return for invalid cells
   }
   vmesh::VelocityMesh *vmesh = vmeshes[cellIndex];
   vmesh::VelocityBlockContainer *blockContainer = blockContainers[cellIndex];
   split::SplitVector<vmesh::GlobalID>* list_with_replace_new = dev_list_with_replace_new[cellIndex];
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_delete = dev_list_delete[cellIndex];
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_to_replace = dev_list_to_replace[cellIndex];
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_with_replace_old = dev_list_with_replace_old[cellIndex];

   const vmesh::LocalID nBlocksBeforeAdjust = sizes[cellIndex * 4 + 0];
   const vmesh::LocalID nBlocksToChange = sizes[cellIndex * 4 + 2];
   const vmesh::LocalID nBlocksAfterAdjust = sizes[cellIndex * 4 + 1];

   if (blockIdx.x >= nBlocksToChange) {
      return; // Early return if outside list of blocks to change
   }
   // COMMENTED OUT LINES ARE OUT-OF-DATE, REVIEW BEFORE USE
   //const int gpuBlocks = gridDim.x;
   //const int blocki = blockIdx.x;
   //const int blockSize = blockDim.x; // WID3
   const uint ti = threadIdx.x; // [0,blockSize)

   // Each GPU block / workunit could manage several Vlasiator velocity blocks at once.
   // However, thread syncs inside the kernel prevent this.
   //const uint vlasiBlocksPerWorkUnit = WARPSPERBLOCK * GPUTHREADS / WID3;
   //const uint workUnitIndex = ti / WID3; // [0,vlasiBlocksPerWorkUnit)
   //const uint index = blocki * vlasiBlocksPerWorkUnit + workUnitIndex; // [0,nBlocksToChange)
   //const uint vlasiBlocksPerWorkUnit = 1;
   //const uint workUnitIndex = 1;

   // This index into vectors can be adjusted along the way
   uint index = (uint)blockIdx.x;

   const int b_tid = ti % WID3; // [0,WID3)

   const vmesh::LocalID n_with_replace_new = list_with_replace_new->size();
   const vmesh::LocalID n_delete = list_delete->size();
   const vmesh::LocalID n_to_replace = list_to_replace->size();
   const vmesh::LocalID n_with_replace_old = list_with_replace_old->size();
   // For tracking mass-loss
   //__shared__ Realf massloss[blockSize];
   __shared__ Realf massloss[WID3];

   // Each block / workunit Processes one block from the lists.

   /*********
       Check if should delete item from end of vmesh.
       For this, we get both GID and LID from the vector.
   **/
   if (index < n_delete) {
      #ifdef DEBUG_SPATIAL_CELL
      const vmesh::GlobalID rmGID = (list_delete->at(index)).first;
      const vmesh::GlobalID rmLID = (list_delete->at(index)).second;
      #else
      const vmesh::GlobalID rmGID = ((*list_delete)[index]).first;
      const vmesh::GlobalID rmLID = ((*list_delete)[index]).second;
      #endif

      #ifdef DEBUG_SPATIAL_CELL
      if (rmGID == vmesh->invalidGlobalID()) {
         if (rmLID != vmesh->invalidLocalID()) {
            // Valid LID but invalid GID: only remove from vmesh localToGlobal?
            if (b_tid==0) {
               printf("Removing blocks: Valid LID %u but invalid GID!\n",rmLID);
            }
         } else {
            if (b_tid==0) {
               printf("Removing blocks: Invalid LID and GID!\n");
            }
         }
         return;
      }
      if (rmLID == vmesh->invalidLocalID()) {
         if (rmGID != vmesh->invalidGlobalID()) {
            // Valid GID but invalid LID: only remove from vmesh globalToLocal?
            if (b_tid==0) {
               printf("Removing blocks: Valid GID %ul but invalid LID!\n",rmGID);
            }
         }
         return;
      }
      if ((unsigned long)rmLID >= (unsigned long)nBlocksBeforeAdjust) {
         if (b_tid==0) {
            printf("Trying to outright remove block which has LID %ul >= nBlocksBeforeAdjust %ul!\n",rmLID,nBlocksBeforeAdjust);
         }
         return;
      }
      if ((unsigned long)rmLID < (unsigned long)nBlocksAfterAdjust) {
         if (b_tid==0) {
            printf("Trying to outright remove block which has LID %u smaller than nBlocksAfterAdjust %u!\n",rmLID,nBlocksAfterAdjust);
         }
         return;
      }
      #endif

      // Track mass loss:
      Realf* rm_avgs = blockContainer->getData(rmLID);
      Real* rm_block_parameters = blockContainer->getParameters(rmLID);
      const Real rm_DV3 = rm_block_parameters[BlockParams::DVX]
         * rm_block_parameters[BlockParams::DVY]
         * rm_block_parameters[BlockParams::DVZ];
      // thread-sum for rho
      massloss[ti] = rm_avgs[b_tid]*rm_DV3;
      __syncthreads();
      // Implemented just a simple non-optimized thread sum
      for (int s=WID3/2; s>0; s>>=1) {
         if (b_tid < s) {
            massloss[ti] += massloss[ti + s];
         }
         __syncthreads();
      }
      // Bookkeeping only by one thread
      if (b_tid==0) {
         Realf old = atomicAdd(&gpu_rhoLossAdjust[cellIndex], massloss[b_tid]);
      }
      __syncthreads();

      // Delete from vmesh
      #ifdef USE_BATCH_WARPACCESSORS
      vmesh->warpDeleteBlock(rmGID,rmLID,b_tid);
      #else
      if (b_tid==0) {
         vmesh->deleteBlock(rmGID,rmLID);
      }
      #endif
      // GPUTODO debug checks
      return;
   }
   index -= n_delete;

   /*********
       Check if should replace existing block with either
       existing block from end of vmesh or new block
   **/
   if (index < n_to_replace) {
      #ifdef DEBUG_SPATIAL_CELL
      const vmesh::GlobalID rmGID = (list_to_replace->at(index)).first;
      const vmesh::GlobalID rmLID = (list_to_replace->at(index)).second;
      #else
      const vmesh::GlobalID rmGID = ((*list_to_replace)[index]).first;
      const vmesh::GlobalID rmLID = ((*list_to_replace)[index]).second;
      #endif
      //const vmesh::LocalID rmLID = vmesh->warpGetLocalID(rmGID,b_tid);

      #ifdef DEBUG_SPATIAL_CELL
      if (rmGID == vmesh->invalidGlobalID()) {
         if (rmLID != vmesh->invalidLocalID()) {
            // Valid LID but invalid GID: only remove from vmesh localToGlobal?
            if (b_tid==0) {
               printf("Replacing blocks: Valid LID %u but invalid GID!\n",rmLID);
            }
         } else {
            if (b_tid==0) {
               printf("Replacing blocks: Invalid LID and GID!\n");
            }
         }
         return;
      }
      if (rmLID == vmesh->invalidLocalID()) {
         if (rmGID != vmesh->invalidGlobalID()) {
            // Valid GID but invalid LID: only remove from vmesh globalToLocal?
            if (b_tid==0) {
               printf("Replacing blocks: Valid GID %ul but invalid LID!\n",rmGID);
            }
         }
         return;
      }
      if (rmLID >= nBlocksBeforeAdjust) {
         if (b_tid==0) {
            printf("Trying to replace block which has LID %ul >= nBlocksBeforeAdjust %ul!\n",rmLID,nBlocksBeforeAdjust);
         }
         return;
      }
      #endif

      // Track mass loss:
      Realf* rm_avgs = blockContainer->getData(rmLID);
      Real* rm_block_parameters = blockContainer->getParameters(rmLID);
      const Real rm_DV3 = rm_block_parameters[BlockParams::DVX]
         * rm_block_parameters[BlockParams::DVY]
         * rm_block_parameters[BlockParams::DVZ];
      // thread-sum for rho
      massloss[ti] = rm_avgs[b_tid]*rm_DV3;
      __syncthreads();
      // Implemented just a simple non-optimized thread sum
      for (int s=WID3/2; s>0; s>>=1) {
         if (b_tid < s) {
            massloss[ti] += massloss[ti + s];
         }
         __syncthreads();
      }
      // Bookkeeping only by one thread
      if (b_tid==0) {
         Realf old = atomicAdd(&gpu_rhoLossAdjust[cellIndex], massloss[b_tid]);
      }
      __syncthreads();

      // Figure out what to use as replacement
      vmesh::GlobalID replaceGID;
      vmesh::LocalID replaceLID;

      // First option: replace with existing block from end of vmesh
      if (index < n_with_replace_old) {
         #ifdef DEBUG_SPATIAL_CELL
         replaceGID = (list_with_replace_old->at(index)).first;
         replaceLID = (list_with_replace_old->at(index)).second;
         #else
         replaceGID = ((*list_with_replace_old)[index]).first;
         replaceLID = ((*list_with_replace_old)[index]).second;
         #endif

         Realf* repl_avgs = blockContainer->getData(replaceLID);
         Real*  repl_block_parameters = blockContainer->getParameters(replaceLID);
         rm_avgs[b_tid] = repl_avgs[b_tid];
         if (b_tid < BlockParams::N_VELOCITY_BLOCK_PARAMS) {
            rm_block_parameters[b_tid] = repl_block_parameters[b_tid];
         }
         __syncthreads();

      } else {
         // Second option: add new block instead
         #ifdef DEBUG_SPATIAL_CELL
         replaceGID = list_with_replace_new->at(index - n_with_replace_old);
         #else
         replaceGID = (*list_with_replace_new)[index - n_with_replace_old];
         #endif
         replaceLID = vmesh->invalidLocalID();

         rm_avgs[b_tid] = 0;
         if (b_tid==0) {
            // Write in block parameters
            vmesh->getBlockInfo(replaceGID, rm_block_parameters+BlockParams::VXCRD);
         }
         __syncthreads();
      }
      // Remove hashmap entry for removed block, add instead created block
      #ifdef USE_BATCH_WARPACCESSORS
      vmesh->warpReplaceBlock(rmGID,rmLID,replaceGID,b_tid);
      #else
      if (b_tid==0) {
         vmesh->replaceBlock(rmGID,rmLID,replaceGID);
      }
      #endif
      #ifdef DEBUG_SPATIAL_CELL
      __syncthreads();
      if (vmesh->getGlobalID(rmLID) != replaceGID) {
         if (b_tid==0) {
            printf("Error! Replacing did not result in wanted GID at old LID in update_velocity_blocks_kernel! \n");
         }
         __syncthreads();
      }
      if (vmesh->getLocalID(replaceGID) != rmLID) {
         if (b_tid==0) {
            printf("Error! Replacing did not result in old LID at replaced GID in update_velocity_blocks_kernel! \n");
         }
         __syncthreads();
      }
      #endif

      return;
   }
   index -= n_to_replace;

   /*********
       Finally check if we should add new block after end of current vmesh
       We have reserved/used some entries from the beginning of the list_with_replace_new
       for the previous section, so now we access that with a different index.
   **/
   const uint add_index = index + (n_to_replace - n_with_replace_old);
   if (add_index < n_with_replace_new) {
      #ifdef DEBUG_SPATIAL_CELL
      const vmesh::GlobalID addGID = list_with_replace_new->at(add_index);
      if (b_tid==0) {
         if (vmesh->getLocalID(addGID) != vmesh->invalidLocalID()) {
            printf("Trying to add new GID %u to mesh which already contains it! index=%u addindex=%u\n",addGID,index,add_index);
         }
      }
      #else
      const vmesh::GlobalID addGID = (*list_with_replace_new)[add_index];
      #endif

      // We need to add the data of addGID to a new LID. Here we still use the regular index.
      const vmesh::LocalID addLID = nBlocksBeforeAdjust + index;
      Realf* add_avgs = blockContainer->getData(addLID);
      #ifdef DEBUG_SPATIAL_CELL
      __syncthreads();
      if (addGID == vmesh->invalidGlobalID()) {
         printf("Error! invalid addGID!\n");
         return;
      }
      if (addLID == vmesh->invalidLocalID()) {
         printf("Error! invalid addLID!\n");
         return;
      }
      #endif
      Real* add_block_parameters = blockContainer->getParameters(addLID);
      // Zero out blockdata
      add_avgs[b_tid] = 0;
      if (b_tid==0) {
         // Write in block parameters
         vmesh->getBlockInfo(addGID, add_block_parameters+BlockParams::VXCRD);
      }
      __syncthreads();

      // Insert new hashmap entry into vmesh
      #ifdef USE_BATCH_WARPACCESSORS
      vmesh->warpPlaceBlock(addGID,addLID,b_tid);
      #else
      if (b_tid==0) {
         vmesh->placeBlock(addGID,addLID);
      }
      #endif
      #ifdef DEBUG_SPATIAL_CELL
      __syncthreads();
      if (vmesh->getGlobalID(addLID) == vmesh->invalidGlobalID()) {
         printf("Error! invalid GID after add from addLID!\n");
      }
      if (vmesh->getLocalID(addGID) == vmesh->invalidLocalID()) {
         printf("Error! invalid LID after add from addGID!\n");
      }
      #endif
      return;
   }

   // Fall-through error!
   if (b_tid==0) {
      printf("Error! Fall through in batch_update_velocity_blocks_kernel! index %u nBlocksBeforeAdjust %u nBlocksAfterAdjust %u \n",
             index,nBlocksBeforeAdjust,nBlocksAfterAdjust);
   }
   __syncthreads();
}

#endif
