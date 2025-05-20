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

#include "gpu_acc_map.hpp"
#include "../spatial_cells/block_adjust_gpu.hpp"

__device__ void inline swapBlockIndices(vmesh::LocalID &blockIndices0,vmesh::LocalID &blockIndices1,vmesh::LocalID &blockIndices2, const uint dimension){
   vmesh::LocalID temp;
   // Switch block indices according to dimensions, the algorithm has
   // been written for integrating along z.
   switch (dimension){
   case 0:
      /*i and k coordinates have been swapped*/
      temp=blockIndices2;
      blockIndices2=blockIndices0;
      blockIndices0=temp;
      break;
   case 1:
      /*in values j and k coordinates have been swapped*/
      temp=blockIndices2;
      blockIndices2=blockIndices1;
      blockIndices1=temp;
      break;
   case 2:
      break;
   }
}


/* Fills the target probe block with the invalid value for vmesh::LocalID
   Also clears provided vectors
*/
__global__ void fill_probe_invalid(
   vmesh::LocalID *probeCube,
   const size_t nTot,
   const vmesh::LocalID invalid,
   // Pass these for emptying
   split::SplitVector<vmesh::GlobalID>* *lists_with_replace_new,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* *lists_delete,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* *lists_to_replace,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* *lists_with_replace_old,
   // This one is resized and re-used as a LIDlist
   split::SplitVector<vmesh::GlobalID> ** dev_vbwcl_vec,
   const vmesh::LocalID nBlocks,
   const uint cellOffset
   ) {
   const size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = ind; i < nTot; i += gridDim.x * blockDim.x) {
      probeCube[i] = invalid;
   }
   if (ind==0) {
      lists_with_replace_new[cellOffset]->clear();
      lists_delete[cellOffset]->clear();
      lists_to_replace[cellOffset]->clear();
      lists_with_replace_old[cellOffset]->clear();
      dev_vbwcl_vec[cellOffset]->device_resize(nBlocks,false); // do not construct / reset new entries
   }
}

/* Takes contents of vmesh and places in probe cube.

   One option for a probe cube would be to reduce (with __ballot_sync) along the
   direction of propagation to get the number of blocks in a column. However, this
   does not have an obvious way to support gathering columnsets (several columns at
   one set of parallel indices).

   Thus, instead, for following analysis of the probe cube, we want the warp/wavefront
   to read a dimension *not* propagating along, because then the wavefront can loop
   over the dimension to propagate along.

   Since we read in LID order, writes will be jumbled anyhow, but we can make future
   accesses to the probe cube efficient by writing in a smart order.

   To even better parallelize, the two non-propagated dimensions are merged
   (so it isn't an actual cube).

   TODO: ensure the first and second dimensions are powers of two for
   optimized reads? Then stepping will be based on array edge sizes instead of D0/1/2.
 */
__global__ void fill_probe_ordered(
   vmesh::VelocityMesh** __restrict__ vmeshes,
   vmesh::LocalID *probeCube,
   const uint* __restrict__ gpu_block_indices_to_probe,
   const vmesh::LocalID nBlocks,
   const uint cellOffset
   ) {
   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const vmesh::LocalID LID = blockDim.x * blockIdx.x + ti;
   if (LID >= nBlocks) {
      return;
   }
   const vmesh::VelocityMesh* __restrict__ vmesh = vmeshes[cellOffset];
   // Store in probe cube with ordering so that reading will be fast
   const vmesh::GlobalID GID = vmesh->getGlobalID(LID);
   vmesh::LocalID indices[3];
   vmesh->getIndices(GID,indices[0],indices[1],indices[2]);

   // Use pre-calculated probe indices
   const int target = indices[0] * gpu_block_indices_to_probe[0]
      + indices[1] * gpu_block_indices_to_probe[1]
      + indices[2] * gpu_block_indices_to_probe[2];

   probeCube[target] = LID;
}

/* Flattens probe cube into two reduction results (counters).
   In the probe cube, there's the dimension of acceleration (size Dacc)
   and the other two dimensions, merged (size Dother).
   For each index in the other two dimensions, we have a position in
   v-space associated with the potential for constructing columns.
   Thus, that position is now termed a potential column position or potColumn.

   This kernel loops over Dacc to find:
   (1) How many acceleration columns were found for each potColumn
   (2) How many blocks were found for each potColumn
*/
__global__ void flatten_probe_cube(
   const vmesh::LocalID* __restrict__ probeCube,
   vmesh::LocalID* probeFlattened,
   const vmesh::LocalID Dacc,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   const vmesh::LocalID invalid
   ) {
   // Probe cube contents have been ordered based on acceleration dimesion
   // so this kernel always reads in the same way.

   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const vmesh::LocalID ind = blockDim.x * blockIdx.x + ti;

   if (ind < Dother) {
      // Per-thread counters
      vmesh::LocalID foundBlocks = 0;
      vmesh::LocalID foundCols = 0;
      bool inCol = false;

      for (vmesh::LocalID j = 0; j < Dacc; j++) {
         if (probeCube[j*Dother + ind] == invalid) {
            // No block at this index.
            if (inCol) {
               // finish current column
               foundCols++;
               inCol = false;
            }
         } else {
            // Valid block found at this index
            foundBlocks++;
            if (!inCol) {
               // start new column
               inCol = true;
            }
         }
      }
      // Finished loop. If we are "still in a colum", count that.
      if (inCol) {
         foundCols++;
      }
      // Store values in global memory array
      probeFlattened[ind] = foundCols;
      probeFlattened[flatExtent + ind] = foundBlocks;
   }
}

/* This kernel performs exclusive prefix scans of the flattened probe cube.
   Produces:

   (1) the cumulative sum of columns up to the beginning of each potColumn
   (2) the cumulative sum of column sets up to the beginning of each potColumn
   (3) the cumulative sum of blocks up to the beginning of each potColumn

   This is not a fully optimized scan as it uses only a single block
   in order to perform it all in one kernel. As the fixed-size shared memory
   buffer gets overwritten each cycle, we store the actual prefix scan results
   into the third and fourth entries in the probeFlattened buffer and keep
   track of the accumulated offset to the prefix.

   The count of columns to be evaluated is estimated to remain somewhat small-ish,
   so there should not be all that many loops of the cycle to deal with.
*/

// Defined in splitvector headers (32 and 5)
//#define NUM_BANKS 16
//#define LOG_NUM_BANKS 4

// Which one provides best bank conflict avoidance?
#define LOG_BANKS 4
// One below gives warning #63-D: shift count is too large yet works.
#define BANK_OFFSET(n)                          \
  ((n) >> (LOG_BANKS) + (n) >> (2 * LOG_BANKS))
//#define BANK_OFFSET(n) ((n) >> LOG_BANKS) // segfaults, do not use
//#define BANK_OFFSET(n) 0 // Reduces to no bank conflict elimination

__global__ void scan_probe(
   vmesh::LocalID *probeFlattened,
   const vmesh::LocalID Dacc,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   const vmesh::LocalID nBlocks, // For early exit
   vmesh::LocalID *dev_returnLID,
   ColumnOffsets* columnData
   ) {

   // Per-thread counters in shared memory for reduction. Double size buffer for better bank conflict avoidance.
   const size_t n = 2*Hashinator::defaults::MAX_BLOCKSIZE;
   __shared__ vmesh::LocalID reductionA[2*Hashinator::defaults::MAX_BLOCKSIZE]; // columns
   __shared__ vmesh::LocalID reductionB[2*Hashinator::defaults::MAX_BLOCKSIZE]; // columnsets
   __shared__ vmesh::LocalID reductionC[2*Hashinator::defaults::MAX_BLOCKSIZE]; // blocks
   __shared__ vmesh::LocalID offsetA;
   __shared__ vmesh::LocalID offsetB;
   __shared__ vmesh::LocalID offsetC;

   const int ti = threadIdx.x;
   if (ti==0) { // Cumulative result gathered per cycle
      offsetA = 0;
      offsetB = 0;
      offsetC = 0;
   }
   __syncthreads();
   size_t majorOffset = 0;
   // Utilizes bank conflict avoidance scheme. To simplify handling, the input buffer
   // is enforced to be a multiple of 2*Hashinator::defaults::MAX_BLOCKSIZE in size.
   while ((majorOffset < flatExtent) && (offsetC<nBlocks)) {
      int offset = 1;
      // Load input into shared memory
      int ai = ti;
      int bi = ti + (n/2);
      int bankOffsetA = BANK_OFFSET(ai);
      int bankOffsetB = BANK_OFFSET(bi);
      reductionA[ai + bankOffsetA] = probeFlattened[majorOffset + ai];
      reductionA[bi + bankOffsetB] = probeFlattened[majorOffset + bi];
      reductionB[ai + bankOffsetA] = (probeFlattened[majorOffset + ai] != 0 ? 1 : 0);
      reductionB[bi + bankOffsetB] = (probeFlattened[majorOffset + bi] != 0 ? 1 : 0);
      reductionC[ai + bankOffsetA] = probeFlattened[flatExtent + majorOffset + ai];
      reductionC[bi + bankOffsetB] = probeFlattened[flatExtent + majorOffset + bi];

      // build sum in place up the tree
      for (int d = n>>1; d > 0; d >>= 1) {
         __syncthreads();
         if (ti < d) {
            int ai = offset*(2*ti+1)-1;
            int bi = offset*(2*ti+2)-1;
            ai += BANK_OFFSET(ai);
            bi += BANK_OFFSET(bi);
            reductionA[bi] += reductionA[ai];
            reductionB[bi] += reductionB[ai];
            reductionC[bi] += reductionC[ai];
         }
         offset *= 2;
      }
      // Clear the last element
      if (ti==0) {
         reductionA[n - 1 + BANK_OFFSET(n - 1)] = 0;
         reductionB[n - 1 + BANK_OFFSET(n - 1)] = 0;
         reductionC[n - 1 + BANK_OFFSET(n - 1)] = 0;
      }

      // traverse down tree & build scan
      for (int d = 1; d < n; d *= 2) {
         offset >>= 1;
         __syncthreads();
         if (ti < d) {
            int ai = offset*(2*ti+1)-1;
            int bi = offset*(2*ti+2)-1;
            ai += BANK_OFFSET(ai);
            bi += BANK_OFFSET(bi);

            vmesh::LocalID t = reductionA[ai];
            reductionA[ai] = reductionA[bi];
            reductionA[bi] += t;
            t = reductionB[ai];
            reductionB[ai] = reductionB[bi];
            reductionB[bi] += t;
            t = reductionC[ai];
            reductionC[ai] = reductionC[bi];
            reductionC[bi] += t;
         }
      }
      __syncthreads();

      // write results to device memory, increment majorOffset and offsetA/B/C.
      // Remember:
      // The flattened version must store:
      // 1) how many columns per potential column position (potColumn) (input for this kernel)
      // 2) how many blocks per potColumn (input for this kernel)
      // 3) cumulative offset into columns per potColumn (output for this kernel)
      // 4) cumulative offset into columnSets per potColumn (output for this kernel)
      // 5) cumulative offset into blocks per potColumn (output for this kernel)

      probeFlattened[2*flatExtent + majorOffset + ai] = reductionA[ai + bankOffsetA] + offsetA;
      probeFlattened[2*flatExtent + majorOffset + bi] = reductionA[bi + bankOffsetB] + offsetA;
      probeFlattened[3*flatExtent + majorOffset + ai] = reductionB[ai + bankOffsetA] + offsetB;
      probeFlattened[3*flatExtent + majorOffset + bi] = reductionB[bi + bankOffsetB] + offsetB;
      probeFlattened[4*flatExtent + majorOffset + ai] = reductionC[ai + bankOffsetA] + offsetC;
      probeFlattened[4*flatExtent + majorOffset + bi] = reductionC[bi + bankOffsetB] + offsetC;
      // Advance to reading next section of input buffer
      majorOffset += n;
      // Increment cumulative offset (exclusive sum result of last bin + contents of that one)
      __syncthreads();
      if (ti==0) {
         offsetA += reductionA[n-1] + probeFlattened[majorOffset-1];
         offsetB += reductionB[n-1] + (probeFlattened[majorOffset-1] != 0 ? 1 : 0);
         offsetC += reductionC[n-1] + probeFlattened[flatExtent + majorOffset-1];
      }
      __syncthreads();
   }
   __syncthreads();
   if (ti == 0) {
      // Store reduction results
      const vmesh::LocalID numCols = offsetA;
      const vmesh::LocalID numColSets = offsetB;
      //printf("found columns %u and columnsets %u, %u blocks vs %d\n",numCols,numColSets,offsetC,nBlocks);
      dev_returnLID[0] = numCols; // Total number of columns
      dev_returnLID[1] = numColSets; // Total number of column sets
      // Resize device-side column offset container vectors. First verify capacity.
      // set dev_returnLID[2] to unity to indicate if re-capacitate on host is needed.
      if ( (columnData->dev_capacityCols() < numCols) ||
           (columnData->dev_capacityColSets() < numColSets) ) {
         dev_returnLID[2] = 1;
         return;
      } else {
         dev_returnLID[2] = 0;
      }
      columnData->device_setSizes(numCols,numColSets);
   }

   // Todo: unrolling  of reduction loops to get even more performance.
   // Memos:
   // Perform all-prefix-sum to gather offsets
   // Look at e.g.
   // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
   // Example 39-2 onwards
   // See also  splitvector's stream compaction mechanism and
   // Credits to  https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
   // Should also be made to work with arbitrary size buffers, not just powers-of-two

}

/** Utilizing the available cumulative offsets, this parallel kernel
    builds the offsets required for columns.

    It read the contents of the probe cube and outputs the stored GIDs
    and LIDs into provided buffers.
*/
__global__ void build_column_offsets(
   vmesh::VelocityMesh** __restrict__ vmeshes,
   const vmesh::LocalID* __restrict__ probeCube,
   const vmesh::LocalID* __restrict__ probeFlattened,
   const vmesh::LocalID D0,
   const vmesh::LocalID D1,
   const vmesh::LocalID D2,
   const int dimension,
   const size_t flatExtent,
   const vmesh::LocalID invalid,
   ColumnOffsets* columnData,
   split::SplitVector<vmesh::GlobalID> ** dev_vbwcl_vec, // use as LIDlist
   const uint cellOffset
   //size_t cellID can be passed for debug purposes
   ) {
   // Probe cube contents have been ordered based on acceleration dimesion
   // so this kernel always reads in the same way.

   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const vmesh::LocalID ind = blockDim.x * blockIdx.x + ti;
   // Caller function verified this cast is safe
   vmesh::LocalID* LIDlist = reinterpret_cast<vmesh::LocalID*>(dev_vbwcl_vec[cellOffset]->data());
   //const vmesh::VelocityMesh* __restrict__ vmesh = vmeshes[cellOffset]; // For debug printouts

   // if (ti+ind==0) {
   //    printf("CID%lu Total cols %lu colsets %lu\n",cellID,(size_t)columnData->columnBlockOffsets.size(),(size_t)columnData->setColumnOffsets.size());
   // }
   // __syncthreads();
   // definition: potColumn is a potential column(set), i.e. a stack from the probe cube.
   // potColumn indexes/offsets into columnData and LIDlist
   const vmesh::LocalID N_cols = probeFlattened[ind];
   //const vmesh::LocalID N_blocks_per_colset = probeFlattened[flatExtent + ind];
   const vmesh::LocalID offset_cols = probeFlattened[2*flatExtent + ind];
   const vmesh::LocalID offset_colsets = probeFlattened[3*flatExtent + ind];
   const vmesh::LocalID offset_blocks = probeFlattened[4*flatExtent + ind];

   // Here we use ind to back-calculate the transverse "x" and "y" indices (i,j) of the column(set).
   // which is by agreement propagated in the "z"-direction.
   int i,j;
   int Dacc, Dother;
   switch (dimension) {
      case 0:
         // propagate along x
         Dacc = D0;
         Dother = D1*D2;
         i = ind % D1; // Z (last dimension)
         j = ind / D1; // Y
         break;
      case 1:
         // propagate along y
         Dacc = D1;
         Dother = D0*D2;
         i = ind / D2; // X
         j = ind % D2; // Z (last dimension)
         break;
      case 2:
         // propagate along z
         Dacc = D2;
         Dother = D0*D1;
         i = ind / D1; // X
         j = ind % D1; // Y (last dimension)
         break;
      default:
         assert("ERROR! incorrect dimension!\n");
   }
   // Todo: Store i,j,minBlockK, maxBlockK
   if (ind < Dother) {
      if (N_cols != 0) {
         // Update values in columnSets vector
         //printf("CID%lu ind %d    offset_colsets %d     offset_cols %d    Ncols %d\n",cellID,ind,offset_colsets,offset_cols,N_cols);
         columnData->setColumnOffsets[offset_colsets] = offset_cols;
         columnData->setNumColumns[offset_colsets] = N_cols;
      }
      // Per-thread counters
      vmesh::LocalID foundBlocks = 0;
      vmesh::LocalID foundBlocksThisCol = 0;
      vmesh::LocalID foundCols = 0;
      bool inCol = false;

      // Loop through acceleration dimension of cube
      for (vmesh::LocalID k = 0; k < Dacc; k++) {
         // Early return when all columns have been completed
         if (foundCols >= N_cols) {
            return;
         }
         const vmesh::LocalID LID = probeCube[k*Dother + ind];
         if (LID == invalid) {
            // No block at this index.
            if (inCol) {
               // finish current column
               columnData->columnNumBlocks[offset_cols + foundCols] = foundBlocksThisCol;
               //printf("CID%lu ind %d    col %d+%d = %d    blocks %d\n",cellID,ind,offset_cols,foundCols,offset_cols + foundCols,foundBlocksThisCol);
               foundCols++;
               inCol = false;
            }
         } else {
            // Valid block found at this index!
            // Store LID into buffer
            LIDlist[offset_blocks + foundBlocks] = LID;
            // const vmesh::GlobalID GID = vmesh->getGlobalID(LID);;
            //printf("CID%lu GID %d LID %d offset_blocks %d foundblocks %d\n",cellID,GID,LID,offset_blocks,foundBlocks);
            if (!inCol) {
               // start new column
               inCol = true;
               foundBlocksThisCol = 0;
               columnData->columnBlockOffsets[offset_cols + foundCols] = offset_blocks + foundBlocks;
               columnData->i[offset_cols + foundCols] = i;
               columnData->j[offset_cols + foundCols] = j;
               columnData->kBegin[offset_cols + foundCols] = k;
               //printf("CID%lu ind %d    col %d+%d = %d    blocks-offset %d+%d = %d\n",cellID,ind,offset_cols,foundCols,offset_cols + foundCols,offset_blocks,foundBlocks,offset_blocks + foundBlocks);
            }
            foundBlocks++;
            foundBlocksThisCol++;
         }
      }
      // Finished loop. If we are "still in a colum", count that.
      if (inCol) {
         columnData->columnNumBlocks[offset_cols + foundCols] = foundBlocksThisCol;
         //printf("CID%lu ind %d    col %d+%d = %d     blocks %d\n",cellID,ind,offset_cols,foundCols,offset_cols + foundCols,foundBlocksThisCol);
         //foundCols++:
      }
   }
}

__global__ void __launch_bounds__(VECL,4) reorder_blocks_by_dimension_kernel(
   vmesh::VelocityBlockContainer** __restrict__ blockContainers,
   Vec *gpu_blockDataOrdered,
   const uint* __restrict__ gpu_cell_indices_to_id,
   split::SplitVector<vmesh::GlobalID> ** dev_vbwcl_vec, // use as LIDlist
   const ColumnOffsets* __restrict__ columnData,
   const vmesh::LocalID valuesSizeRequired,
   const uint cellOffset
) {
   // Takes the contents of blockData, sorts it into blockDataOrdered,
   // performing transposes as necessary
   // Works column-per-column and adds the necessary one empty block at each end
   const int ti = threadIdx.x;
   const uint iColumn = blockIdx.x;
   #ifdef DEBUG_ACC
   const int nThreads = blockDim.x; // should be equal to VECL
   if (nThreads != VECL) {
      if (ti==0) {
         printf("Warning! VECL not matching thread count for GPU kernel!\n");
      }
   }
   #endif
   const vmesh::VelocityBlockContainer* __restrict__ blockContainer = blockContainers[cellOffset];

   // Caller function verified this cast is safe
   vmesh::LocalID* LIDlist = reinterpret_cast<vmesh::LocalID*>(dev_vbwcl_vec[cellOffset]->data());

   // Each gpuBlock deals with one column.
   {
      const uint inputOffset = columnData->columnBlockOffsets[iColumn];
      const uint outputOffset = (inputOffset + 2 * iColumn) * (WID3/VECL);
      const uint columnLength = columnData->columnNumBlocks[iColumn];

      // Loop over column blocks
      for (uint b = 0; b < columnLength; b++) {
         // Slices
         for (uint k=0; k<WID; ++k) {
            // Each block slice can span multiple VECLs (equal to gputhreads per block)
            for (uint j = 0; j < WID; j += VECL/WID) {
               // full-block index
               const int input = k*WID2 + j*VECL + ti;
               // directional indices
               const int input_2 = input / WID2; // last (slowest) index
               const int input_1 = (input - input_2 * WID2) / WID; // medium index
               const int input_0 = input - input_2 * WID2 - input_1 * WID; // first (fastest) index
               // slice vector index
               const int jk = j / (VECL/WID);
               const int sourceindex = input_0 * gpu_cell_indices_to_id[0]
                  + input_1 * gpu_cell_indices_to_id[1]
                  + input_2 * gpu_cell_indices_to_id[2];

               #ifdef DEBUG_ACC
               assert((inputOffset + b) < blockContainer->size() && "reorder_blocks_by_dimension_kernel too large LID");
               assert((outputOffset + i_pcolumnv_gpu_b(jk, k, b, columnLength)) < valuesSizeRequired && "output error");
               #endif
               const vmesh::LocalID LID = LIDlist[inputOffset + b];
               const Realf* __restrict__ gpu_blockData = blockContainer->getData(LID);
               gpu_blockDataOrdered[outputOffset + i_pcolumnv_gpu_b(jk, k, b, columnLength)][ti]
                  = gpu_blockData[sourceindex ];

            } // end loop k (layers per block)
         } // end loop b (blocks per column)
      } // end loop j (vecs per layer)

      // Set first and last blocks to zero
      for (uint k=0; k<WID; ++k) {
         for (uint j = 0; j < WID; j += VECL/WID){
               int jk = j / (VECL/WID);
               #ifdef DEBUG_ACC
               assert((outputOffset + i_pcolumnv_gpu_b(jk, k, columnLength, columnLength)) < valuesSizeRequired && "output error");
               #endif
               gpu_blockDataOrdered[outputOffset + i_pcolumnv_gpu_b(jk, k, -1, columnLength)][ti] = 0.0;
               gpu_blockDataOrdered[outputOffset + i_pcolumnv_gpu_b(jk, k, columnLength, columnLength)][ti] = 0.0;
         }
      }
   } // end iColumn
   // Note: this kernel does not memset gpu_blockData to zero.
   // A separate memsetasync call is required for that.
}


// Using columns, evaluate which blocks are target or source blocks
__global__ void __launch_bounds__(GPUTHREADS,4) evaluate_column_extents_kernel(
   const uint dimension,
   vmesh::VelocityMesh** __restrict__ vmeshes,
   ColumnOffsets* gpu_columnData,
   split::SplitVector<vmesh::GlobalID>* *lists_with_replace_new,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>* *allMaps,
   const uint* __restrict__ gpu_block_indices_to_id,
   const Realf intersection,
   const Realf intersection_di,
   const Realf intersection_dj,
   const Realf intersection_dk,
   const int bailout_velocity_space_wall_margin,
   const int max_v_length,
   const Realf v_min,
   const Realf dv,
   uint *bailout_flag,
   const uint cellOffset
   ) {
   const uint warpSize = blockDim.x;
   const uint setIndex = blockIdx.x;
   const uint ti = threadIdx.x;

   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_require = allMaps[2*cellOffset];
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_remove = allMaps[2*cellOffset+1];
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new = lists_with_replace_new[cellOffset];
   const vmesh::VelocityMesh* __restrict__ vmesh = vmeshes[cellOffset];
   // Shared within all threads in one block (one columnSet)
   __shared__ int isTargetBlock[MAX_BLOCKS_PER_DIM];
   __shared__ int isSourceBlock[MAX_BLOCKS_PER_DIM];

   if (setIndex < gpu_columnData->setColumnOffsets.size()) {

      // Clear flags used for this columnSet
      for(uint tti = 0; tti < MAX_BLOCKS_PER_DIM; tti += warpSize ) {
         const uint index = tti + ti;
         if (index < MAX_BLOCKS_PER_DIM) {
            isTargetBlock[index] = 0;
            isSourceBlock[index] = 0;
         }
      }
      __syncthreads();

      /*need x,y coordinate of this column set */
      const vmesh::LocalID set_i = gpu_columnData->i[gpu_columnData->setColumnOffsets[setIndex]];
      const vmesh::LocalID set_j = gpu_columnData->j[gpu_columnData->setColumnOffsets[setIndex]];
      //const vmesh::LocalID set_kBegin = gpu_columnData->kBegin[gpu_columnData->setColumnOffsets[setIndex]];

      /* Compute the maximum starting point of the lagrangian (target) grid
         within the 4 corner cells in this block. Needed for computing
         maximum extent of target column.
      */

      Realf intersectionMins[4];
      intersectionMins[0] = intersection + (set_i * WID + 0) * intersection_di +
         (set_j * WID + 0) * intersection_dj;
      intersectionMins[1] = intersection + (set_i * WID + 0) * intersection_di +
         (set_j * WID + WID - 1) * intersection_dj;
      intersectionMins[2] = intersection + (set_i * WID + WID - 1) * intersection_di +
         (set_j * WID + 0) * intersection_dj;
      intersectionMins[3] = intersection + (set_i * WID + WID - 1) * intersection_di +
         (set_j * WID + WID - 1) * intersection_dj;

      Realf min_intersectionMin = std::min(std::min(intersectionMins[0],intersectionMins[1]),
                                           std::min(intersectionMins[2],intersectionMins[3]));
      Realf max_intersectionMin = std::max(std::max(intersectionMins[0],intersectionMins[1]),
                                           std::max(intersectionMins[2],intersectionMins[3]));

      // Now record which blocks are target blocks
      for (uint columnIndex = gpu_columnData->setColumnOffsets[setIndex];
           columnIndex < gpu_columnData->setColumnOffsets[setIndex] + gpu_columnData->setNumColumns[setIndex] ;
           ++columnIndex) {
         // Not parallelizing this at this level; not going to be many columns within a set
         // (and we want to manage each columnSet within one block)

         // Abort all threads if vector capacity bailout
         if (bailout_flag[1] ) {
            return;
         }

         const vmesh::LocalID n_cblocks = gpu_columnData->columnNumBlocks[columnIndex];
         const vmesh::LocalID kBegin = gpu_columnData->kBegin[columnIndex];
         const vmesh::LocalID kEnd = kBegin + n_cblocks -1;

         /* firstBlockV is in z the minimum velocity value of the lower
          *  edge in source grid.
          * lastBlockV is in z the maximum velocity value of the upper
          *  edge in source grid. */
         const Realf firstBlockMinV = (WID * kBegin) * dv + v_min;
         const Realf lastBlockMaxV = (WID * (kEnd + 1)) * dv + v_min;

         /* gk is now the k value in terms of cells in target
            grid. This distance between max_intersectionMin (so lagrangian
            plan, well max value here) and V of source grid, divided by
            intersection_dk to find out how many grid cells that is*/
         const int firstBlock_gk = (int)((firstBlockMinV - max_intersectionMin)/intersection_dk);
         const int lastBlock_gk = (int)((lastBlockMaxV - min_intersectionMin)/intersection_dk);

         int firstBlockIndexK = firstBlock_gk/WID;
         int lastBlockIndexK = lastBlock_gk/WID;

         // now enforce mesh limits for target column blocks (and check if we are
         // too close to the velocity space boundaries)
         firstBlockIndexK = (firstBlockIndexK >= 0)            ? firstBlockIndexK : 0;
         firstBlockIndexK = (firstBlockIndexK < max_v_length ) ? firstBlockIndexK : max_v_length - 1;
         lastBlockIndexK  = (lastBlockIndexK  >= 0)            ? lastBlockIndexK  : 0;
         lastBlockIndexK  = (lastBlockIndexK  < max_v_length ) ? lastBlockIndexK  : max_v_length - 1;
         if(firstBlockIndexK < bailout_velocity_space_wall_margin
            || firstBlockIndexK >= max_v_length - bailout_velocity_space_wall_margin
            || lastBlockIndexK < bailout_velocity_space_wall_margin
            || lastBlockIndexK >= max_v_length - bailout_velocity_space_wall_margin
            ) {
            // Pass bailout (hitting the wall) flag back to host
            if (ti==0) {
               bailout_flag[0] = 1;
            }
         }

         //store source blocks
         for (uint blockK = kBegin; blockK <= kEnd; blockK +=warpSize){
            if ((blockK+ti) <= kEnd) {
               isSourceBlock[blockK+ti] = 1; // Does not need to be atomic, as long as it's no longer zero
            }
         }
         __syncthreads();

         //store target blocks
         for (uint blockK = (uint)firstBlockIndexK; blockK <= (uint)lastBlockIndexK; blockK+=warpSize){
            if ((blockK+ti) <= (uint)lastBlockIndexK) {
               isTargetBlock[blockK+ti] = 1; // Does not need to be atomic, as long as it's no longer zero
            }
         }
         __syncthreads();

         if (ti==0) {
            // Store for each column firstBlockIndexK, and lastBlockIndexK
            gpu_columnData->minBlockK[columnIndex] = firstBlockIndexK;
            gpu_columnData->maxBlockK[columnIndex] = lastBlockIndexK;
         }
      } // end loop over columns in set
      __syncthreads();

      for (uint blockT = 0; blockT < MAX_BLOCKS_PER_DIM; blockT +=warpSize) {
         const uint blockK = blockT + ti;
         // Not using warp accessors, as each thread has different block
         if (blockK < MAX_BLOCKS_PER_DIM) {
            if (isTargetBlock[blockK] != 0) {
               const int targetBlock =
                  set_i  * gpu_block_indices_to_id[0] +
                  set_j  * gpu_block_indices_to_id[1] +
                  blockK * gpu_block_indices_to_id[2];
               dev_map_require->set_element(targetBlock, vmesh->getLocalID(targetBlock));
            }
            if (isTargetBlock[blockK] !=0 && isSourceBlock[blockK] == 0 )  {
               const int targetBlock =
                  set_i  * gpu_block_indices_to_id[0] +
                  set_j  * gpu_block_indices_to_id[1] +
                  blockK * gpu_block_indices_to_id[2];
               if (!list_with_replace_new->device_push_back(targetBlock)) {
                  bailout_flag[1]=1; // out of capacity
               }
            }
            if (isTargetBlock[blockK] == 0 && isSourceBlock[blockK] != 0 )  {
               const int targetBlock =
                  set_i  * gpu_block_indices_to_id[0] +
                  set_j  * gpu_block_indices_to_id[1] +
                  blockK * gpu_block_indices_to_id[2];
               dev_map_remove->set_element(targetBlock, vmesh->getLocalID(targetBlock));
            }
         } // block within MAX_BLOCKS_PER_DIM
      } // loop over all potential blocks
   } // if valid setIndex
}

__global__ void __launch_bounds__(VECL,4) acceleration_kernel(
   vmesh::VelocityMesh** __restrict__ vmeshes,
   vmesh::VelocityBlockContainer **blockContainers,
   const Vec* __restrict__ gpu_blockDataOrdered,
   const uint* __restrict__ gpu_cell_indices_to_id,
   const uint* __restrict__ gpu_block_indices_to_id,
   const ColumnOffsets* __restrict__ gpu_columnData,
   const Realf intersection,
   const Realf intersection_di,
   const Realf intersection_dj,
   const Realf intersection_dk,
   const Realf v_min,
   const Realf i_dv,
   const Realf dv,
   const Realf minValue,
   const size_t invalidLID,
   const uint cellOffset
) {
   //const uint gpuBlocks = gridDim.x * gridDim.y * gridDim.z;
   //const uint warpSize = blockDim.x * blockDim.y * blockDim.z;
   const uint blocki = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
   const uint w_tid = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;

   const vmesh::VelocityMesh* __restrict__ vmesh = vmeshes[cellOffset];
   vmesh::VelocityBlockContainer *blockContainer = blockContainers[cellOffset];
   Realf *gpu_blockData = blockContainer->getData();
   const uint column = blocki;
   {
      /* New threading with each warp/wavefront working on one vector */
      const Realf v_r0 = ( (WID * gpu_columnData->kBegin[column]) * dv + v_min);

      // i,j,k are relative to the order in which we copied data to the values array.
      // After this point in the k,j,i loops there should be no branches based on dimensions
      // Note that the i dimension is vectorized, and thus there are no loops over i
      // Iterate through the perpendicular directions of the column
      for (uint j = 0; j < WID; j += VECL/WID) {
         // If VECL=WID2 (WID=4, VECL=16, or WID=8, VECL=64, then j==0)
         // This loop is still needed for e.g. Warp=VECL=32, WID2=64 (then j==0 or 4)
         const vmesh::LocalID nblocks = gpu_columnData->columnNumBlocks[column];

         const uint i_indices = w_tid % WID;
         const uint j_indices = j + w_tid/WID;
         //int jk = j / (VECL/WID);

         const int target_cell_index_common =
            i_indices * gpu_cell_indices_to_id[0] +
            j_indices * gpu_cell_indices_to_id[1];
         const Realf intersection_min =
            intersection +
            (gpu_columnData->i[column] * WID + (Realf)i_indices) * intersection_di +
            (gpu_columnData->j[column] * WID + (Realf)j_indices) * intersection_dj;

         const Realf gk_intersection_min =
            intersection +
            (gpu_columnData->i[column] * WID + (Realf)( intersection_di > 0 ? 0 : WID-1 )) * intersection_di +
            (gpu_columnData->j[column] * WID + (Realf)( intersection_dj > 0 ? j : j+VECL/WID-1 )) * intersection_dj;
         const Realf gk_intersection_max =
            intersection +
            (gpu_columnData->i[column] * WID + (Realf)( intersection_di < 0 ? 0 : WID-1 )) * intersection_di +
            (gpu_columnData->j[column] * WID + (Realf)( intersection_dj < 0 ? j : j+VECL/WID-1 )) * intersection_dj;

         // loop through all perpendicular slices in column and compute the mapping as integrals.
         for (uint k=0; k < WID * nblocks; ++k) {
            // Compute reconstructions
            // Checked on 21.01.2022: Realf a[length] goes on the register despite being an array. Explicitly declaring it
            // as __shared__ had no impact on performance.
            size_t valuesOffset = (gpu_columnData->columnBlockOffsets[column] + 2*column) * (WID3/VECL); // there are WID3/VECL elements of type Vec per block
#ifdef ACC_SEMILAG_PLM
            Realf a[2];
            compute_plm_coeff(gpu_blockDataOrdered + valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), (k + WID), a, minValue, w_tid);
#endif
#ifdef ACC_SEMILAG_PPM
            Realf a[3];
            compute_ppm_coeff(gpu_blockDataOrdered + valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), h4, (k + WID), a, minValue, w_tid);
#endif
#ifdef ACC_SEMILAG_PQM
            Realf a[5];
            compute_pqm_coeff(gpu_blockDataOrdered + valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), h8, (k + WID), a, minValue, w_tid);
#endif

            // set the initial value for the integrand at the boundary at v = 0
            // (in reduced cell units), this will be shifted to target_density_1, see below.
            Realf target_density_r = 0.0;

            const Realf v_r = v_r0  + (k+1)* dv;
            const Realf v_l = v_r0  + k* dv;
            const int lagrangian_gk_l = trunc((v_l-gk_intersection_max)/intersection_dk);
            const int lagrangian_gk_r = trunc((v_r-gk_intersection_min)/intersection_dk);

            //limits in lagrangian k for target column. Also take into
            //account limits of target column
            // Now all w_tids in the warp should have the same gk loop extents
            const int minGk = max(lagrangian_gk_l, int(gpu_columnData->minBlockK[column] * WID));
            const int maxGk = min(lagrangian_gk_r, int((gpu_columnData->maxBlockK[column] + 1) * WID - 1));
            // Run along the column and perform the polynomial reconstruction
            for(int gk = minGk; gk <= maxGk; gk++) {
               const int blockK = gk/WID;
               const int gk_mod_WID = (gk - blockK * WID);

               //the block of the Lagrangian cell to which we map
               //const int target_block(target_block_index_common + blockK * block_indices_to_id[2]);
               // This already contains the value index via target_cell_index_commom
               const int tcell(target_cell_index_common + gk_mod_WID * gpu_cell_indices_to_id[2]);
               //the velocity between which we will integrate to put mass
               //in the targe cell. If both v_r and v_l are in same cell
               //then v_1,v_2 should be between v_l and v_r.
               //v_1 and v_2 normalized to be between 0 and 1 in the cell.
               //For vector elements where gk is already larger than needed (lagrangian_gk_r), v_2=v_1=v_r and thus the value is zero.
               const Realf v_norm_r = (  min(  max( (gk + 1) * intersection_dk + intersection_min, v_l), v_r) - v_l) * i_dv;

               /*shift, old right is new left*/
               const Realf target_density_l = target_density_r;

               // compute right integrand
#ifdef ACC_SEMILAG_PLM
               target_density_r = v_norm_r * ( a[0] + v_norm_r * a[1] );
#endif
#ifdef ACC_SEMILAG_PPM
               target_density_r = v_norm_r * ( a[0] + v_norm_r * ( a[1] + v_norm_r * a[2] ) );
#endif
#ifdef ACC_SEMILAG_PQM
               target_density_r = v_norm_r * ( a[0] + v_norm_r * ( a[1] + v_norm_r * ( a[2] + v_norm_r * ( a[3] + v_norm_r * a[4] ) ) ) );
#endif

               //store value
               Realf tval = target_density_r - target_density_l;

               const int targetBlock =
                  gpu_columnData->i[column] * gpu_block_indices_to_id[0] +
                  gpu_columnData->j[column] * gpu_block_indices_to_id[1] +
                  blockK                * gpu_block_indices_to_id[2];
               const vmesh::LocalID tblockLID = vmesh->getLocalID(targetBlock);
               // Using a warp search here seems to get only partial warp masks, resulting in an error
               //const vmesh::LocalID tblockLID = vmesh->warpGetLocalID(targetBlock, w_tid);
               if (isfinite(tval) && (tval>0) && (tblockLID != invalidLID) ) {
                  (&gpu_blockData[tblockLID*WID3])[tcell] += tval;
               }
            } // for loop over target k-indices of current source block
         } // for-loop over source blocks
      } //for loop over j index
   } // End this column
} // end semilag acc kernel




/*
   Here we map from the current time step grid, to a target grid which
   is the lagrangian departure grid (so th grid at timestep +dt,
   tracked backwards by -dt)
*/
__host__ bool gpu_acc_map_1d(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   spatial_cell::SpatialCell* spatial_cell,
   const uint popID,
   const Realf intersection,
   const Realf intersection_di,
   const Realf intersection_dj,
   const Realf intersection_dk,
   const uint dimension,
   const int Dacc, // velocity block max dimension, direction of acceleration
   const int Dother, // Product of other two dimensions (max blocks)
   const size_t cellIndex
   ) {
   gpuStream_t stream = gpu_getStream();
   // Thread id used for persistent device memory pointers
   const uint cpuThreadID = gpu_getThread();

   const vector<CellID> cellsToAdjust {(CellID)spatial_cell->parameters[CellParams::CELLID]};
   const uint cellOffset = (uint)cellIndex;
   const uint nCells = 1;

   vmesh::VelocityMesh* vmesh    = spatial_cell->get_velocity_mesh(popID);
   vmesh::VelocityBlockContainer* blockContainer = spatial_cell->get_velocity_blocks(popID);

   //nothing to do if no blocks
   vmesh::LocalID nBlocksBeforeAdjust = vmesh->size();
   if (nBlocksBeforeAdjust == 0) {
      return true;
   }

   // These are only used for host-side method calls
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map_require = spatial_cell->velocity_block_with_content_map;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map_remove = spatial_cell->velocity_block_with_no_content_map;
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new = spatial_cell->list_with_replace_new;

   auto minValue = spatial_cell->getVelocityBlockMinValue(popID);
   // These query velocity mesh parameters which are duplicated for both host and device
   const vmesh::LocalID D0 = vmesh->getGridLength()[0];
   const vmesh::LocalID D1 = vmesh->getGridLength()[1];
   const vmesh::LocalID D2 = vmesh->getGridLength()[2];
   const Realf dv    = vmesh->getCellSize()[dimension];
   const Realf v_min = vmesh->getMeshMinLimits()[dimension];
   const int max_v_length  = (int)vmesh->getGridLength()[dimension];
   const Realf i_dv = 1.0/dv;

   // Some kernels in here require the number of threads to be equal to VECL.
   // Future improvements would be to allow setting it directly to WID3.
   // Other kernels (not handling block data) can use GPUTHREADS which
   // is equal to NVIDIA: 32 or AMD: 64.

   /** New merged kernel approach without sorting for columns

      First, we generate a "probe cube". It started off as an actual
      cube, but the transverse dimensions are considered as one.
      One dimension is that of the current acceleration (Dacc), and the other
      dimension is the product of the other two maximal velocity block
      domain extents (Dother). The "flattened" version is one where data is gathered
      over the acceleration direction into a single value.
   */

   // The flattened version of the probe cube must store:
   // 1) how many columns per potential column position (potColumn)
   // 2) how many blocks per potColumn
   // 3) cumulative offset into columns per potColumn
   // 4) cumulative offset into columnSets per potColumn
   // 5) cumulative offset into blocks per potColumn
   // For reductions, each slice of the flattened array should have a size a multiple of 2*MAX_BLOCKSIZE:
   const size_t flatExtent = 2*Hashinator::defaults::MAX_BLOCKSIZE * (1 + ((Dother - 1) / (2*Hashinator::defaults::MAX_BLOCKSIZE)));

   // Pointers to device memory buffers:
   // probe cube and flattened version now re-use gpu_blockDataOrdered[cpuThreadID].
   // Due to alignment, Flattened version is at start of buffer, followed by the cube.
   vmesh::LocalID *probeFlattened = reinterpret_cast<vmesh::LocalID*>(gpu_blockDataOrdered[cpuThreadID]);
   vmesh::LocalID *probeCube = reinterpret_cast<vmesh::LocalID*>(gpu_blockDataOrdered[cpuThreadID]) + flatExtent*GPU_PROBEFLAT_N;
   /**
      For the gathered LIDlist, we re-use the allocation of spatial_cell->dev_velocity_block_with_content_list,
      It which contains variables of type vmesh::GlobalID (which should be the same as vmesh::LocalID, uint_32t).
      To ensure this static_cast is safe, we verify the sizes.
   */
   if (sizeof(vmesh::LocalID) != sizeof(vmesh::GlobalID)) {
      string message = " ERROR! vmesh::LocalID and vmesh::GlobalID are of different sizes, and thus";
      message += " the acceleration solver cannot safely use the spatial_cell->dev_list_delete";
      message += " Hashinator::splitVector object for storing a list of LIDs.";
      bailout(true, message, __FILE__, __LINE__);
   }

   // Columndata has copies on both host and device containing splitvectors with unified memory
   ColumnOffsets *columnData = gpu_columnOffsetData[cpuThreadID];

   // Fill probe cube vmesh invalid LID values, flattened array with zeros
   CHK_ERR( gpuMemsetAsync(probeFlattened, 0, flatExtent*GPU_PROBEFLAT_N*sizeof(vmesh::LocalID),stream) );
   const size_t grid_fill_invalid = 1 + ((Dacc*Dother - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   fill_probe_invalid<<<grid_fill_invalid,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      probeCube,
      Dacc*Dother,
      vmesh->invalidLocalID(),
      // Pass vectors for clearing
      dev_lists_with_replace_new,
      dev_lists_delete,
      dev_lists_to_replace,
      dev_lists_with_replace_old,
      dev_vbwcl_vec, // dev_velocity_block_with_content_list, // Resize to use as LIDlist
      nBlocksBeforeAdjust,
      cellOffset
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Read in GID list from vmesh, store LID values into probe cube in correct order
   // Launch params, fast ceil for positive ints
   const size_t grid_fill_ord = 1 + ((nBlocksBeforeAdjust - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   fill_probe_ordered<<<grid_fill_ord,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      dev_vmeshes,
      probeCube,
      gpu_block_indices_to_probe,
      nBlocksBeforeAdjust,
      cellOffset
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Now we perform reductions / flattenings / scans of the probe cube.
   // The kernel loops over the acceleration direction (Dacc).
   const size_t grid_cube = 1 + ((Dother - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   flatten_probe_cube<<<grid_cube,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      probeCube,
      probeFlattened,
      Dacc,
      Dother,
      flatExtent,
      vmesh->invalidLocalID()
      );
   CHK_ERR( gpuPeekAtLastError() );

   // This kernel performs an exclusive prefix scan to get offsets for storing
   // data from potential columns into the columnData container. Also gives us the total
   // counts of columns, columnsets, and blocks, and uses the first two to resize
   // our splitvector containers inside columnData.

   // A proper prefix scan needs to be a two-phase process, thus two kernels,
   // but here we do an iterative loop processing MAX_BLOCKSIZE elements at once.
   // Not as efficient but simpler, and will be parallelized over spatial cells.
   scan_probe<<<1,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      probeFlattened,
      Dacc,
      Dother,
      flatExtent,
      nBlocksBeforeAdjust,
      returnLID[cpuThreadID],
      columnData
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Copy back to host sizes of found columns etc
   CHK_ERR( gpuMemcpyAsync(host_returnLID[cpuThreadID], returnLID[cpuThreadID], 3*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, stream) );
   CHK_ERR( gpuStreamSynchronize(stream) );
   // Read count of columns and columnsets, calculate required size of buffers
   const vmesh::LocalID host_totalColumns = host_returnLID[cpuThreadID][0];
   const vmesh::LocalID host_totalColumnSets = host_returnLID[cpuThreadID][1];
   const vmesh::LocalID host_recapacitateVectors = host_returnLID[cpuThreadID][2];
   const vmesh::LocalID host_valuesSizeRequired = (nBlocksBeforeAdjust + 2*host_totalColumns) * WID3 / VECL;
   if (host_recapacitateVectors) {
      // Can't call CPU reallocation directly as then GPU/CPU copies go out of sync
      gpu_acc_allocate_perthread(cpuThreadID, host_totalColumns, host_totalColumnSets);
   }
   // Now we have gathered all the required offsets into probeFlattened, and can
   // now launch a kernel which constructs the columns offsets in parallel.
   build_column_offsets<<<grid_cube,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      dev_vmeshes,
      probeCube,
      probeFlattened,
      D0,D1,D2,
      dimension,
      flatExtent,
      vmesh->invalidLocalID(),
      columnData,
      dev_vbwcl_vec, //dev_velocity_block_with_content_list, // use as LIDlist
      cellOffset
      //(size_t)spatial_cell->SpatialCell::parameters[CellParams::CELLID] //can be passed for debug purposes
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Launch kernels for transposing and ordering velocity space data into columns
   reorder_blocks_by_dimension_kernel<<<host_totalColumns, VECL, 0, stream>>> (
      dev_VBCs,
      gpu_blockDataOrdered[cpuThreadID],
      gpu_cell_indices_to_id,
      dev_vbwcl_vec, //dev_velocity_block_with_content_list, // use as LIDlist
      columnData,
      host_valuesSizeRequired,
      cellOffset
      );
   CHK_ERR( gpuPeekAtLastError() );
   //CHK_ERR( gpuStreamSynchronize(stream) );

   // Calculate target column extents
   do {
      CHK_ERR( gpuMemsetAsync(returnLID[cpuThreadID], 0, 2*sizeof(vmesh::LocalID), stream) );
      map_require->clear<false>(Hashinator::targets::device,stream,std::pow(2,spatial_cell->vbwcl_sizePower));
      map_remove->clear<false>(Hashinator::targets::device,stream,std::pow(2,spatial_cell->vbwncl_sizePower));
      // Hashmap clear includes a stream sync
      //CHK_ERR( gpuStreamSynchronize(stream) );
      evaluate_column_extents_kernel<<<host_totalColumnSets, GPUTHREADS, 0, stream>>> (
         dimension,
         dev_vmeshes,
         columnData,
         dev_lists_with_replace_new,
         dev_allMaps,
         gpu_block_indices_to_id,
         intersection,
         intersection_di,
         intersection_dj,
         intersection_dk,
         Parameters::bailout_velocity_space_wall_margin,
         max_v_length,
         v_min,
         dv,
         returnLID[cpuThreadID], //gpu_bailout_flag:
                                 // - element[0]: touching velspace wall
                                 // - element[1]: splitvector list_with_replace_new capacity error
         cellOffset
         );
      CHK_ERR( gpuPeekAtLastError() );
      // Check if we need to bailout due to hitting v-space edge
      CHK_ERR( gpuMemcpyAsync(host_returnLID[cpuThreadID], returnLID[cpuThreadID], 2*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, stream) );
      CHK_ERR( gpuStreamSynchronize(stream) );
      if (host_returnLID[cpuThreadID][0] != 0) { //host_wallspace_margin_bailout_flag
         string message = "Some target blocks in acceleration are going to be less than ";
         message += std::to_string(Parameters::bailout_velocity_space_wall_margin);
         message += " blocks away from the current velocity space walls for population ";
         message += getObjectWrapper().particleSpecies[popID].name;
         message += " at CellID ";
         message += std::to_string(spatial_cell->parameters[CellParams::CELLID]);
         message += ". Consider expanding velocity space for that population.";
         bailout(true, message, __FILE__, __LINE__);
      }

      // Check whether we exceeded the column data splitVectors on the way
      if (host_returnLID[cpuThreadID][1] != 0) {
         // If so, recapacitate and try again. We'll take at least our current velspace size
         // (plus safety factor), or, if that wasn't enough, twice what we had before.
         size_t newCapacity = (size_t)(spatial_cell->getReservation(popID)*BLOCK_ALLOCATION_FACTOR);
         //printf("column data recapacitate! %lu newCapacity\n",(long unsigned)newCapacity);
         list_with_replace_new->clear();
         spatial_cell->setReservation(popID, newCapacity);
         spatial_cell->applyReservation(popID);
      }
      // Loop until we return without an out-of-capacity error
   } while (host_returnLID[cpuThreadID][1] != 0);

   /** Use block adjustment callers / lambda rules for extracting required map contents,
       building up vectors to use for parallel adjustment
   */

   // Finds Blocks (GID,LID) to be rescued from end of v-space
   extract_to_delete_or_move_caller(
      dev_allMaps+2*cellOffset, //dev_has_content_maps, // input maps
      dev_lists_with_replace_old+cellOffset, // output vecs
      dev_contentSizes+5*cellOffset, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      dev_vmeshes+cellOffset, // rule_meshes
      dev_allMaps+2*cellOffset+1, //dev_has_no_content_maps// rule_maps
      dev_lists_with_replace_new+cellOffset, // rule_vectors
      nCells,
      stream
      );
   // Find Blocks (GID,LID) to be outright deleted
   extract_to_delete_or_move_caller(
      dev_allMaps+2*cellOffset+1,//dev_has_no_content_maps, // input maps
      dev_lists_delete+cellOffset, // output vecs
      dev_contentSizes+5*cellOffset, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      dev_vmeshes+cellOffset, // rule_meshes
      dev_allMaps+2*cellOffset+1, //dev_has_no_content_maps, // rule_maps
      dev_lists_with_replace_new+cellOffset, // rule_vectors
      nCells,
      stream
      );
   // Find Blocks (GID,LID) to be replaced with new ones
   extract_to_replace_caller(
      dev_allMaps+2*cellOffset+1,//dev_has_no_content_maps, // input maps
      dev_lists_to_replace+cellOffset, // output vecs
      dev_contentSizes+5*cellOffset, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      dev_vmeshes+cellOffset, // rule_meshes
      dev_allMaps+2*cellOffset+1,//dev_has_no_content_maps, // rule_maps
      dev_lists_with_replace_new+cellOffset, // rule_vectors
      nCells,
      stream
      );
   //CHK_ERR( gpuStreamSynchronize(stream) );

   // Note: in this call, unless hitting v-space walls, we only grow the vspace size
   // and thus do not delete blocks or replace with old blocks. The call now uses the
   // batch block adjust interface.
   uint largestBlocksToChange; // Not needed
   uint largestBlocksBeforeOrAfter; // Not needed
   batch_adjust_blocks_caller_nonthreaded(
      mpiGrid,
      cellsToAdjust,
      cellOffset,
      largestBlocksToChange,
      largestBlocksBeforeOrAfter,
      popID);
   const vmesh::LocalID nBlocksAfterAdjust = host_contentSizes[1 + 5*cellOffset];

   // Velocity space has now all extra blocks added and/or removed for the transform target
   // and will not change shape anymore.
   spatial_cell->largestvmesh = spatial_cell->largestvmesh > nBlocksAfterAdjust ? spatial_cell->largestvmesh : nBlocksAfterAdjust;

   // Zero out target data on device (unified). Note: pointer needs to be re-fetched
   // here in case of reallocation if VBC size was increased during block adjustment.
   //GPUTODO: direct access to blockContainer getData causes page fault
   Realf *blockData = blockContainer->getData();
   CHK_ERR( gpuMemsetAsync(blockData, 0, nBlocksAfterAdjust*WID3*sizeof(Realf), stream) );

   // GPUTODO: Adapt to work as VECL=WID3 instead of VECL=WID2
   acceleration_kernel<<<host_totalColumns, VECL, 0, stream>>> (
      dev_vmeshes,
      dev_VBCs,
      gpu_blockDataOrdered[cpuThreadID],
      gpu_cell_indices_to_id,
      gpu_block_indices_to_id,
      columnData,
      intersection,
      intersection_di,
      intersection_dj,
      intersection_dk,
      v_min,
      i_dv,
      dv,
      minValue,
      vmesh->invalidLocalID(),
      cellOffset
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(stream) );

   return true;
}
