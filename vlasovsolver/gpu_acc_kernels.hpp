#define i_pcolumnv_gpu(j, k, k_block, num_k_blocks) ( ((j) / ( VECL / WID)) * WID * ( num_k_blocks + 2) + (k) + ( k_block + 1 ) * WID )
#define i_pcolumnv_gpu_b(planeVectorIndex, k, k_block, num_k_blocks) ( planeVectorIndex * WID * ( num_k_blocks + 2) + (k) + ( k_block + 1 ) * WID )


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_acc_map.hpp"
#include "gpu_acc_sort_blocks.hpp"
#include "vec.h"
#include "../definitions.h"
#include "../object_wrapper.h"
#include "../arch/gpu_base.hpp"
#include "../spatial_cells/spatial_cell_gpu.hpp"
#include "cpu_face_estimates.hpp"
#include "cpu_1d_pqm.hpp"
#include "cpu_1d_ppm.hpp"
#include "cpu_1d_plm.hpp"

#ifdef DEBUG_VLASIATOR
   #ifndef DEBUG_ACC
   #define DEBUG_ACC
   #endif
#endif


using namespace std;
using namespace spatial_cell;

/* Fills the target probe block with the invalid value for vmesh::LocalID
   Also clears provided vectors
*/
__global__ void fill_probe_invalid(
   vmesh::LocalID *probeCube,
   const size_t nTot,
   const vmesh::LocalID invalid,
   // Pass these for emptying
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_delete,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_to_replace,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_with_replace_old
   ) {
   const size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = ind; i < nTot; i += gridDim.x * blockDim.x) {
      probeCube[i] = invalid;
   }
   if (ind==0) {
      list_with_replace_new->clear();
      list_delete->clear();
      list_to_replace->clear();
      list_with_replace_old->clear();
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
   const vmesh::VelocityMesh* __restrict__ vmesh,
   vmesh::LocalID *probeCube,
   const vmesh::LocalID D0,
   const vmesh::LocalID D1,
   const vmesh::LocalID D2,
   const vmesh::LocalID nBlocks,
   const int dimension
   ) {
   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const vmesh::LocalID LID = blockDim.x * blockIdx.x + ti;
   if (LID >= nBlocks) {
      return;
   }
   // Store in probe cube with ordering so that reading will be fast
   const vmesh::GlobalID GID = vmesh->getGlobalID(LID);
   vmesh::LocalID indices[3];
   vmesh->getIndices(GID,indices[0],indices[1],indices[2]);
   int target;
   switch (dimension) {
      case 0:
         // propagate along X, flatten Y+Z
         target = indices[0]*D1*D2 + indices[1]*D2 + indices[2];
         break;
      case 1:
         // propagate along Y, flatten X+Z
         target = indices[1]*D0*D2 + indices[0]*D2 + indices[2];
         break;
      case 2:
         // propagate along Z, flatten X+Y
         target = indices[2]*D0*D1 + indices[0]*D1 + indices[1];
         break;
      default:
         return;
   }
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

/* This mini-kernel simply sums the probe cube flattening
   results to find out how many columns and how many columnsets we need.
*/
__global__ void reduce_probe_A(
   const vmesh::LocalID* __restrict__ probeFlattened,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   const vmesh::LocalID nBlocks,
   vmesh::LocalID *dev_returnLID,
   ColumnOffsets* columnData
   ) {
   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const int blockSize = blockDim.x; // Hashinator::defaults::MAX_BLOCKSIZE

   // Per-thread counters in shared memory for reduction.
   __shared__ vmesh::LocalID reductionA[Hashinator::defaults::MAX_BLOCKSIZE];
   __shared__ vmesh::LocalID reductionB[Hashinator::defaults::MAX_BLOCKSIZE];
   __shared__ vmesh::LocalID reductionC[Hashinator::defaults::MAX_BLOCKSIZE];
   reductionA[ti] = 0;
   reductionB[ti] = 0;
   reductionC[ti] = 0;

   // Fill reduction arrays
   int i = ti;
   while (i < Dother) {
      reductionA[ti] += probeFlattened[i]; // Number of columns per potColumn
      reductionB[ti] += (probeFlattened[i] != 0 ? 1 : 0); // Number of columnSets per potColumn
      reductionC[ti] += probeFlattened[flatExtent+i]; // Number of blocks per potColumn
      i += blockSize;
   }
   __syncthreads();
   for (unsigned int s=blockSize/2; s>0; s>>=1) {
      if (ti < s) {
         reductionA[ti] += reductionA[ti + s];
         reductionB[ti] += reductionB[ti + s];
         reductionC[ti] += reductionC[ti + s];
      }
      __syncthreads();
   }
   __syncthreads();

   // Todo: unrolling  of reduction loops to get even more performance.
   // Also can do several loads on first adds of reduction.
   // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

   if (ti == 0) {
      //printf("found columns %d and columnsets %d, %d blocks vs %d\n",reductionA[0],reductionB[0],reductionC[0],nBlocks);
      // Store reduction results
      dev_returnLID[0] = reductionA[0]; // Total number of columns
      dev_returnLID[1] = reductionB[0]; // Total number of columnSets
      // Resize device-side column offset container vectors
      columnData->columnBlockOffsets.device_resize(reductionA[0]);
      columnData->columnNumBlocks.device_resize(reductionA[0]);
      columnData->setColumnOffsets.device_resize(reductionB[0]);
      columnData->setNumColumns.device_resize(reductionB[0]);
   }
}


/* This kernel performs exclusive prefix scans of the flattened probe cube.
   Produces:

   (1) the cumulative sum of columns up to the beginning of each potColumn
   (2) the cumulative sum of column sets up to the beginning of each potColumn
   (3) the cumulative sum of blocks up to the beginning of each potColumn

   This is not a fully optimized scan as it uses only a single block
   in order to perform it all in one kernel. As the fixed-size shared memory
   buffer gets overwritten each cycle, we store the actual cumulative sums
   into the third and fourth entries in the probeFlattened buffer.
*/
//#define NUM_BANKS 16 // Defined in splitvector headers
//#define LOG_NUM_BANKS 4
//#ifdef ZERO_BANK_CONFLICTS
// #define CONFLICT_FREE_OFFSET(n) \
//      ((n) >> (LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))
//#else
//#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
//#define CONFLICT_FREE_OFFSET(n) 0
//#endif
#define BANK_OFFSET(n) 0

__global__ void scan_probe_A(
   vmesh::LocalID *probeFlattened,
   const vmesh::LocalID Dacc,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   const vmesh::LocalID nBlocks // For early exit
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
      if (ti==0) {
         offsetA += reductionA[n-1] + probeFlattened[majorOffset-1];
         offsetB += reductionB[n-1] + (probeFlattened[majorOffset-1] != 0 ? 1 : 0);
         offsetC += reductionC[n-1] + probeFlattened[flatExtent + majorOffset-1];
      }
      __syncthreads();
      // if (ti==0) {
      //    printf("majoroffset %lu cumulative offsets A (columns) %u, B (coulmnsets) %u, C (blocks) %u\n",majorOffset,offsetA,offsetB,offsetC);
      // }
      // __syncthreads();
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
   const vmesh::VelocityMesh* __restrict__ vmesh,
   const vmesh::LocalID* __restrict__ probeCube,
   const vmesh::LocalID* __restrict__ probeFlattened,
   const vmesh::LocalID Dacc,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   const vmesh::LocalID invalid,
   ColumnOffsets* columnData,
   vmesh::GlobalID *GIDs,
   vmesh::LocalID *LIDs
   ) {
   // Probe cube contents have been ordered based on acceleration dimesion
   // so this kernel always reads in the same way.

   const int ti = threadIdx.x; // [0,Hashinator::defaults::MAX_BLOCKSIZE)
   const vmesh::LocalID ind = blockDim.x * blockIdx.x + ti;

   // potColumn indexes/offsets into columnData and LIDs/GIDs
   const vmesh::LocalID N_cols = probeFlattened[ind];
   //const vmesh::LocalID N_blocks = probeFlattened[flatExtent + ind];
   const vmesh::LocalID offset_cols = probeFlattened[2*flatExtent + ind];
   const vmesh::LocalID offset_colsets = probeFlattened[3*flatExtent + ind];
   const vmesh::LocalID offset_blocks = probeFlattened[4*flatExtent + ind];

   if (ind < Dother) {
      if (N_cols != 0) {
         // Update values in columnSets vector
         //printf("ind %d    offset_colsets %d     offset_cols %d    Ncols %d\n",ind,offset_colsets,offset_cols,N_cols);
         columnData->setColumnOffsets.at(offset_colsets) = offset_cols;
         columnData->setNumColumns.at(offset_colsets) = N_cols;
      }
      // Per-thread counters
      vmesh::LocalID foundBlocks = 0;
      vmesh::LocalID foundBlocksThisCol = 0;
      vmesh::LocalID foundCols = 0;
      bool inCol = false;

      // Loop through acceleration dimension of cube
      for (vmesh::LocalID j = 0; j < Dacc; j++) {
         // Early return when all columns have been completed
         if (foundCols >= N_cols) {
            return;
         }
         const vmesh::LocalID LID = probeCube[j*Dother + ind];
         if (LID == invalid) {
            // No block at this index.
            if (inCol) {
               // finish current column
               columnData->columnNumBlocks.at(offset_cols + foundCols) = foundBlocksThisCol;
               //printf("ind %d    col %d+%d = %d    blocks %d\n",ind,offset_cols,foundCols,offset_cols + foundCols,foundBlocksThisCol);
               foundCols++;
               inCol = false;
            }
         } else {
            // Valid block found at this index!
            // Store GID and LID into buffers
            LIDs[offset_blocks + foundBlocks] = LID;
            const vmesh::GlobalID GID = vmesh->getGlobalID(LID);;
            GIDs[offset_blocks + foundBlocks] = GID;
            //printf("GID %d LID %d offset_blocks %d foundblocks %d\n",GID,LID,offset_blocks,foundBlocks);
            if (!inCol) {
               // start new column
               inCol = true;
               foundBlocksThisCol = 0;
               columnData->columnBlockOffsets.at(offset_cols + foundCols) = offset_blocks + foundBlocks;
               //printf("ind %d    col %d+%d = %d    blocks-offset %d+%d = %d\n",ind,offset_cols,foundCols,offset_cols + foundCols,offset_blocks,foundBlocks,offset_blocks + foundBlocks);
            }
            foundBlocks++;
            foundBlocksThisCol++;
         }
      }
      // Finished loop. If we are "still in a colum", count that.
      if (inCol) {
         columnData->columnNumBlocks.at(offset_cols + foundCols) = foundBlocksThisCol;
         //printf("ind %d    col %d+%d = %d     blocks %d\n",ind,offset_cols,foundCols,offset_cols + foundCols,foundBlocksThisCol);
         //foundCols++:
      }
   }
}

// debug kernel: print probeflattened, columnData
__global__ void print_debug_kernel(
   const vmesh::VelocityMesh* __restrict__ vmesh,
   const vmesh::LocalID* __restrict__ probeFlattened,
   const vmesh::LocalID Dother,
   const size_t flatExtent,
   ColumnOffsets* columnData,
   vmesh::GlobalID *GIDs,
   vmesh::LocalID *LIDs,
   vmesh::LocalID nBlocks
   ) {
   vmesh::LocalID nCols = columnData->columnNumBlocks.size();
   vmesh::LocalID nColSets = columnData->setNumColumns.size();
   vmesh::LocalID DotherSq = sqrt(Dother);
   printf("\n\ncolumnData: %d columns, %d columnSets\n",nCols,nColSets);
   for (int i=0; i<nCols; ++i) {
      printf("  I=%3d    ColumnBlockOffsets %5u nBlocks %5u\n",i,columnData->columnBlockOffsets.at(i),columnData->columnNumBlocks.at(i));
   }
   for (int i=0; i<nColSets; ++i) {
      printf("  J=%3d    setColumnOffsets %5u setNumColumns %5u\n",i,columnData->setColumnOffsets.at(i),columnData->setNumColumns.at(i));
   }
   printf("\n\nprobeFlattened, size %u with flatExtent %lu\n",Dother,flatExtent);
   for (int i=0; i<5; ++i) {
      for (int j=0; j<Dother; ++j) {
         printf("%3u ",probeFlattened[i*flatExtent+j]);
         if (j%DotherSq==Dother-1) {
            printf("\n");
         }
      }
      printf("\n\n");
   }
   printf("\n\nGIDs and LIDs in order\n");
   for (int i=0; i<nBlocks; ++i) {
      vmesh::GlobalID fGID = vmesh->getGlobalID(LIDs[i]);
      printf("   (%5d)   GID %5u    LID %5u",i,GIDs[i],LIDs[i]);
      if (fGID!=GIDs[i]) {
         printf("  MM! %5u",fGID);
      }
      printf("\n");
   }
   printf("\n\n");
}
   // Probe cube contents have been ordered based on acceleration dimesion
   // so this kernel always reads in the same way.


   // definition: potColumn is a potential column(set), i.e. a stack from the probe cube.

   // With prefix sum, we gather:
   // (1) the cumulative sum of column sets up to the beginning of each potColumn
   // (2) the cumulative sum of columns up to the beginning of each potColumn
   // (3) the cumulative sum of blocks up to the beginning of each potColumn

   // We can then fill in the values of columnData:
   // Each potColumn in the flattened probe array has now an index (1) into setNumColumns & setColumnOffsets
   // (the cumulative sum of nonzeor columnSets up to that point).

   // Thus: with that index (1), store how many columns are in that set into setNumColumns.
   // (only that number is larger than zero - ignores empty potColumns)

   // Also, with that index (1) store the cumulative count of columns up to that potColum (2)
   // into setColumnOffsets
   // (only if the current potColumn is not empty).


   // After this, we need to do another pass-through loop over the probeCube.
   // For each potColum, we:
   // a) Store the number of blocks in each column into
   //    columnNumBlocks[setColumnOffsets + localColumnIndex]
   // b) Store the cumulative block offset (count) for each column into
   //    columnBlockOffsets[setColumnOffsets + localColumnIndex]
   // c) Store the GID and LID for each block in each column into
   //    provided buffers, using the value of b) and the process through that column.




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

__global__ void __launch_bounds__(VECL,4) reorder_blocks_by_dimension_kernel(
   const vmesh::VelocityBlockContainer* __restrict__ blockContainer,
   Vec *gpu_blockDataOrdered,
   const uint* __restrict__ gpu_cell_indices_to_id,
   const vmesh::LocalID* __restrict__ gpu_LIDlist,
   const ColumnOffsets* __restrict__ columnData,
   const vmesh::LocalID valuesSizeRequired
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
   // Each gpuBlock deals with one column.
   {
      const uint inputOffset = columnData->columnBlockOffsets.at(iColumn);
      const uint outputOffset = (inputOffset + 2 * iColumn) * (WID3/VECL);
      const uint columnLength = columnData->columnNumBlocks.at(iColumn);

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
               const vmesh::LocalID LID = gpu_LIDlist[inputOffset + b];
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

// Serial kernel only to avoid page faults or prefetches
__global__ void __launch_bounds__(1,4) count_columns_kernel (
   const ColumnOffsets* __restrict__ gpu_columnData,
   vmesh::LocalID* returnLID, // gpu_totalColumns, gpu_valuesSizeRequired
   // Pass vectors for clearing
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_delete,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_to_replace,
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* list_with_replace_old
   ) {
   // const int gpuBlocks = gridDim.x * gridDim.y * gridDim.z;
   // const int warpSize = blockDim.x * blockDim.y * blockDim.z;
   const int blocki = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
   const int ti = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
   if ((blocki==0)&&(ti==0)) {
      for(uint setIndex=0; setIndex< gpu_columnData->setColumnOffsets.size(); ++setIndex) {
         returnLID[0] += gpu_columnData->setNumColumns[setIndex];
         for(uint columnIndex = gpu_columnData->setColumnOffsets[setIndex]; columnIndex < gpu_columnData->setColumnOffsets[setIndex] + gpu_columnData->setNumColumns[setIndex] ; columnIndex ++){
            returnLID[1] += (gpu_columnData->columnNumBlocks[columnIndex] + 2) * WID3 / VECL;
         }
      }
   }
   // Also clear these vectors
   if ((blocki==0)&&(ti==0)) {
      list_with_replace_new->clear();
      list_delete->clear();
      list_to_replace->clear();
      list_with_replace_old->clear();
   }
}

// Serial kernel only to avoid page faults or prefetches
__global__ void __launch_bounds__(1,4) offsets_into_columns_kernel(
   const ColumnOffsets* __restrict__ gpu_columnData,
   Column *gpu_columns,
   const uint valuesSizeRequired
   ) {
   // const int gpuBlocks = gridDim.x * gridDim.y * gridDim.z;
   // const int warpSize = blockDim.x * blockDim.y * blockDim.z;
   const int blocki = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
   const int ti = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
   if ((blocki==0)&&(ti==0)) {
      uint valuesColumnOffset = 0;
      for( uint setIndex=0; setIndex< gpu_columnData->setColumnOffsets.size(); ++setIndex) {
         for (uint columnIndex = gpu_columnData->setColumnOffsets[setIndex]; columnIndex < gpu_columnData->setColumnOffsets[setIndex] + gpu_columnData->setNumColumns[setIndex] ; columnIndex ++){
            gpu_columns[columnIndex].nblocks = gpu_columnData->columnNumBlocks[columnIndex];
            gpu_columns[columnIndex].valuesOffset = valuesColumnOffset;
            if (valuesColumnOffset >= valuesSizeRequired) {
               printf("(ERROR: Overflowing the values array (%d > %d) with column %d\n",valuesColumnOffset,valuesSizeRequired,columnIndex);
            }
            valuesColumnOffset += (gpu_columnData->columnNumBlocks[columnIndex] + 2) * (WID3/VECL); // there are WID3/VECL elements of type Vec per block
         }
      }
   }
}

// Using columns, evaluate which blocks are target or source blocks
__global__ void __launch_bounds__(GPUTHREADS,4) evaluate_column_extents_kernel(
   const uint dimension,
   const vmesh::VelocityMesh* __restrict__ vmesh,
   const ColumnOffsets* __restrict__ gpu_columnData,
   Column *gpu_columns,
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_require,
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_remove,
   const vmesh::GlobalID* __restrict__ GIDlist,
   const uint* __restrict__ gpu_block_indices_to_id,
   const Realf intersection,
   const Realf intersection_di,
   const Realf intersection_dj,
   const Realf intersection_dk,
   const int bailout_velocity_space_wall_margin,
   const int max_v_length,
   const Realf v_min,
   const Realf dv,
   uint *bailout_flag
   ) {
   const uint warpSize = blockDim.x * blockDim.y * blockDim.z;
   const uint blocki = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
   const uint ti = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;

   // Shared within all threads in one block
   __shared__ int isTargetBlock[MAX_BLOCKS_PER_DIM];
   __shared__ int isSourceBlock[MAX_BLOCKS_PER_DIM];
   const uint setIndex=blocki;
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

      /*need x,y coordinate of this column set of blocks, take it from first
        block in first column*/
      vmesh::LocalID setFirstBlockIndices0,setFirstBlockIndices1,setFirstBlockIndices2;
      vmesh->getIndices(GIDlist[gpu_columnData->columnBlockOffsets[gpu_columnData->setColumnOffsets[setIndex]]],
                        setFirstBlockIndices0, setFirstBlockIndices1, setFirstBlockIndices2);
      swapBlockIndices(setFirstBlockIndices0,setFirstBlockIndices1,setFirstBlockIndices2,dimension);
      /*compute the maximum starting point of the lagrangian (target) grid
        (base level) within the 4 corner cells in this
        block. Needed for computig maximum extent of target column*/

      Realf max_intersectionMin = intersection +
         (setFirstBlockIndices0 * WID + 0) * intersection_di +
         (setFirstBlockIndices1 * WID + 0) * intersection_dj;
      max_intersectionMin =  std::max(max_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + 0) * intersection_di +
                                      (setFirstBlockIndices1 * WID + WID - 1) * intersection_dj);
      max_intersectionMin =  std::max(max_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + WID - 1) * intersection_di +
                                      (setFirstBlockIndices1 * WID + 0) * intersection_dj);
      max_intersectionMin =  std::max(max_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + WID - 1) * intersection_di +
                                      (setFirstBlockIndices1 * WID + WID - 1) * intersection_dj);

      Realf min_intersectionMin = intersection +
         (setFirstBlockIndices0 * WID + 0) * intersection_di +
         (setFirstBlockIndices1 * WID + 0) * intersection_dj;
      min_intersectionMin =  std::min(min_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + 0) * intersection_di +
                                      (setFirstBlockIndices1 * WID + WID - 1) * intersection_dj);
      min_intersectionMin =  std::min(min_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + WID - 1) * intersection_di +
                                      (setFirstBlockIndices1 * WID + 0) * intersection_dj);
      min_intersectionMin =  std::min(min_intersectionMin,
                                      intersection +
                                      (setFirstBlockIndices0 * WID + WID - 1) * intersection_di +
                                      (setFirstBlockIndices1 * WID + WID - 1) * intersection_dj);

      //now, record which blocks are target blocks
      for (uint columnIndex = gpu_columnData->setColumnOffsets[setIndex];
           columnIndex < gpu_columnData->setColumnOffsets[setIndex] + gpu_columnData->setNumColumns[setIndex] ;
           ++columnIndex) {
         // Not parallelizing this at this level; not going to be many columns within a set

         // Abort all threads if vector capacity bailout
         if (bailout_flag[1] ) {
            return;
         }

         const vmesh::LocalID n_cblocks = gpu_columnData->columnNumBlocks[columnIndex];
         const vmesh::GlobalID* cblocks = GIDlist + gpu_columnData->columnBlockOffsets[columnIndex]; //column blocks
         vmesh::LocalID firstBlockIndices0,firstBlockIndices1,firstBlockIndices2;
         vmesh::LocalID lastBlockIndices0,lastBlockIndices1,lastBlockIndices2;
         vmesh->getIndices(cblocks[0],
                           firstBlockIndices0, firstBlockIndices1, firstBlockIndices2);
         vmesh->getIndices(cblocks[n_cblocks -1],
                           lastBlockIndices0, lastBlockIndices1, lastBlockIndices2);
         swapBlockIndices(firstBlockIndices0,firstBlockIndices1,firstBlockIndices2, dimension);
         swapBlockIndices(lastBlockIndices0,lastBlockIndices1,lastBlockIndices2, dimension);

         /* firstBlockV is in z the minimum velocity value of the lower
          *  edge in source grid.
          * lastBlockV is in z the maximum velocity value of the upper
          *  edge in source grid. */
         const Realf firstBlockMinV = (WID * firstBlockIndices2) * dv + v_min;
         const Realf lastBlockMaxV = (WID * (lastBlockIndices2 + 1)) * dv + v_min;

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
         for (uint blockK = firstBlockIndices2; blockK <= lastBlockIndices2; blockK +=warpSize){
            if ((blockK+ti) <= lastBlockIndices2) {
               //const int old  = atomicAdd(&isSourceBlock[blockK+ti],1);
               isSourceBlock[blockK+ti] = 1; // Does not need to be atomic, as long as it's no longer zero
            }
         }
         __syncthreads();

         //store target blocks
         for (uint blockK = (uint)firstBlockIndexK; blockK <= (uint)lastBlockIndexK; blockK+=warpSize){
            if ((blockK+ti) <= (uint)lastBlockIndexK) {
               isTargetBlock[blockK+ti] = 1; // Does not need to be atomic, as long as it's no longer zero
               //const int old  = atomicAdd(&isTargetBlock[blockK+ti],1);
            }
         }
         __syncthreads();

         if (ti==0) {
            // Set columns' transverse coordinates
            gpu_columns[columnIndex].i = setFirstBlockIndices0;
            gpu_columns[columnIndex].j = setFirstBlockIndices1;
            gpu_columns[columnIndex].kBegin = firstBlockIndices2;

            //store also for each column firstBlockIndexK, and lastBlockIndexK
            gpu_columns[columnIndex].minBlockK = firstBlockIndexK;
            gpu_columns[columnIndex].maxBlockK = lastBlockIndexK;
         }
      } // end loop over columns in set
      __syncthreads();

      for (uint blockT = 0; blockT < MAX_BLOCKS_PER_DIM; blockT +=warpSize) {
         const uint blockK = blockT + ti;
         // Not using warp accessors, as each thread has different block
         if (blockK < MAX_BLOCKS_PER_DIM) {
            if(isTargetBlock[blockK]!=0)  {
               const int targetBlock =
                  setFirstBlockIndices0 * gpu_block_indices_to_id[0] +
                  setFirstBlockIndices1 * gpu_block_indices_to_id[1] +
                  blockK                * gpu_block_indices_to_id[2];
               dev_map_require->set_element(targetBlock,vmesh->getLocalID(targetBlock));
               // if(!BlocksRequired->device_push_back(targetBlock)) {
               //    bailout_flag[1]=1;
               //    return;
               // }
            }
            if(isTargetBlock[blockK]!=0 && isSourceBlock[blockK]==0 )  {
               const int targetBlock =
                  setFirstBlockIndices0 * gpu_block_indices_to_id[0] +
                  setFirstBlockIndices1 * gpu_block_indices_to_id[1] +
                  blockK                * gpu_block_indices_to_id[2];
               if(!list_with_replace_new->device_push_back(targetBlock)) {
                  bailout_flag[1]=1; // out of capacity
               }

            }
            if(isTargetBlock[blockK]==0 && isSourceBlock[blockK]!=0 )  {
               const int targetBlock =
                  setFirstBlockIndices0 * gpu_block_indices_to_id[0] +
                  setFirstBlockIndices1 * gpu_block_indices_to_id[1] +
                  blockK                * gpu_block_indices_to_id[2];
               dev_map_remove->set_element(targetBlock,vmesh->getLocalID(targetBlock));
               // GPUTODO: could use device_insert to verify insertion, but not worth it
               // if(!list_to_replace->device_push_back(
               //       Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>(targetBlock,vmesh->getLocalID(targetBlock)))) {
               //    bailout_flag[2]=1; // out of capacity
               //    return;
               // }
            }
         } // block within MAX_BLOCKS_PER_DIM
      } // loop over all potential blocks
   } // if valid setIndex
}

__global__ void __launch_bounds__(VECL,4) acceleration_kernel(
   const vmesh::VelocityMesh* __restrict__ vmesh,
   vmesh::VelocityBlockContainer *blockContainer,
   const Vec* __restrict__ gpu_blockDataOrdered,
   const uint* __restrict__ gpu_cell_indices_to_id,
   const uint* __restrict__ gpu_block_indices_to_id,
   const Column* __restrict__ gpu_columns,
   const uint totalColumns, // not used
   const Realf intersection,
   const Realf intersection_di,
   const Realf intersection_dj,
   const Realf intersection_dk,
   const Realf v_min,
   const Realf i_dv,
   const Realf dv,
   const Realf minValue,
   const size_t invalidLID
) {
   //const uint gpuBlocks = gridDim.x * gridDim.y * gridDim.z;
   //const uint warpSize = blockDim.x * blockDim.y * blockDim.z;
   const uint blocki = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
   const uint w_tid = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;

   Realf *gpu_blockData = blockContainer->getData();
   const uint column = blocki;
   {
      /* New threading with each warp/wavefront working on one vector */
      const Realf v_r0 = ( (WID * gpu_columns[column].kBegin) * dv + v_min);

      // i,j,k are relative to the order in which we copied data to the values array.
      // After this point in the k,j,i loops there should be no branches based on dimensions
      // Note that the i dimension is vectorized, and thus there are no loops over i
      // Iterate through the perpendicular directions of the column
      for (uint j = 0; j < WID; j += VECL/WID) {
         // If VECL=WID2 (WID=4, VECL=16, or WID=8, VECL=64, then j==0)
         // This loop is still needed for e.g. Warp=VECL=32, WID2=64 (then j==0 or 4)
         const vmesh::LocalID nblocks = gpu_columns[column].nblocks;

         const uint i_indices = w_tid % WID;
         const uint j_indices = j + w_tid/WID;
         //int jk = j / (VECL/WID);

         const int target_cell_index_common =
            i_indices * gpu_cell_indices_to_id[0] +
            j_indices * gpu_cell_indices_to_id[1];
         const Realf intersection_min =
            intersection +
            (gpu_columns[column].i * WID + (Realf)i_indices) * intersection_di +
            (gpu_columns[column].j * WID + (Realf)j_indices) * intersection_dj;

         const Realf gk_intersection_min =
            intersection +
            (gpu_columns[column].i * WID + (Realf)( intersection_di > 0 ? 0 : WID-1 )) * intersection_di +
            (gpu_columns[column].j * WID + (Realf)( intersection_dj > 0 ? j : j+VECL/WID-1 )) * intersection_dj;
         const Realf gk_intersection_max =
            intersection +
            (gpu_columns[column].i * WID + (Realf)( intersection_di < 0 ? 0 : WID-1 )) * intersection_di +
            (gpu_columns[column].j * WID + (Realf)( intersection_dj < 0 ? j : j+VECL/WID-1 )) * intersection_dj;

         // loop through all perpendicular slices in column and compute the mapping as integrals.
         for (uint k=0; k < WID * nblocks; ++k) {
            // Compute reconstructions
            // Checked on 21.01.2022: Realf a[length] goes on the register despite being an array. Explicitly declaring it
            // as __shared__ had no impact on performance.
#ifdef ACC_SEMILAG_PLM
            Realf a[2];
            compute_plm_coeff(gpu_blockDataOrdered + gpu_columns[column].valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), (k + WID), a, minValue, w_tid);
#endif
#ifdef ACC_SEMILAG_PPM
            Realf a[3];
            compute_ppm_coeff(gpu_blockDataOrdered + gpu_columns[column].valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), h4, (k + WID), a, minValue, w_tid);
#endif
#ifdef ACC_SEMILAG_PQM
            Realf a[5];
            compute_pqm_coeff(gpu_blockDataOrdered + gpu_columns[column].valuesOffset + i_pcolumnv_gpu(j, 0, -1, nblocks), h8, (k + WID), a, minValue, w_tid);
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
            const int minGk = max(lagrangian_gk_l, int(gpu_columns[column].minBlockK * WID));
            const int maxGk = min(lagrangian_gk_r, int((gpu_columns[column].maxBlockK + 1) * WID - 1));
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
                  gpu_columns[column].i * gpu_block_indices_to_id[0] +
                  gpu_columns[column].j * gpu_block_indices_to_id[1] +
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
