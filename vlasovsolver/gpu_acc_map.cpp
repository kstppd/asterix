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

#include "gpu_acc_kernels.hpp"

/*
   Here we map from the current time step grid, to a target grid which
   is the lagrangian departure grid (so th grid at timestep +dt,
   tracked backwards by -dt)
*/
__host__ bool gpu_acc_map_1d(spatial_cell::SpatialCell* spatial_cell,
                              const uint popID,
                              Real in_intersection,
                              Real in_intersection_di,
                              Real in_intersection_dj,
                              Real in_intersection_dk,
                              const uint dimension
   ) {
   // Ensure previous actions have completed?
   //CHK_ERR( gpuStreamSynchronize(stream) );
   gpuStream_t stream = gpu_getStream();

   // Conversion here:
   Realf intersection = (Realf)in_intersection;
   Realf intersection_di = (Realf)in_intersection_di;
   Realf intersection_dj = (Realf)in_intersection_dj;
   Realf intersection_dk = (Realf)in_intersection_dk;

   //spatial_cell->dev_upload_population(popID); // Should not be necessary.
   vmesh::VelocityMesh* vmesh    = spatial_cell->get_velocity_mesh(popID);
   vmesh::VelocityBlockContainer* blockContainer = spatial_cell->get_velocity_blocks(popID);
   vmesh::VelocityMesh* dev_vmesh    = spatial_cell->dev_get_velocity_mesh(popID);
   vmesh::VelocityBlockContainer* dev_blockContainer = spatial_cell->dev_get_velocity_blocks(popID);

   //nothing to do if no blocks
   vmesh::LocalID nBlocksBeforeAdjust = vmesh->size();
   if (nBlocksBeforeAdjust == 0) {
      return true;
   }

   auto minValue = spatial_cell->getVelocityBlockMinValue(popID);
   // These query velocity mesh parameters which are duplicated for both host and device
   const vmesh::LocalID D0 = vmesh->getGridLength()[0];
   const vmesh::LocalID D1 = vmesh->getGridLength()[1];
   const vmesh::LocalID D2 = vmesh->getGridLength()[2];
   const Realf dv    = vmesh->getCellSize()[dimension];
   const Realf v_min = vmesh->getMeshMinLimits()[dimension];
   const int max_v_length  = (int)vmesh->getGridLength()[dimension];
   const Realf i_dv = 1.0/dv;

   // Thread id used for persistent device memory pointers
   const uint cpuThreadID = gpu_getThread();

   // Some kernels in here require the number of threads to be equal to VECL.
   // Future improvements would be to allow setting it directly to WID3.
   // Other kernels (not handling block data) can use GPUTHREADS which
   // is equal to NVIDIA: 32 or AMD: 64.

   /*< used when computing id of target block, 0 for compiler */
   uint block_indices_to_id[3] = {0, 0, 0};
   uint cell_indices_to_id[3] = {0, 0, 0};
   // 13.11.2023: for some reason these hostRegister calls say the memory is already registered.
   // CHK_ERR(gpuHostRegister(block_indices_to_id, 3*sizeof(uint),gpuHostRegisterPortable));
   // CHK_ERR(gpuHostRegister(cell_indices_to_id, 3*sizeof(uint),gpuHostRegisterPortable));

   Realf is_temp;
   switch (dimension) {
      case 0: /* i and k coordinates have been swapped*/
         /*swap intersection i and k coordinates*/
         is_temp=intersection_di;
         intersection_di=intersection_dk;
         intersection_dk=is_temp;

         /*set values in array that is used to convert block indices to id using a dot product*/
         block_indices_to_id[0] = D0*D1;
         block_indices_to_id[1] = D0;
         block_indices_to_id[2] = 1;

         /*set values in array that is used to convert block indices to id using a dot product*/
         cell_indices_to_id[0]=WID2;
         cell_indices_to_id[1]=WID;
         cell_indices_to_id[2]=1;
         break;
      case 1: /* j and k coordinates have been swapped*/
         /*swap intersection j and k coordinates*/
         is_temp=intersection_dj;
         intersection_dj=intersection_dk;
         intersection_dk=is_temp;

         /*set values in array that is used to convert block indices to id using a dot product*/
         block_indices_to_id[0]=1;
         block_indices_to_id[1] = D0*D1;
         block_indices_to_id[2] = D0;

         /*set values in array that is used to convert block indices to id using a dot product*/
         cell_indices_to_id[0]=1;
         cell_indices_to_id[1]=WID2;
         cell_indices_to_id[2]=WID;
         break;
      case 2:
         /*set values in array that is used to convert block indices to id using a dot product*/
         block_indices_to_id[0]=1;
         block_indices_to_id[1] = D0;
         block_indices_to_id[2] = D0*D1;

         // set values in array that is used to convert block indices to id using a dot product.
         cell_indices_to_id[0]=1;
         cell_indices_to_id[1]=WID;
         cell_indices_to_id[2]=WID2;
         break;
   }
   // Ensure allocations
   spatial_cell->setReservation(popID, nBlocksBeforeAdjust);
   spatial_cell->applyReservation(popID);
   gpu_vlasov_allocate_perthread(cpuThreadID, nBlocksBeforeAdjust);

   // Copy indexing information to device (async)
   CHK_ERR( gpuMemcpyAsync(gpu_cell_indices_to_id[cpuThreadID], cell_indices_to_id, 3*sizeof(uint), gpuMemcpyHostToDevice, stream) );
   CHK_ERR( gpuMemcpyAsync(gpu_block_indices_to_id[cpuThreadID], block_indices_to_id, 3*sizeof(uint), gpuMemcpyHostToDevice, stream) );

   // Re-use maps from cell itself
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map_require = spatial_cell->velocity_block_with_content_map;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map_remove = spatial_cell->velocity_block_with_no_content_map;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_require = spatial_cell->dev_velocity_block_with_content_map;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *dev_map_remove = spatial_cell->dev_velocity_block_with_no_content_map;

   // pointers to device memory buffers
   vmesh::GlobalID *GIDlist = gpu_GIDlist[cpuThreadID];
   vmesh::LocalID *LIDlist = gpu_LIDlist[cpuThreadID];
   // vmesh::GlobalID *BlocksID_mapped = gpu_BlocksID_mapped[cpuThreadID];
   // vmesh::GlobalID *BlocksID_mapped_sorted = gpu_BlocksID_mapped[cpuThreadID];
   // vmesh::LocalID *LIDlist_unsorted = gpu_LIDlist_unsorted[cpuThreadID];
   // vmesh::LocalID *columnNBlocks = gpu_columnNBlocks[cpuThreadID];

   // Columndata is device construct but contains splitvectors
   ColumnOffsets *columnData = gpu_columnOffsetData[cpuThreadID];
   //columnData->prefetchDevice(stream);

   // These splitvectors are in host memory (only used for a clear call)
   split::SplitVector<vmesh::GlobalID> *list_with_replace_new = spatial_cell->list_with_replace_new;
   // These splitvectors are in device memory
   split::SplitVector<vmesh::GlobalID> *dev_list_with_replace_new = spatial_cell->dev_list_with_replace_new;
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* dev_list_delete = spatial_cell->dev_list_delete;
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* dev_list_to_replace = spatial_cell->dev_list_to_replace;
   split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>* dev_list_with_replace_old = spatial_cell->dev_list_with_replace_old;

   // Call function for sorting block list and building columns from it.
   // Can probably be further optimized.
   //gpuStream_t priorityStream = gpu_getPriorityStream();
   //CHK_ERR( gpuMemsetAsync(columnNBlocks, 0, gpu_acc_columnContainerSize*sizeof(vmesh::LocalID), stream) );
   //CHK_ERR( gpuStreamSynchronize(stream) ); // Yes needed because we use priority stream for block list sorting
   /*
   sortBlocklistByDimension(dev_vmesh,
                            nBlocksBeforeAdjust,
                            dimension,
                            BlocksID_mapped,
                            BlocksID_mapped_sorted,
                            GIDlist,
                            LIDlist_unsorted,
                            LIDlist,
                            columnNBlocks,
                            columnData,
                            cpuThreadID,
                            stream
      );
   CHK_ERR( gpuStreamSynchronize(stream) ); // Yes needed to get column data back to regular stream

   // Calculate total sum of columns and total values size
   CHK_ERR( gpuMemsetAsync(returnLID[cpuThreadID], 0, 2*sizeof(vmesh::LocalID), stream) );
   // this needs to be serial, but is fast.
   count_columns_kernel<<<1, 1, 0, stream>>> (
      columnData,
      returnLID[cpuThreadID], //gpu_totalColumns,gpu_valuesSizeRequired
      // Pass vectors for clearing
      dev_list_with_replace_new,
      dev_list_delete,
      dev_list_to_replace,
      dev_list_with_replace_old
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuMemcpyAsync(host_returnLID[cpuThreadID], returnLID[cpuThreadID], 2*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, stream) );
   CHK_ERR( gpuStreamSynchronize(stream) );
   const vmesh::LocalID host_totalColumns = host_returnLID[cpuThreadID][0];
   const vmesh::LocalID host_valuesSizeRequired = host_returnLID[cpuThreadID][1];
   // Update tracker of maximum encountered column count
   if (gpu_acc_foundColumnsCount < host_totalColumns) {
      gpu_acc_foundColumnsCount = host_totalColumns;
   }

   // Create array of column objects
   Column host_columns[host_totalColumns];
   // and copy it into device memory
   Column *columns = gpu_columns[cpuThreadID];
   CHK_ERR( gpuMemcpyAsync(columns, &host_columns, host_totalColumns*sizeof(Column), gpuMemcpyHostToDevice, stream) );
   //SSYNC;

   // this needs to be serial, but is fast.
   offsets_into_columns_kernel<<<1, 1, 0, stream>>> (
      columnData,
      columns,
      host_valuesSizeRequired
      );
   CHK_ERR( gpuPeekAtLastError() );
   //CHK_ERR( gpuStreamSynchronize(stream) );
   */

   // New merged kernel approach without sorting for columns

   /**
      First, we generate a "probe cube". It started off as an actual
      cube, but was then flattened so that there's only two dimensions.
      One dimension is that of the current acceleration, and the other
      dimension is the sum of the other two maximal velocity block
      domain extents.
   */

   // Find probe cube extents
   int Dacc, Dother;
   switch (dimension) {
      case 0:
         // propagate along x
         Dacc = D0;
         Dother = D1*D2;
         break;
      case 1:
         // propagate along y
         Dacc = D1;
         Dother = D0*D2;
         break;
      case 2:
         // propagate along z
         Dacc = D2;
         Dother = D0*D1;
         break;
      default:
         std::cerr<<" incorrect dimension!"<<std::endl;
   }

   empty_vectors_kernel<<<1, 1, 0, stream>>> (
      dev_list_with_replace_new,
      dev_list_delete,
      dev_list_to_replace,
      dev_list_with_replace_old
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Allocate probeCube (buffer of LIDs) and reduction target
   vmesh::LocalID *probeCube, *probeFlattened;
   const int nFlatteneds = 5;
   // The flattened version must store:
   // 1) how many columns per potential column position (potColumn)
   // 2) how many blocks per potColumn
   // 3) cumulative offset into columns per potColumn
   // 4) cumulative offset into columnSets per potColumn
   // 5) cumulative offset into blocks per potColumn
   // For reductions, each slice of the flattened array should have a size a multiple of 2*MAX_BLOCKSIZE:
   const size_t flatExtent = 2*Hashinator::defaults::MAX_BLOCKSIZE * (1 + ((Dother - 1) / (2*Hashinator::defaults::MAX_BLOCKSIZE)));
   CHK_ERR( gpuMalloc((void**)&probeCube, Dacc*Dother*sizeof(vmesh::LocalID)) );
   CHK_ERR( gpuMalloc((void**)&probeFlattened, flatExtent*nFlatteneds*sizeof(vmesh::LocalID)) );

   // Fill probe cube vmesh invalid LID values, flattened array with zeros
   const size_t grid_fill_invalid = 1 + ((Dacc*Dother - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   fill_probe_invalid<<<grid_fill_invalid,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(probeCube,Dacc*Dother,vmesh->invalidLocalID());
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuMemsetAsync(probeFlattened, 0, flatExtent*nFlatteneds*sizeof(vmesh::LocalID)) );

   // Read in GID list from vmesh, store LID values into probe cube in correct order
   // Launch params, fast ceil for positive ints
   const size_t grid_fill_ord = 1 + ((nBlocksBeforeAdjust - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   fill_probe_ordered<<<grid_fill_ord,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      dev_vmesh,
      probeCube,
      D0,D1,D2,
      nBlocksBeforeAdjust,
      dimension
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(stream) );

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
   CHK_ERR( gpuStreamSynchronize(stream) );

   // Next we reduce counts and offsets.
   // TODO: Make two-phase kernels for more parallel reductions? Perhaps not needed.
   // Especially as batch operation will launch for many cells at once.

   // First kernel just counts total number of columns and columnsets, and resizes
   // the contents of columnData.
   // To keep things simple for now, it launches with just 1 block and max threads.
   reduce_probe_A<<<1,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      probeFlattened,
      Dother,
      flatExtent,
      nBlocksBeforeAdjust,
      returnLID[cpuThreadID],
      columnData
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(stream) );

   // Next kernel performs an exclusive prefix scan to get offsets for storing
   // data from potential columns into the columnData container.

   // A proper prefix scan needs to be a two-phase process, thus two kernels,
   // but here we do an iterative loop processing MAX_BLOCKSIZE elements at once.
   // Not as efficient but simpler, and will be parallelized over spatial cells.
   scan_probe_A<<<1,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      probeFlattened,
      Dacc,
      Dother,
      flatExtent
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(stream) );

   // Now we have gathered all the required offsets into probeFlattened, and can
   // now launch a kernel which constructs the columns offsets in parallel.
   build_column_offsets<<<grid_cube,Hashinator::defaults::MAX_BLOCKSIZE,0,stream>>>(
      dev_vmesh,
      probeCube,
      probeFlattened,
      Dacc,
      Dother,
      flatExtent,
      vmesh->invalidLocalID(),
      columnData,
      GIDlist,
      LIDlist
      );
   CHK_ERR( gpuPeekAtLastError() );

   // Copy back to host sizes of found columns etc
   CHK_ERR( gpuMemcpyAsync(host_returnLID[cpuThreadID], returnLID[cpuThreadID], 2*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, stream) );
   CHK_ERR( gpuStreamSynchronize(stream) );
   CHK_ERR( gpuFreeAsync(probeCube,stream) );
   CHK_ERR( gpuFreeAsync(probeFlattened,stream) );

   // Read count of columns and columnsets, calculate required size of buffers
   const vmesh::LocalID host_totalColumns = host_returnLID[cpuThreadID][0];
   //const vmesh::LocalID host_totalColumnSets = host_returnLID[cpuThreadID][1];
   const vmesh::LocalID host_valuesSizeRequired = (nBlocksBeforeAdjust + 2*host_totalColumns) * WID3 / VECL;
   // Update tracker of maximum encountered column count
   if (gpu_acc_foundColumnsCount < host_totalColumns) {
      gpu_acc_foundColumnsCount = host_totalColumns;
   }

   // Create array of column objects
   Column host_columns[host_totalColumns];
   // and copy it into device memory
   Column *columns = gpu_columns[cpuThreadID];
   CHK_ERR( gpuMemcpyAsync(columns, &host_columns, host_totalColumns*sizeof(Column), gpuMemcpyHostToDevice, stream) );









   // Launch kernels for transposing and ordering velocity space data into columns
   reorder_blocks_by_dimension_kernel<<<host_totalColumns, VECL, 0, stream>>> (
      dev_blockContainer,
      gpu_blockDataOrdered[cpuThreadID],
      gpu_cell_indices_to_id[cpuThreadID],
      LIDlist,
      columnData,
      host_valuesSizeRequired
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
      evaluate_column_extents_kernel<<<host_totalColumns, GPUTHREADS, 0, stream>>> (
         dimension,
         dev_vmesh,
         columnData,
         columns,
         dev_list_with_replace_new,
         dev_map_require,
         dev_map_remove,
         GIDlist,
         gpu_block_indices_to_id[cpuThreadID],
         intersection,
         intersection_di,
         intersection_dj,
         intersection_dk,
         Parameters::bailout_velocity_space_wall_margin,
         max_v_length,
         v_min,
         dv,
         returnLID[cpuThreadID] //gpu_bailout_flag:
                                // - element[0]: touching velspace wall
                                // - element[1]: splitvector list_with_replace_new capacity error
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
         // If so, recapacitate and try again.
         // We'll take at least our current velspace size (plus safety factor), or, if that wasn't enough,
         // twice what we had before.
         size_t newCapacity = (size_t)(spatial_cell->getReservation(popID)*BLOCK_ALLOCATION_FACTOR);
         //printf("column data recapacitate! %lu newCapacity\n",(long unsigned)newCapacity);
         list_with_replace_new->clear();
         spatial_cell->setReservation(popID, newCapacity);
         spatial_cell->applyReservation(popID);
      }
      // Loop until we return without an out-of-capacity error
   } while (host_returnLID[cpuThreadID][1] != 0);

   /** Rules used in extracting keys or elements from hashmaps
       Now these include passing pointers to GPU memory in order to evaluate
       nBlocksAfterAdjust without going via host. Pointers are copied by value.
   */
   const vmesh::GlobalID emptybucket = map_require->get_emptybucket();
   const vmesh::GlobalID tombstone   = map_require->get_tombstone();

   auto rule_delete_move = [emptybucket, tombstone, dev_map_remove, dev_list_with_replace_new, dev_vmesh]
      __host__ __device__(const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval) -> bool {
                              const vmesh::LocalID nBlocksAfterAdjust1 = dev_vmesh->size()
                                 + dev_list_with_replace_new->size() - dev_map_remove->size();
                              return kval.first != emptybucket &&
                                 kval.first != tombstone &&
                                 kval.second >= nBlocksAfterAdjust1 &&
                                 kval.second != vmesh::INVALID_LOCALID; };
   auto rule_to_replace = [emptybucket, tombstone, dev_map_remove, dev_list_with_replace_new, dev_vmesh]
      __host__ __device__(const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval) -> bool {
                             const vmesh::LocalID nBlocksAfterAdjust2 = dev_vmesh->size()
                                + dev_list_with_replace_new->size() - dev_map_remove->size();
                             return kval.first != emptybucket &&
                                kval.first != tombstone &&
                                kval.second < nBlocksAfterAdjust2; };

   // Additions are gathered directly into list instead of a map/set
   map_require->extractPatternLoop(*dev_list_with_replace_old, rule_delete_move, stream);
   map_remove->extractPatternLoop(*dev_list_delete, rule_delete_move, stream);
   map_remove->extractPatternLoop(*dev_list_to_replace, rule_to_replace, stream);
   //CHK_ERR( gpuStreamSynchronize(stream) );

   // Note: in this call, unless hitting v-space walls, we only grow the vspace size
   // and thus do not delete blocks or replace with old blocks.
   vmesh::LocalID nBlocksAfterAdjust = spatial_cell->adjust_velocity_blocks_caller(popID);
   // Velocity space has now all extra blocks added and/or removed for the transform target
   // and will not change shape anymore.
   spatial_cell->largestvmesh = spatial_cell->largestvmesh > nBlocksAfterAdjust ? spatial_cell->largestvmesh : nBlocksAfterAdjust;

   // Zero out target data on device (unified) (note, pointer needs to be re-fetched
   // here in case VBC size was increased)
   //GPUTODO: direct access to blockContainer getData causes page fault
   Realf *blockData = blockContainer->getData();
   CHK_ERR( gpuMemsetAsync(blockData, 0, nBlocksAfterAdjust*WID3*sizeof(Realf), stream) );
   //CHK_ERR( gpuStreamSynchronize(stream) );

   // GPUTODO: Adapt to work as VECL=WID3 instead of VECL=WID2
   acceleration_kernel<<<host_totalColumns, VECL, 0, stream>>> (
      dev_vmesh,
      dev_blockContainer,
      gpu_blockDataOrdered[cpuThreadID],
      gpu_cell_indices_to_id[cpuThreadID],
      gpu_block_indices_to_id[cpuThreadID],
      columns,
      host_totalColumns,
      intersection,
      intersection_di,
      intersection_dj,
      intersection_dk,
      v_min,
      i_dv,
      dv,
      minValue,
      vmesh->invalidLocalID()
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(stream) );

   return true;
}
