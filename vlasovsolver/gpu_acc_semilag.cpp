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

#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>
#include <phiprof.hpp>
#include "../definitions.h"

#include "gpu_acc_semilag.hpp"
#include "cpu_acc_intersections.hpp"
#include "gpu_acc_map.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

/*!
  Propagates the distribution function in velocity space of given real
  space cell using a semi-Lagrangian acceleration approach..

  Based on SLICE-3D algorithm: Zerroukat, M., and T. Allen. "A
  three‐dimensional monotone and conservative semi‐Lagrangian scheme
  (SLICE‐3D) for transport problems." Quarterly Journal of the Royal
  Meteorological Society 138.667 (2012): 1640-1651.

 * @param mpiGrid DCCRG container of spatial cells
 * @param acceleratedCells vector of cells for which to perform acceleration
 * @param popID ID of the accelerated particle species.
 * @param map_order Order in which vx,vy,vz mappings are performed.
*/

void gpu_accelerate_cells(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                          const std::vector<CellID>& acceleratedCells,
                          const uint popID,
                          const uint map_order
   ) {

   uint gpuMaxBlockCount = 0;
   // Calculate intersections (should be constant cost per cell)
   int intersections_id {phiprof::initializeTimer("cell-compute-intersections")};
   #pragma omp parallel
   {
      uint threadGpuMaxBlockCount = 0;
      #pragma omp for schedule(static,1)
      for (size_t c=0; c<acceleratedCells.size(); ++c) {
         const CellID cellID = acceleratedCells[c];
         SpatialCell* SC = mpiGrid[cellID];
         Population& pop = SC->get_population(popID);
         compute_cell_intersections(SC, popID, map_order, pop.subcycleDt, intersections_id);

         const vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         const uint blockCount = vmesh->size();
         threadGpuMaxBlockCount = std::max(threadGpuMaxBlockCount,blockCount);
      }
      #pragma omp critical
      {
         gpuMaxBlockCount = std::max(gpuMaxBlockCount,threadGpuMaxBlockCount);
      }
   }

   // Ensure accelerator has enough temporary memory allocated
   phiprof::Timer verificationTimer {"gpu ACC allocation verifications"};
   const uint nCellsAlloc = std::max((uint)acceleratedCells.size(),gpu_getMaxThreads());
   gpu_vlasov_allocate(gpuMaxBlockCount);
   gpu_acc_allocate(gpuMaxBlockCount);
   gpu_batch_allocate(nCellsAlloc,0);
   #pragma omp parallel for schedule(static,1)
   for (size_t c=0; c<acceleratedCells.size(); ++c) {
      const CellID cellID = acceleratedCells[c];
      SpatialCell* SC = mpiGrid[cellID];
      SC->setReservation(popID, gpuMaxBlockCount);
      SC->applyReservation(popID);
   }
   verificationTimer.stop();

   // Do some overall preparation regarding dimensions and acceleration order
   const uint D0 = (*vmesh::getMeshWrapper()->velocityMeshes)[popID].gridLength[0];
   const uint D1 = (*vmesh::getMeshWrapper()->velocityMeshes)[popID].gridLength[1];
   const uint D2 = (*vmesh::getMeshWrapper()->velocityMeshes)[popID].gridLength[2];

   std::vector<int> dimOrder(3);
   switch(map_order) {
      case 0: { //Map order XYZ
         dimOrder={0,1,2};
         break;
      }
      case 1: { //Map order YZX
         dimOrder={1,2,0};
         break;
      }
      case 2: { //Map order ZXY
         dimOrder={2,0,1};
         break;
      }
      default:
         std::cerr<<"ERROR! Incorrect map_order "<<map_order<<"!"<<std::endl;
         abort();
   }

   /**
      Loop over three velocity dimensions, based on map_order,
      and accelerate all cells for that dimension.
   */
   for (int dimIndex = 0; dimIndex<3; ++dimIndex) {
      // Determine intersections for each mapping order
      int dimension = dimOrder[dimIndex];

      /*< used when computing id of target block, 0 to quite compiler warnings */
      uint block_indices_to_id[3] = {0, 0, 0};
      uint block_indices_to_probe[3] = {0, 0, 0};
      uint cell_indices_to_id[3] = {0, 0, 0};

      // Find probe cube extents as well
      int Dacc, Dother;

      switch (dimension) {
         case 0: /* i and k coordinates have been swapped */
            /* set values in array that is used to convert block indices to id using a dot product */
            block_indices_to_id[0] = D0*D1;
            block_indices_to_id[1] = D0;
            block_indices_to_id[2] = 1;

            /* set values in array that is used to convert block indices to position in probe cube
               propagate along X, flatten Y+Z */
            block_indices_to_probe[0] = D1*D2;
            block_indices_to_probe[1] = D2;
            block_indices_to_probe[2] = 1;
            Dacc = D0;
            Dother = D1*D2;

            /* set values in array that is used to convert block indices to id using a dot product */
            cell_indices_to_id[0]=WID2;
            cell_indices_to_id[1]=WID;
            cell_indices_to_id[2]=1;
            break;
         case 1: /* j and k coordinates have been swapped */
            /* set values in array that is used to convert block indices to id using a dot product */
            block_indices_to_id[0]=1;
            block_indices_to_id[1] = D0*D1;
            block_indices_to_id[2] = D0;

            /* set values in array that is used to convert block indices to position in probe cube
               propagate along Y, flatten X+Z */
            block_indices_to_probe[0] = D2;
            block_indices_to_probe[1] = D0*D2;
            block_indices_to_probe[2] = 1;
            Dacc = D1;
            Dother = D0*D2;

            /* set values in array that is used to convert block indices to id using a dot product */
            cell_indices_to_id[0]=1;
            cell_indices_to_id[1]=WID2;
            cell_indices_to_id[2]=WID;
            break;
         case 2:
            /* set values in array that is used to convert block indices to id using a dot product */
            block_indices_to_id[0]=1;
            block_indices_to_id[1] = D0;
            block_indices_to_id[2] = D0*D1;

            /* set values in array that is used to convert block indices to position in probe cube
               propagate along Z, flatten X+Y */
            block_indices_to_probe[0] = D1;
            block_indices_to_probe[1] = 1;
            block_indices_to_probe[2] = D0*D1;
            Dacc = D2;
            Dother = D0*D1;

            /* set values in array that is used to convert block indices to id using a dot product. */
            cell_indices_to_id[0]=1;
            cell_indices_to_id[1]=WID;
            cell_indices_to_id[2]=WID2;
            break;
         default:
            std::cerr<<"Invalid dimension "<<dimension<<"!"<<std::endl;
            abort();
      }

      // Copy indexing information to device (better to pass in a single struct? as an argument?)
      CHK_ERR( gpuMemcpy(gpu_cell_indices_to_id, cell_indices_to_id, 3*sizeof(uint), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(gpu_block_indices_to_id, block_indices_to_id, 3*sizeof(uint), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(gpu_block_indices_to_probe, block_indices_to_probe, 3*sizeof(uint), gpuMemcpyHostToDevice) );

      // Call acceleration solver
      int timerId {phiprof::initializeTimer("cell-semilag-acc")};
      // Dynamic cost due to varying block counts. (now threaded with OpenMP)
#pragma omp parallel for schedule(dynamic,1)
      for (size_t c=0; c<acceleratedCells.size(); ++c) {

         const CellID cellID = acceleratedCells[c];
         SpatialCell* SC = mpiGrid[cellID];
         Population& pop = SC->get_population(popID);

         Realf intersections[4];
         // Place intersections into array so that propagation direction is "z"-coordinate
         switch (dimension) {
            case 0:
               // X: swap intersection i and k coordinates
               intersections[0]=(Realf)pop.intersection_x;
               intersections[1]=(Realf)pop.intersection_x_dk;
               intersections[2]=(Realf)pop.intersection_x_dj;
               intersections[3]=(Realf)pop.intersection_x_di;
               break;
            case 1:
               // Y: swap intersection j and k coordinates
               intersections[0]=(Realf)pop.intersection_y;
               intersections[1]=(Realf)pop.intersection_y_di;
               intersections[2]=(Realf)pop.intersection_y_dk;
               intersections[3]=(Realf)pop.intersection_y_dj;
               break;
            case 2:
               // Z: k remains propagation coordinate, no swaps
               intersections[0]=(Realf)pop.intersection_z;
               intersections[1]=(Realf)pop.intersection_z_di;
               intersections[2]=(Realf)pop.intersection_z_dj;
               intersections[3]=(Realf)pop.intersection_z_dk;
               break;
            default:
               std::cerr<<"Invalid dimension "<<dimension<<"!"<<std::endl;
               abort();
         }
         // Launch acceleration solver
         phiprof::Timer semilagAccTimer {timerId};
         gpu_acc_map_1d(mpiGrid,
                        SC,
                        popID,
                        intersections[0],
                        intersections[1],
                        intersections[2],
                        intersections[3],
                        dimension,
                        Dacc,
                        Dother
            );
         semilagAccTimer.stop();
      }
   }
}
