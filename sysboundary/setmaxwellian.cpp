/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
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

/*!\file setmaxwellian.cpp
 * \brief Implementation of the class SysBoundaryCondition::Maxwellian to handle cells classified as sysboundarytype::MAXWELLIAN.
 */

#include <cstdlib>
#include <iostream>

#include "setmaxwellian.h"
#include "../vlasovsolver/vlasovmover.h"
#include "../object_wrapper.h"

#include "../projects/project.h" // for MaxwellianPhaseSpaceDensity

namespace SBC {
   Maxwellian::Maxwellian() : Inflow() {}
   Maxwellian::~Maxwellian() {}

   void Maxwellian::addParameters() {
      Readparameters::addComposing(
          "maxwellian.face", "List of faces on which set Maxwellian boundary conditions are to be applied ([xyz][+-]).");
      Readparameters::add("maxwellian.precedence",
                          "Precedence value of the set Maxwellian boundary condition (integer), the higher the stronger.",
                          3);
      Readparameters::add("maxwellian.reapplyUponRestart",
                          "If 0 (default), keep going with the state existing in the restart file. If 1, calls again "
                          "applyInitialState. Can be used to change boundary condition behaviour during a run.",
                          0);
      Readparameters::add("maxwellian.t_interval", "Time interval in seconds for applying the varying inflow condition.",
                          0.0); // 0 = re-calculate every time
      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;

         Readparameters::add(pop + "_maxwellian.file_x+",
                             "Input files for the set Maxwellian inflow parameters on face x+. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.file_x-",
                             "Input files for the set Maxwellian inflow parameters on face x-. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.file_y+",
                             "Input files for the set Maxwellian inflow parameters on face y+. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.file_y-",
                             "Input files for the set Maxwellian inflow parameters on face y-. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.file_z+",
                             "Input files for the set Maxwellian inflow parameters on face z+. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.file_z-",
                             "Input files for the set Maxwellian inflow parameters on face z-. Data format per line: time "
                             "(s) density (p/m^3) Temperature (K) Vx Vy Vz (m/s) Bx By Bz (T).",
                             "");
         Readparameters::add(pop + "_maxwellian.dynamic",
                             "Boolean value, is the set Maxwellian inflow dynamic in time or not.", 0);
      }
   }

   void Maxwellian::getParameters() {
      Readparameters::get("maxwellian.face", faceList);
      Readparameters::get("maxwellian.precedence", precedence);

      uint reapply;
      Readparameters::get("maxwellian.reapplyUponRestart", reapply);
      Readparameters::get("maxwellian.t_interval", tInterval);
      this->applyUponRestart = false;
      if(reapply == 1) {
         this->applyUponRestart = true;
      }

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;

         InflowSpeciesParameters sP;
         sP.nParams = 9;

         Readparameters::get(pop + "_maxwellian.dynamic", dynamic);
         Readparameters::get(pop + "_maxwellian.file_x+", sP.files[0]);
         Readparameters::get(pop + "_maxwellian.file_x-", sP.files[1]);
         Readparameters::get(pop + "_maxwellian.file_y+", sP.files[2]);
         Readparameters::get(pop + "_maxwellian.file_y-", sP.files[3]);
         Readparameters::get(pop + "_maxwellian.file_z+", sP.files[4]);
         Readparameters::get(pop + "_maxwellian.file_z-", sP.files[5]);

         speciesParams.push_back(sP);
      }
   }

   /*  Here the while loop iterates  from the centre of the maxwellian in blocksize (4*dvx) increments,
   *  and looks at the centre of the first velocity cell in the block (+0.5dvx), checking if the
   *  phase-space density there is large enough to be included due to sparsity threshold.
   *  That results in a "blocks radius"  vRadiusSquared from the centre of the Maxwellian distribution.
   *  Then we iterate through the actual blocks and calculate their radius R2 based on their velocity coordinates
   *  and the plasma bulk velocity. Blocks that fullfil R2<vRadiusSquared are included to blocksToInitialize.
   */
   vmesh::LocalID Maxwellian::findBlocksToInitialize(
      const uint popID,
      spatial_cell::SpatialCell& cell,
      creal& rho,
      creal& T,
      creal& VX0,
      creal& VY0,
      creal& VZ0) {

      bool search = true;
      uint counter = 0;
      vmesh::VelocityMesh *vmesh = cell.get_velocity_mesh(popID);
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;

      vmesh::GlobalID *GIDbuffer;
      #ifdef USE_GPU
      // Host-pinned memory buffer, max possible size
      const vmesh::LocalID* vblocks_ini = cell.get_velocity_grid_length(popID);
      const uint blocksCount = vblocks_ini[0]*vblocks_ini[1]*vblocks_ini[2];
      CHK_ERR( gpuMallocHost((void**)&GIDbuffer,blocksCount*sizeof(vmesh::GlobalID)) );
      #endif
      // Non-GPU: insert directly into vmesh

      Real V_crds[3];
      Real dV[3];
      dV[0] = cell.get_velocity_grid_block_size(popID)[0];
      dV[1] = cell.get_velocity_grid_block_size(popID)[1];
      dV[2] = cell.get_velocity_grid_block_size(popID)[2];
      creal minValue = cell.getVelocityBlockMinValue(popID);
      // Single cell, not block
      const Real dvx=cell.get_velocity_grid_cell_size(popID)[0];
      const Real dvy=cell.get_velocity_grid_cell_size(popID)[1];
      const Real dvz=cell.get_velocity_grid_cell_size(popID)[2];

      while (search) {
         if (0.1 * minValue > projects::MaxwellianPhaseSpaceDensity(mass, rho, T, counter*dV[0]+0.5*dvx, 0.5*dvy, 0.5*dvz) || counter > vblocks_ini[0]) {
            search = false;
         }
         counter++;
      }
      counter+=2;

      Real vRadiusSquared = (Real)counter * (Real)counter * dV[0] * dV[0];

      #ifndef USE_GPU
      // sphere volume is 4/3 pi r^3, approximate that 5*counterX*counterY*counterZ is enough.
      vmesh::LocalID currentMaxSize = LID + 5*counter*counter*counter;
      vmesh->setNewSize(currentMaxSize);
      GIDbuffer = vmesh->getGrid()->data();
      #endif

      vmesh::LocalID LID = 0;
      for (uint kv=0; kv<vblocks_ini[2]; ++kv) {
         for (uint jv=0; jv<vblocks_ini[1]; ++jv) {
            for (uint iv=0; iv<vblocks_ini[0]; ++iv) {
               const vmesh::GlobalID GID = vmesh->getGlobalID(iv,jv,kv);

               cell.get_velocity_block_coordinates(popID,GID,V_crds);
               V_crds[0] += 0.5*dV[0] - VX0;
               V_crds[1] += 0.5*dV[1] - VY0;
               V_crds[2] += 0.5*dV[2] - VZ0;
               Real R2 = ((V_crds[0])*(V_crds[0])
                          + (V_crds[1])*(V_crds[1])
                          + (V_crds[2])*(V_crds[2]));

               #ifndef USE_GPU
               if (LID >= currentMaxSize) {
                  currentMaxSize = LID + counter*counter*counter;
                  vmesh->setNewSize(currentMaxSize);
                  GIDbuffer = vmesh->getGrid()->data();
               }
               #endif
               if (R2 < vRadiusSquared) {
                  GIDbuffer[LID] = GID;
                  LID++;
               }
            }
         }
      }

      // Set final size of vmesh
      cell.get_population(popID).N_blocks = LID;

      #ifdef USE_GPU
      // Copy data into place
      cell.dev_resize_vmesh(popID,LID);
      vmesh::GlobalID *GIDtarget = vmesh->getGrid()->data();
      gpuStream_t stream = gpu_getStream();
      CHK_ERR( gpuMemcpyAsync(GIDtarget, GIDbuffer, LID*sizeof(vmesh::GlobalID), gpuMemcpyHostToDevice, stream));
      CHK_ERR( gpuStreamSynchronize(stream) );
      CHK_ERR( gpuFreeHost(GIDbuffer));
      #else
      vmesh->setNewSize(LID);
      #endif

      return LID;
   }

   /*!\brief Generate the template cell for the face corresponding to the index passed.
    * This function generates a spatial cell which is to be used as a template for the
    * system boundary condition.
    * \param templateCell Address of the template cell to be generated.
    * \param B Address of the magnetic field to be used as template.
    * \param inputDataIndex Index used for the location of the input data.
    * \param t Current simulation time.
    */
   void Maxwellian::generateTemplateCell(spatial_cell::SpatialCell& templateCell, Real (&B)[3], int inputDataIndex,
                                         creal t) {
      Real initRho, initT, initV0X, initV0Y, initV0Z, Bx = 0, By = 0, Bz = 0, buffer[8];

      templateCell.sysBoundaryFlag = this->getIndex();
      templateCell.sysBoundaryLayer = 1;
      #ifdef USE_GPU
      templateCell.prefetchDevice();
      #endif

      // Init all particle species
      for (uint popID=0; popID<getObjectWrapper().particleSpecies.size(); ++popID) {
         templateCell.clear(popID,false); //clear, do not de-allocate memory
         // Interpolate is in setbyuser.cpp and .h
         interpolate(inputDataIndex, popID, t, &buffer[0]);
         initRho = buffer[0];
         initT = buffer[1];
         initV0X = buffer[2];
         initV0X = buffer[3];
         initV0X = buffer[4];
         Bx = buffer[5];
         By = buffer[6];
         Bz = buffer[7];
         const Real mass = getObjectWrapper().particleSpecies[popID].mass;

         phiprof::Timer setVSpacetimer {"Set Velocity Space"};
         // Find list of blocks to initialize.
         const uint nRequested = this->findBlocksToInitialize(popID,templateCell, initRho, initT, initV0X, initV0Y, initV0Z);
         // stores in vmesh->getGrid() (localToGlobalMap)
         // with count in cell.get_population(popID).N_blocks

         // Resize and populate mesh
         templateCell.setNewSizeClear(popID, nRequested);
         templateCell.prepare_to_receive_blocks(popID);

         // Set the reservation value (capacity is increased in add_velocity_blocks
         const Realf minValue = templateCell.getVelocityBlockMinValue(popID);

         // fills v-space into target
         phiprof::Timer fillTimer {"fill phasespace"};

         #ifdef USE_GPU
         vmesh::VelocityMesh *vmesh = templateCell.dev_get_velocity_mesh(popID);
         vmesh::VelocityBlockContainer* VBC = templateCell.dev_get_velocity_blocks(popID);
         #else
         vmesh::VelocityMesh *vmesh = templateCell.get_velocity_mesh(popID);
         vmesh::VelocityBlockContainer* VBC = templateCell.get_velocity_blocks(popID);
         #endif
         // Loop over blocks
         Realf rhosum = 0;
         arch::parallel_reduce<arch::null>(
            {WID, WID, WID, nRequested},
            ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
               vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
               Realf* bufferData = VBC->getData();
               const vmesh::GlobalID blockGID = GIDlist[initIndex];
               // Calculate parameters for new block
               Real blockCoords[6];
               vmesh->getBlockInfo(blockGID,&blockCoords[0]);
               creal vxBlock = blockCoords[0];
               creal vyBlock = blockCoords[1];
               creal vzBlock = blockCoords[2];
               creal dvxCell = blockCoords[3];
               creal dvyCell = blockCoords[4];
               creal dvzCell = blockCoords[5];
               ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
                  creal vx = vxBlock + (i+0.5)*dvxCell - initV0X;
                  creal vy = vyBlock + (j+0.5)*dvyCell - initV0Y;
                  creal vz = vzBlock + (k+0.5)*dvzCell - initV0Z;
                  const Realf value = projects::MaxwellianPhaseSpaceDensity(mass,initRho,initT,vx,vy,vz);
                  bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
                  //lsum[0] += value;
               };
            }, rhosum);
         fillTimer.stop();

         // Set and apply the reservation value
         #ifdef USE_GPU
         phiprof::Timer reservationTimer {"set apply reservation"};
         templateCell.setReservation(popID,nRequested,true); // Force to this value
         templateCell.applyReservation(popID);
         reservationTimer.stop();
         #endif

         //let's get rid of blocks not fulfilling the criteria here to save memory.
         templateCell.adjustSingleCellVelocityBlocks(popID,true);

      } // for-loop over particle species

      B[0] = Bx;
      B[1] = By;
      B[2] = Bz;

      calculateCellMoments(&templateCell,true,false,true);

      templateCell.parameters[CellParams::RHOM_R] = templateCell.parameters[CellParams::RHOM];
      templateCell.parameters[CellParams::VX_R] = templateCell.parameters[CellParams::VX];
      templateCell.parameters[CellParams::VY_R] = templateCell.parameters[CellParams::VY];
      templateCell.parameters[CellParams::VZ_R] = templateCell.parameters[CellParams::VZ];
      templateCell.parameters[CellParams::RHOQ_R] = templateCell.parameters[CellParams::RHOQ];
      templateCell.parameters[CellParams::P_11_R] = templateCell.parameters[CellParams::P_11];
      templateCell.parameters[CellParams::P_22_R] = templateCell.parameters[CellParams::P_22];
      templateCell.parameters[CellParams::P_33_R] = templateCell.parameters[CellParams::P_33];
      templateCell.parameters[CellParams::RHOM_V] = templateCell.parameters[CellParams::RHOM];
      templateCell.parameters[CellParams::VX_V] = templateCell.parameters[CellParams::VX];
      templateCell.parameters[CellParams::VY_V] = templateCell.parameters[CellParams::VY];
      templateCell.parameters[CellParams::VZ_V] = templateCell.parameters[CellParams::VZ];
      templateCell.parameters[CellParams::RHOQ_V] = templateCell.parameters[CellParams::RHOQ];
      templateCell.parameters[CellParams::P_11_V] = templateCell.parameters[CellParams::P_11];
      templateCell.parameters[CellParams::P_22_V] = templateCell.parameters[CellParams::P_22];
      templateCell.parameters[CellParams::P_33_V] = templateCell.parameters[CellParams::P_33];
   }

   std::string Maxwellian::getName() const { return "Maxwellian"; }
   uint Maxwellian::getIndex() const { return sysboundarytype::MAXWELLIAN; }

} // namespace SBC
