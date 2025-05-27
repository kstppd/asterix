/*
 * This file is part of Vlasiator.
 * Copyright 2010-2020 University of Helsinki
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


// This is the GPU version of cpu_pitch_angle_diffusion

#include "../parameters.h"
#include "../object_wrapper.h"
#include <math.h>
//#include <cmath> // NaN Inf checks
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <Eigen/Geometry>
#include "vec.h"
#include "gpu_pitch_angle_diffusion.hpp"
#include "common_pitch_angle_diffusion.hpp"

#define MUSPACE(var,v_ind,mu_ind) var.at((mu_ind)*nbins_v + (v_ind))

using namespace spatial_cell;
using namespace Eigen;

template <typename Lambda> inline static void loop_over_block(Lambda loop_body) {

   for (int k = 0; k < WID; ++k) {
      for (int j = 0; j < WID; j +=VECL/WID) { // Iterate through coordinates (z,y)

         // create vectors with the i and j indices in the vector position on the plane.
#if VECL == 4 && WID == 4
         const Veci i_indices = Veci({0, 1, 2, 3});
         const Veci j_indices = Veci({j, j, j, j});
#elif VECL == 4 && WID == 8
#error "__FILE__ : __LINE__ : VECL == 4 && WID == 8 cannot work!"
#elif VECL == 8 && WID == 4
         const Veci i_indices = Veci({0, 1, 2, 3,
               0, 1, 2, 3});
         const Veci j_indices = Veci({j, j, j, j,
               j + 1, j + 1, j + 1, j + 1});
#elif VECL == 8 && WID == 8
         const Veci i_indices = Veci({0, 1, 2, 3, 4, 5, 6, 7});
         const Veci j_indices = Veci({j, j, j, j, j, j, j, j});
#elif VECL == 16 && WID == 4
         const Veci i_indices = Veci({0, 1, 2, 3,
               0, 1, 2, 3,
               0, 1, 2, 3,
               0, 1, 2, 3});
         const Veci j_indices = Veci({j, j, j, j,
               j + 1, j + 1, j + 1, j + 1,
               j + 2, j + 2, j + 2, j + 2,
               j + 3, j + 3, j + 3, j + 3});
#elif VECL == 16 && WID == 8
         const Veci i_indices = Veci({0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7});
         const Veci j_indices = Veci({j,   j,   j,   j,   j,   j,   j,   j,
               j+1, j+1, j+1, j+1, j+1, j+1, j+1, j+1});
#elif VECL == 16 && WID == 16
         const Veci i_indices = Veci({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
         const Veci j_indices = Veci({j, j, j, j, j, j, j, j, j, j,  j,  j,  j,  j,  j, j});
#elif VECL == 32 && WID == 4
#error "__FILE__ : __LINE__ : VECL == 32 && WID == 4 cannot work, too long vector for one plane!"
#elif VECL == 32 && WID == 8
         const Veci i_indices = Veci({0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7});
         const Veci j_indices = Veci({j,   j,   j,   j,   j,   j,   j,   j,
               j+1, j+1, j+1, j+1, j+1, j+1, j+1, j+1,
               j+2, j+2, j+2, j+2, j+2, j+2, j+2, j+2,
               j+3, j+3, j+3, j+3, j+3, j+3, j+3, j+3});
#elif VECL == 64 && WID == 4
#error "__FILE__ : __LINE__ : VECL == 64 && WID == 4 cannot work, too long vector for one plane!"
#elif VECL == 64 && WID == 8
         const Veci i_indices = Veci({0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7,
               0, 1, 2, 3, 4, 5, 6, 7});
         const Veci j_indices = Veci({j,   j,   j,   j,   j,   j,   j,   j,
               j+1, j+1, j+1, j+1, j+1, j+1, j+1, j+1,
               j+2, j+2, j+2, j+2, j+2, j+2, j+2, j+2,
               j+3, j+3, j+3, j+3, j+3, j+3, j+3, j+3,
               j+4, j+4, j+4, j+4, j+4, j+4, j+4, j+4,
               j+5, j+5, j+5, j+5, j+5, j+5, j+5, j+5,
               j+6, j+6, j+6, j+6, j+6, j+6, j+6, j+6,
               j+7, j+7, j+7, j+7, j+7, j+7, j+7, j+7});
#else
#error "This VECL && This WID cannot work!"
#define xstr(s) str(s)
#define str(s) #s
#pragma message "VECL =" xstr(VECL)
#pragma message "WID = "xstr(WID)
#endif

         loop_body(i_indices,j_indices,k,j);

      }
   }
}

/* Storage of Temperature anisotropy to beta parallel array for pitch-angle diffusion parametrization
   see: Parametrization of coefficients for sub-grid modeling of pitch-angle diffusion in global
   magnetospheric hybrid-Vlasov simulations, M. Dubart, M. Battarbee, U. Ganse, A. Osmane, F. Spanier,
   J. Suni, G. Cozzani, K. Horaites, K. Papadakis, Y. Pfau-Kempf, V. Tarvus, and M. Palmroth,
   Physics of Plasmas 30, 123903 (2023)
   https://doi.org/10.1063/5.0176376
 */

/* Linear interpolation of diffusion coefficient from above array
 */
Realf interpolateNuFromArray(
   const Real Taniso_in,
   const Real betaParallel_in
   ) {
   Real Taniso = Taniso_in;
   Real betaParallel = betaParallel_in;
   int betaIndx = -1;
   int TanisoIndx = -1;
   for (size_t i = 0; i < betaParaArray.size(); i++) {
      if (betaParallel >= betaParaArray[i]) {
         betaIndx   = i;
      }
   }
   for (size_t i = 0; i < TanisoArray.size()  ; i++) {
      if (Taniso       >= TanisoArray[i]  ) {
         TanisoIndx = i;
      }
   }

   if ( (betaIndx < 0) || (TanisoIndx < 0) ) {
      // Values below table lower bounds; no diffusion required.
      return 0.0;
   } else {
      // Interpolate values from table; if values are above bounds, cap to maximum value.
      if (betaIndx >= (int)betaParaArray.size()-1) {
         betaIndx = (int)betaParaArray.size()-2; // force last bin
         betaParallel = betaParaArray[betaIndx+1]; // force interpolation to bin top
      }
      if (TanisoIndx >= (int)TanisoArray.size()-1) {
         TanisoIndx = (int)TanisoArray.size()-2; // force last bin
         Taniso = TanisoArray[TanisoIndx+1]; // force interpolation to bin top
      }
      // bi-linear interpolation with weighted mean to find nu0(betaParallel,Taniso)
      const Real beta1   = betaParaArray[betaIndx];
      const Real beta2   = betaParaArray[betaIndx+1];
      const Real Taniso1 = TanisoArray[TanisoIndx];
      const Real Taniso2 = TanisoArray[TanisoIndx+1];
      const Real nu011   = nu0Array[betaIndx*n_Taniso+TanisoIndx];
      const Real nu012   = nu0Array[betaIndx*n_Taniso+TanisoIndx+1];
      const Real nu021   = nu0Array[(betaIndx+1)*n_Taniso+TanisoIndx];
      const Real nu022   = nu0Array[(betaIndx+1)*n_Taniso+TanisoIndx+1];
      // Weights
      const Real w11 = (beta2 - betaParallel)*(Taniso2 - Taniso)  / ( (beta2 - beta1)*(Taniso2-Taniso1) );
      const Real w12 = (beta2 - betaParallel)*(Taniso  - Taniso1) / ( (beta2 - beta1)*(Taniso2-Taniso1) );
      const Real w21 = (betaParallel - beta1)*(Taniso2 - Taniso)  / ( (beta2 - beta1)*(Taniso2-Taniso1) );
      const Real w22 = (betaParallel - beta1)*(Taniso  - Taniso1) / ( (beta2 - beta1)*(Taniso2-Taniso1) );
      // Linear interpolation (with fudge factor divisor)
      return (w11*nu011 + w12*nu012 + w21*nu021 + w22*nu022)/Parameters::PADfudge;
   }
}

__global__ void build2dArrayOfFvmu(size_t *dev_cellIdxArray, vmesh::LocalID *dev_velocityBlockIdxArray, int maxGPUIndex){
   int idx = blockIdx.x*blockDim.x + threadIdx.x;

   if(idx >= maxGPUIndex){return;}

   int j = idx%WID;
   int k = (idx/WID)%WID;
   int totalBlockIndex = idx/WID2; // Corresponds to index spatial and velocity blocks
   size_t cellIdx = dev_cellIdxArray[totalBlockIndex];
   vmesh::LocalID velocityBlockIdx = dev_velocityBlockIdxArray[totalBlockIndex];
}

void pitchAngleDiffusion(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid, const uint popID){

   // Ensure nu0 dat file is read, if requested
   if (P::PADcoefficient < 0) {
      readNuArrayFromFile();
   }

   int nbins_v  = Parameters::PADvbins;
   int nbins_mu = Parameters::PADmubins;
   const Real dmubins = 2.0/nbins_mu;

   // resonance gap filling coefficient, not needed assuming even number of bins in mu-space
   const Real epsilon = 0.0;

   phiprof::Timer diffusionTimer {"pitch-angle-diffusion"};

   const auto LocalCells=getLocalCells();
   
   //!!!TEST REGION STARTS, DO NOT INCLUDE IN PRODUCTION CODE!!!
   /*
   std::cout << LocalCells.size() << ", " << nbins_v << ", " << nbins_mu << "\n";

   const auto CellID                  = LocalCells[0];
   SpatialCell& cell                  = *mpiGrid[CellID];

   std::cout << cell.get_number_of_velocity_blocks(popID)*WID3/VECL << '\n';
   std::cout << cell.get_number_of_velocity_blocks(popID) << '\n';
   std::cout << VECL << '\n';
   std::cout << WID << '\n';
   std::cout << omp_get_max_threads() << '\n';

   int global_rank = -1;
   MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
   int device = -1;
   gpuGetDevice(&device);
   std::cout << "Rank " << global_rank << " using GPU " << device << '\n';
   */
   //!!!TEST REGION ENDS, PRODUCTION CODE CONTINUES!!!

   size_t numberOfLocalCells = LocalCells.size();

   std::vector<Real> bValues (3*numberOfLocalCells, 0.0);
   std::vector<Real> nu0Values (numberOfLocalCells, 0.0);
   std::vector<Realf> density_pre_adjust (numberOfLocalCells, 0.0);
   
   std::vector<bool> spatialLoopComplete(numberOfLocalCells, false);

   for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells

      const auto CellID                  = LocalCells[CellIdx];
      SpatialCell& cell                  = *mpiGrid[CellID];
      const Real* parameters             = cell.get_block_parameters(popID);
      const size_t meshID = getObjectWrapper().particleSpecies[popID].velocityMesh;
      const vmesh::MeshParameters& vMesh = vmesh::getMeshWrapper()->velocityMeshes->at(meshID);

      // Ensure mass conservation
      if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
         Vec vectorSum {0};
         Vec vectorAdd {0};
         for (size_t i=0; i<cell.get_number_of_velocity_blocks(popID)*WID3/VECL; ++i) {
            vectorAdd.load(&cell.get_data(popID)[i*VECL]);
            vectorSum += vectorAdd;
            //density_pre_adjust[CellIdx] += cell.get_data(popID)[i];
         }
         density_pre_adjust[CellIdx] = horizontal_add(vectorSum);
      }

      // Diffusion coefficient to use in this cell
      Real nu0 = 0.0;

      const std::array<Real,3> B = {cell.parameters[CellParams::PERBXVOL] +  cell.parameters[CellParams::BGBXVOL],
         cell.parameters[CellParams::PERBYVOL] +  cell.parameters[CellParams::BGBYVOL],
         cell.parameters[CellParams::PERBZVOL] +  cell.parameters[CellParams::BGBZVOL]};
      const Real Bnorm           = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
      const std::array<Real,3> b = {B[0]/Bnorm, B[1]/Bnorm, B[2]/Bnorm};
      
      // Save values of b to vector 
      bValues[3*CellIdx] = b[0];
      bValues[3*CellIdx+1] = b[1];
      bValues[3*CellIdx+2] = b[2];

      if (P::PADcoefficient >= 0) {
         // User-provided single diffusion coefficient
         nu0 = P::PADcoefficient;
      } else {
         // Use nu0 values based on Taniso and betaPara read from file
         if (!nuArrayRead) {
            std::cerr<<" ERROR! Attempting to interpolate nu0 value but file has not been read."<<std::endl;
            abort();
         }

         // Perform Eigen rotation to find parallel and perpendicular pressure
         Eigen::Matrix3d rot = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d{b[0], b[1], b[2]}, Eigen::Vector3d{0, 0, 1}).normalized().toRotationMatrix();
         Eigen::Matrix3d Ptensor {
            {cell.parameters[CellParams::P_11], cell.parameters[CellParams::P_12], cell.parameters[CellParams::P_13]},
            {cell.parameters[CellParams::P_12], cell.parameters[CellParams::P_22], cell.parameters[CellParams::P_23]},
            {cell.parameters[CellParams::P_13], cell.parameters[CellParams::P_23], cell.parameters[CellParams::P_33]},
         };
         Eigen::Matrix3d transposerot = rot.transpose();
         Eigen::Matrix3d Pprime = rot * Ptensor * transposerot;

         // Anisotropy
         Real Taniso = 0.0;
         if (Pprime(2, 2) > std::numeric_limits<Real>::min()) {
            Taniso = (Pprime(0, 0) + Pprime(1, 1)) / (2 * Pprime(2, 2));
         }
         // Beta Parallel
         Real betaParallel = 0.0;
         if (Bnorm > 0) {
            betaParallel = 2.0 * physicalconstants::MU_0 * Pprime(2, 2) / (Bnorm*Bnorm);
         }
         // Find anisotropy and beta parallel indexes from read table
         nu0 = interpolateNuFromArray(Taniso,betaParallel);
      }

      nu0Values[CellIdx] = nu0;

      // Enable nu0 disk output; skip cells where diffusion is not required (or diffusion coefficient is very small).
      cell.parameters[CellParams::NU0] = nu0;
      if (nu0 <= 0.001) {
         spatialLoopComplete[CellIdx] = true;
      }
   } // End spatial cell loop

   std::vector<std::vector<int>>   fcount (numberOfLocalCells, std::vector<int>(nbins_v*nbins_mu,0)); // Array to count number of f stored for each spatial cells
   std::vector<std::vector<Realf>> fmu    (numberOfLocalCells, std::vector<Realf>(nbins_v*nbins_mu,0.0)); // Array to store f(v,mu) for each spatial cells
   std::vector<std::vector<Realf>> dfdmu  (numberOfLocalCells, std::vector<Realf>(nbins_v*nbins_mu,0.0)); // Array to store dfdmu for each spatial cells
   std::vector<std::vector<Realf>> dfdmu2 (numberOfLocalCells, std::vector<Realf>(nbins_v*nbins_mu,0.0)); // Array to store dfdmumu for each spatial cells
   std::vector<std::vector<Realf>> dfdt_mu (numberOfLocalCells, std::vector<Realf>(nbins_v*nbins_mu,0.0)); // Array to store dfdt_mu for each spatial cells

   std::vector<Real> dtTotalDiff(numberOfLocalCells, 0.0); // Diffusion time elapsed for each spatial cells
   bool allSpatialCellTimeLoopsComplete = true;
   // Check if at least one cell needs to be calculated
   for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
      if(!spatialLoopComplete[CellIdx]){
         allSpatialCellTimeLoopsComplete = false;
         break;
      }
   }

   while (!allSpatialCellTimeLoopsComplete) { // Substep loop

      // Construct cellIdx and velocityBlockIdx arrays
      std::vector<size_t> host_cellIdxArray;
      std::vector<vmesh::LocalID> host_velocityBlockIdxArray;
      // And load CPU data
      std::vector<Real> host_dVbins (numberOfLocalCells);

      int maxBlockIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];
         const Real* parameters             = cell.get_block_parameters(popID);
         const size_t meshID = getObjectWrapper().particleSpecies[popID].velocityMesh;
         const vmesh::MeshParameters& vMesh = vmesh::getMeshWrapper()->velocityMeshes->at(meshID);

         const Real Vmax   = 2*sqrt(3)*vMesh.meshLimits[1];
         host_dVbins[CellIdx] = Vmax/nbins_v;
         
         // Initialised at each substep
         std::fill(fmu[CellIdx].begin(), fmu[CellIdx].end(), 0.0);
         std::fill(fcount[CellIdx].begin(), fcount[CellIdx].end(), 0);

         // Add elements to cellIdx and velocityBlockIdx arrays
         for (vmesh::LocalID n=0; n<cell.get_number_of_velocity_blocks(popID); n++) { // Iterate through velocity blocks
            host_cellIdxArray.push_back(CellIdx);
            host_velocityBlockIdxArray.push_back(n);
            maxBlockIndex++;
         } // End blocks
      } // End spatial cell loop
      int maxGPUIndex = maxBlockIndex*WID2;

      // !!! TEST REGION, DO NOT INCLUDE IN PRODUCITON CODE !!!
      size_t *dev_cellIdxArray;
      vmesh::LocalID *dev_velocityBlockIdxArray;

      gpuMalloc((void**)&dev_cellIdxArray, maxBlockIndex*sizeof(size_t));
      gpuMalloc((void**)&dev_velocityBlockIdxArray, maxBlockIndex*sizeof(vmesh::LocalID));

      gpuMemcpy(dev_cellIdxArray, host_cellIdxArray.data(), maxBlockIndex*sizeof(size_t), gpuMemcpyHostToDevice);
      gpuMemcpy(dev_velocityBlockIdxArray, host_velocityBlockIdxArray.data(), maxBlockIndex*sizeof(vmesh::LocalID), gpuMemcpyHostToDevice);

      int threadsPerBlock = 512;
      int blocksPerGrid = (maxGPUIndex+threadsPerBlock-1)/threadsPerBlock;

      build2dArrayOfFvmu<<<blocksPerGrid, threadsPerBlock>>>(dev_cellIdxArray, dev_velocityBlockIdxArray, maxGPUIndex);

      gpuFree(dev_cellIdxArray);
      gpuFree(dev_velocityBlockIdxArray);

      // !!! TEST REGION ENDS, PRODUCTION CODE CONTINUES !!!

      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];
         const Real* parameters             = cell.get_block_parameters(popID);

         const Real bulkVX = cell.parameters[CellParams::VX];
         const Real bulkVY = cell.parameters[CellParams::VY];
         const Real bulkVZ = cell.parameters[CellParams::VZ];

         vmesh::LocalID numberOfVelocityBlocks = cell.get_number_of_velocity_blocks(popID);
         
         // Load cell values
         std::vector<Vec> CellValue (numberOfVelocityBlocks*WID2);
         for (vmesh::LocalID n=0; n<numberOfVelocityBlocks; n++) { // Iterate through velocity blocks
            loop_over_block([&](Veci i_indices, Veci j_indices, int k, int j) -> void { // Lambda function processor
               CellValue[n*WID2+k*WID+j].load(&cell.get_data(n,popID)[WID2*k + WID*j_indices[0] + i_indices[0]]);
            }); // End of Lambda
         } // End blocks

         // Build 2d array of f(v,mu)
         for (vmesh::LocalID n=0; n<numberOfVelocityBlocks; n++) { // Iterate through velocity blocks

            loop_over_block([&](Veci i_indices, Veci j_indices, int k, int j) -> void { // Lambda function processor

               //Get velocity space coordinates
               const Vec VX(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD]
                              + (to_realf(i_indices) + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX]);
               const Vec VY(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD]
                              + (to_realf(j_indices) + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY]);
               const Vec VZ(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD]
                              + (k + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ]);

               const Vec VplasmaX = VX - bulkVX;
               const Vec VplasmaY = VY - bulkVY;
               const Vec VplasmaZ = VZ - bulkVZ;

               const Vec normV = sqrt(VplasmaX*VplasmaX + VplasmaY*VplasmaY + VplasmaZ*VplasmaZ);
               const Vec Vpara = VplasmaX*bValues[3*CellIdx] + VplasmaY*bValues[3*CellIdx+1] + VplasmaZ*bValues[3*CellIdx+2];
               const Vec mu = Vpara/(normV+std::numeric_limits<Real>::min()); // + min value to avoid division by 0.

               const Veci Vindex = roundi(floor((normV) / host_dVbins[CellIdx]));
               const Vec Vmu = host_dVbins[CellIdx] * (to_realf(Vindex)+0.5); // Take value at the center of the mu cell
               Veci muindex = roundi(floor((mu+1.0) / dmubins));

               const Vec increment = 2.0 * M_PI * Vmu*Vmu * CellValue[n*WID2+k*WID+j];
               for (uint i = 0; i<VECL; i++) {
                  // Safety check to handle edge case where mu = exactly 1.0
                  const int mui = std::max(0,std::min((int)muindex[i],nbins_mu-1));
                  const int vi = std::max(0,std::min((int)Vindex[i],nbins_v-1));
                  MUSPACE(fmu[CellIdx],vi,mui) += increment[i];
                  MUSPACE(fcount[CellIdx],vi,mui) += 1;
               }
            }); // End of Lambda
         } // End blocks
      } // End spatial cell loop

      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];
         const Real* parameters             = cell.get_block_parameters(popID);
         const size_t meshID = getObjectWrapper().particleSpecies[popID].velocityMesh;
         const vmesh::MeshParameters& vMesh = vmesh::getMeshWrapper()->velocityMeshes->at(meshID);

         const Real bulkVX = cell.parameters[CellParams::VX];
         const Real bulkVY = cell.parameters[CellParams::VY];
         const Real bulkVZ = cell.parameters[CellParams::VZ];

         const Real Vmax   = 2*sqrt(3)*vMesh.meshLimits[1];
         const Real dVbins = Vmax/nbins_v;
         const Realf Sparsity   = 0.01 * cell.getVelocityBlockMinValue(popID);

         const Real RemainT  = Parameters::dt - dtTotalDiff[CellIdx]; //Remaining time before reaching simulation time step
         Real checkCFL = std::numeric_limits<Real>::max();

         // Search limits for how many cells in mu-direction should be max evaluated when searching for a near neighbour?
         // Assuming some oversampling; changing these values may result in method breaking at very small plasma frame velocities.
         std::vector<int> cRight (nbins_v*nbins_mu);
         std::vector<int> cLeft (nbins_v*nbins_mu);
         const int rlimit = nbins_mu-1;
         const int llimit = 0;

         for (int indv = 0; indv < nbins_v; indv++) {
            for(int indmu = 0; indmu < nbins_mu; indmu++) {
               if (indmu == 0) {
                  cLeft[indv*nbins_mu+indmu]  = 0;
                  cRight[indv*nbins_mu+indmu] = 1;
                  while( (MUSPACE(fcount[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) == 0) && (indmu + cRight[indv*nbins_mu+indmu] < rlimit) )  { cRight[indv*nbins_mu+indmu] += 1; }
                  if(    (MUSPACE(fcount[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) == 0) && (indmu + cRight[indv*nbins_mu+indmu] == rlimit) ) { cRight[indv*nbins_mu+indmu]  = 0; }
               } else if (indmu == nbins_mu-1) {
                  cLeft[indv*nbins_mu+indmu]  = 1;
                  cRight[indv*nbins_mu+indmu] = 0;
                  while( (MUSPACE(fcount[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]) == 0) && (indmu - cLeft[indv*nbins_mu+indmu] > llimit) )  { cLeft[indv*nbins_mu+indmu] += 1; }
                  if(    (MUSPACE(fcount[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]) == 0) && (indmu - cLeft[indv*nbins_mu+indmu] == llimit) ) { cLeft[indv*nbins_mu+indmu]  = 0; }
               } else {
                  cLeft[indv*nbins_mu+indmu]  = 1;
                  cRight[indv*nbins_mu+indmu] = 1;
                  while( (MUSPACE(fcount[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) == 0) && (indmu + cRight[indv*nbins_mu+indmu] < rlimit) )  { cRight[indv*nbins_mu+indmu] += 1; }
                  if(    (MUSPACE(fcount[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) == 0) && (indmu + cRight[indv*nbins_mu+indmu] == rlimit) ) { cRight[indv*nbins_mu+indmu]  = 0; }
                  while( (MUSPACE(fcount[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu] ) == 0) && (indmu - cLeft[indv*nbins_mu+indmu]  > llimit) )           { cLeft[indv*nbins_mu+indmu]  += 1; }
                  if(    (MUSPACE(fcount[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu] ) == 0) && (indmu - cLeft[indv*nbins_mu+indmu]  == llimit) )          { cLeft[indv*nbins_mu+indmu]   = 0; }
               }
            }
         }

         // Compute space/time derivatives (take first non-zero neighbours) & CFL & Ddt
         for (int indv = 0; indv < nbins_v; indv++) {
            const Real Vmu = dVbins * (float(indv)+0.5);

            // Divide f by count (independent of v but needs to be computed for all mu before derivatives)
            for(int indmu = 0; indmu < nbins_mu; indmu++) {
               if (MUSPACE(fcount[CellIdx],indv,indmu) == 0 || MUSPACE(fmu[CellIdx],indv,indmu) <= 0.0) {
                  MUSPACE(fmu[CellIdx],indv,indmu) = 0;
               } else {
                  MUSPACE(fmu[CellIdx],indv,indmu) = MUSPACE(fmu[CellIdx],indv,indmu) / MUSPACE(fcount[CellIdx],indv,indmu);
               }
            }

            for(int indmu = 0; indmu < nbins_mu; indmu++) {
               // Compute spatial derivatives
               if( (cRight[indv*nbins_mu+indmu] == 0) && (cLeft[indv*nbins_mu+indmu] != 0) ) {
                  MUSPACE(dfdmu[CellIdx] ,indv,indmu) = (MUSPACE(fmu[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) - MUSPACE(fmu[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]))/((cRight[indv*nbins_mu+indmu] + cLeft[indv*nbins_mu+indmu])*dmubins) ;
                  MUSPACE(dfdmu2[CellIdx],indv,indmu) = 0.0;
               } else if( (cLeft[indv*nbins_mu+indmu] == 0) && (cRight[indv*nbins_mu+indmu] != 0) ) {
                  MUSPACE(dfdmu[CellIdx] ,indv,indmu) = (MUSPACE(fmu[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) - MUSPACE(fmu[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]))/((cRight[indv*nbins_mu+indmu] + cLeft[indv*nbins_mu+indmu])*dmubins) ;
                  MUSPACE(dfdmu2[CellIdx],indv,indmu) = 0.0;
               } else if( (cLeft[indv*nbins_mu+indmu] == 0) && (cRight[indv*nbins_mu+indmu] == 0) ) {
                  MUSPACE(dfdmu[CellIdx] ,indv,indmu) = 0.0;
                  MUSPACE(dfdmu2[CellIdx],indv,indmu) = 0.0;
               } else {
                  MUSPACE(dfdmu[CellIdx] ,indv,indmu) = (  MUSPACE(fmu[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) - MUSPACE(fmu[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]))/((cRight[indv*nbins_mu+indmu] + cLeft[indv*nbins_mu+indmu])*dmubins) ;
                  MUSPACE(dfdmu2[CellIdx],indv,indmu) = ( (MUSPACE(fmu[CellIdx],indv,indmu + cRight[indv*nbins_mu+indmu]) - MUSPACE(fmu[CellIdx],indv,indmu))/(cRight[indv*nbins_mu+indmu]*dmubins) - (MUSPACE(fmu[CellIdx],indv,indmu) - MUSPACE(fmu[CellIdx],indv,indmu - cLeft[indv*nbins_mu+indmu]))/(cLeft[indv*nbins_mu+indmu]*dmubins) ) / (0.5 * dmubins * (cRight[indv*nbins_mu+indmu] + cLeft[indv*nbins_mu+indmu]));
               }

               // Compute time derivative
               const Realf mu    = (indmu+0.5)*dmubins - 1.0;
               const Realf Dmumu = nu0Values[CellIdx]/2.0 * ( abs(mu)/(1.0 + abs(mu)) + epsilon ) * (1.0 - mu*mu);
               const Realf dDmu  = nu0Values[CellIdx]/2.0 * ( (mu/abs(mu)) * ((1.0 - mu*mu)/((1.0 + abs(mu))*(1.0 + abs(mu)))) - 2.0*mu*( abs(mu)/(1.0 + abs(mu)) + epsilon));
               // We divide dfdt_mu by the normalization factor 2pi*v^2 already here.
               const Realf dfdt_mu_val = ( dDmu * MUSPACE(dfdmu[CellIdx],indv,indmu) + Dmumu * MUSPACE(dfdmu2[CellIdx],indv,indmu) ) / (2.0 * M_PI * Vmu*Vmu);
               MUSPACE(dfdt_mu[CellIdx],indv,indmu) = dfdt_mu_val;

               // Only consider CFL for non-negative phase-space cells above the sparsity threshold
               const Realf CellValue = MUSPACE(fmu[CellIdx],indv,indmu) / (2.0 * M_PI * Vmu*Vmu);
               const Realf absdfdt = abs(MUSPACE(dfdt_mu[CellIdx],indv,indmu)); // Already scaled
               if (absdfdt > 0.0 && CellValue > Sparsity) {
                  checkCFL = std::min(CellValue * Parameters::PADCFL * (1.0/absdfdt), checkCFL);
               }
            } // End mu loop
         } // End v loop

         // Compute Ddt
         Real Ddt = checkCFL;
         if (Ddt > RemainT) {
            Ddt = RemainT;
         }
         
         for (vmesh::LocalID n=0; n<cell.get_number_of_velocity_blocks(popID); n++) { // Iterate through velocity blocks

            loop_over_block([&](Veci i_indices, Veci j_indices, int k, int j) -> void { // Lambda function processor

               //Get velocity space coordinates
               const Vec VX(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD]
                              + (to_realf(i_indices) + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX]);
               const Vec VY(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD]
                              + (to_realf(j_indices) + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY]);
               const Vec VZ(parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD]
                              + (k + 0.5)*parameters[n * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ]);

               const Vec VplasmaX = VX - bulkVX;
               const Vec VplasmaY = VY - bulkVY;
               const Vec VplasmaZ = VZ - bulkVZ;

               const Vec normV = sqrt(VplasmaX*VplasmaX + VplasmaY*VplasmaY + VplasmaZ*VplasmaZ);
               const Vec Vpara = VplasmaX*bValues[3*CellIdx] + VplasmaY*bValues[3*CellIdx+1] + VplasmaZ*bValues[3*CellIdx+2];
               const Vec mu = Vpara/(normV+std::numeric_limits<Real>::min()); // + min value to avoid division by 0.

               const Veci Vindex = roundi(floor((normV) / dVbins));
               const Vec Vmu = dVbins * (to_realf(Vindex)+0.5); // Take value at the center of the mu cell
               Veci muindex = roundi(floor((mu+1.0) / dmubins));

               // Compute dfdt
               std::array<Realf,VECL> dfdt = {0};
               for (uint i = 0; i < VECL; i++) {
                  // Safety check to handle edge case where mu = exactly 1.0
                  const int mui = std::max(0,std::min((int)muindex[i],nbins_mu-1));
                  const int vi = std::max(0,std::min((int)Vindex[i],nbins_v-1));
                  dfdt[i] = MUSPACE(dfdt_mu[CellIdx],vi,mui); // dfdt_mu was scaled back down by 2pi*v^2 on creation
               }
               Vec dfdtUpdate;
               dfdtUpdate.load(&dfdt[0]);

               // Update cell value, ensuring result is non-negative
               Vec CellValue;
               CellValue.load(&cell.get_data(n,popID)[WID2*k + WID*j_indices[0] + i_indices[0]]);
               Vec NewCellValue    = CellValue + dfdtUpdate * Ddt;
               const Vecb lessZero = NewCellValue < 0.0;
               NewCellValue        = select(lessZero,0.0,NewCellValue);
               NewCellValue.store(&cell.get_data(n,popID)[WID2*k + WID*j_indices[0] + i_indices[0]]);
            }); // End of Lambda
         } // End Blocks
         dtTotalDiff[CellIdx] += Ddt;
      } // End spatial cell loop

      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         allSpatialCellTimeLoopsComplete = true;
         if(dtTotalDiff[CellIdx] < Parameters::dt){
            allSpatialCellTimeLoopsComplete = false;
         }else{
            spatialLoopComplete[CellIdx] = true;
         }
      }

   } // End Time loop
   for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells

      const auto CellID                  = LocalCells[CellIdx];
      SpatialCell& cell                  = *mpiGrid[CellID];

      // Ensure mass conservation
      if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
         Vec vectorSum {0};
         Vec vectorAdd {0};
         for (size_t i=0; i<cell.get_number_of_velocity_blocks(popID)*WID3/VECL; ++i) {
            vectorAdd.load(&cell.get_data(popID)[i*VECL]);
            vectorSum += vectorAdd;
         }
         Realf density_post_adjust = horizontal_add(vectorSum);

         if (density_post_adjust != 0.0 && density_pre_adjust[CellIdx] != density_post_adjust) {
            const Vec adjustRatio = density_pre_adjust[CellIdx]/density_post_adjust;
            Vec vectorAdjust;
            for (size_t i=0; i<cell.get_number_of_velocity_blocks(popID)*WID3/VECL; ++i) {
               vectorAdjust.load(&cell.get_data(popID)[i*VECL]);
               vectorAdjust *= adjustRatio;
               vectorAdjust.store(&cell.get_data(popID)[i*VECL]);
            }
         }
      }
   } // End spatial cell loop
} // End function