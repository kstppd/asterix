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
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "vec.h"
#include "gpu_pitch_angle_diffusion.hpp"
#include "common_pitch_angle_diffusion.hpp"

#define MUSPACE(var,v_ind,mu_ind) var.at((mu_ind)*nbins_v + (v_ind))
#define CELLMUSPACE(var,cellIdx,v_ind,mu_ind) var.at((cellIdx)*nbins_v*nbins_mu+(mu_ind)*nbins_v + (v_ind))
#define GPUCELLMUSPACE(var,cellIdx,v_ind,mu_ind) var[(cellIdx)*nbins_v*nbins_mu+(mu_ind)*nbins_v + (v_ind)]

// The original code used these namespaces and as I'm not sure where they are used (which is a problem with namespaces),
// they are still here
using namespace spatial_cell;
using namespace Eigen;

unsigned int nextPowerOfTwo(unsigned int n) {
   if (n == 0) return 1;
   n--; // Handle exact powers of two
   n |= n >> 1;
   n |= n >> 2;
   n |= n >> 4;
   n |= n >> 8;
   n |= n >> 16;
   return n + 1;
}

__global__ void build2dArrayOfFvmu_kernel(
   size_t *dev_cellIdxArray, Real *dev_parameters, Real *dev_bulkVX, Real* dev_bulkVY,
   Real* dev_bulkVZ, Real *dev_bValues, Real *dev_dVbins, Real *dev_cellValues, Realf *dev_fmu,
   int *dev_fcount, const Real dmubins, int nbins_v, int nbins_mu, int maxBlockIndex
   ){
   
   int totalBlockIndex = blockIdx.x; // Corresponds to index spatial and velocity blocks

   if(totalBlockIndex >= maxBlockIndex){return;}

   const int i = threadIdx.x;
   const int j = threadIdx.y;
   const int k = threadIdx.z;
   size_t cellIdx = dev_cellIdxArray[totalBlockIndex];

   const Real VX = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD]
               + (i + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX];
   const Real VY = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD]
               + (j + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY];
   const Real VZ = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD]
               + (k + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ];

   const Real VplasmaX = VX - dev_bulkVX[cellIdx];
   const Real VplasmaY = VY - dev_bulkVY[cellIdx];
   const Real VplasmaZ = VZ - dev_bulkVZ[cellIdx];
   
   const Real normV = sqrt(VplasmaX*VplasmaX + VplasmaY*VplasmaY + VplasmaZ*VplasmaZ);
   const Real Vpara = VplasmaX*dev_bValues[3*cellIdx] + VplasmaY*dev_bValues[3*cellIdx+1] + VplasmaZ*dev_bValues[3*cellIdx+2];
   const Real mu = Vpara/(normV+std::numeric_limits<Real>::min()); // + min value to avoid division by 0.

   const int Vindex = static_cast<int>(std::nearbyint(floor((normV) / dev_dVbins[cellIdx])));
   const Real Vmu = dev_dVbins[cellIdx] * (Vindex+0.5); // Take value at the center of the mu cell
   int muindex = static_cast<int>(std::nearbyint(floor((mu+1.0) / dmubins)));

   const Real increment = 2.0 * M_PI * Vmu*Vmu * dev_cellValues[totalBlockIndex*WID3+k*WID2+j*WID+i];
   // Safety check to handle edge case where mu = exactly 1.0
   const int mui = std::max(0,std::min(muindex,nbins_mu-1));
   const int vi = std::max(0,std::min(Vindex,nbins_v-1));

   // TODO: can this be done without atomicAdd while avoiding race conditions?
   atomicAdd(&GPUCELLMUSPACE(dev_fmu,cellIdx,vi,mui), increment);
   atomicAdd(&GPUCELLMUSPACE(dev_fcount,cellIdx,vi,mui), 1);
}

__global__ void computeNewCellValues_kernel(
   size_t *dev_cellIdxArray, size_t *dev_remappedCellIdxArray, Real *dev_parameters,
   Real *dev_bulkVX, Real* dev_bulkVY, Real* dev_bulkVZ, Real *dev_bValues,
   Real *dev_dVbins, Real *dev_cellValues, Realf *dev_dfdt_mu, Real *dev_Ddt,
   const Real dmubins, int nbins_v, int nbins_mu, int maxBlockIndex
   ){
   
   int totalBlockIndex = blockIdx.x; // Corresponds to index spatial and velocity blocks
   
   if(totalBlockIndex >= maxBlockIndex){return;}

   const int i = threadIdx.x;
   const int j = threadIdx.y;
   const int k = threadIdx.z;
   size_t cellIdx = dev_cellIdxArray[totalBlockIndex];
   size_t remappedCellIdx = dev_remappedCellIdxArray[totalBlockIndex];

   //Get velocity space coordinates
   const Real VX = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD]
               + (i + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX];
   const Real VY = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD]
               + (j + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY];
   const Real VZ = dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD]
               + (k + 0.5)*dev_parameters[totalBlockIndex * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ];

   const Real VplasmaX = VX - dev_bulkVX[cellIdx];
   const Real VplasmaY = VY - dev_bulkVY[cellIdx];
   const Real VplasmaZ = VZ - dev_bulkVZ[cellIdx];
   
   const Real normV = sqrt(VplasmaX*VplasmaX + VplasmaY*VplasmaY + VplasmaZ*VplasmaZ);
   const Real Vpara = VplasmaX*dev_bValues[3*cellIdx] + VplasmaY*dev_bValues[3*cellIdx+1] + VplasmaZ*dev_bValues[3*cellIdx+2];
   const Real mu = Vpara/(normV+std::numeric_limits<Real>::min()); // + min value to avoid division by 0.

   const int Vindex = static_cast<int>(std::nearbyint(floor((normV) / dev_dVbins[cellIdx])));
   int muindex = static_cast<int>(std::nearbyint(floor((mu+1.0) / dmubins)));

   Realf dfdt = 0.0;
   // Safety check to handle edge case where mu = exactly 1.0
   const int mui = std::max(0,std::min(muindex,nbins_mu-1));
   const int vi = std::max(0,std::min(Vindex,nbins_v-1));
   dfdt = GPUCELLMUSPACE(dev_dfdt_mu,cellIdx,vi,mui); // dfdt_mu was scaled back down by 2pi*v^2 on creation

   // Update cell value, ensuring result is non-negative
   Real NewCellValue    = dev_cellValues[totalBlockIndex*WID3+k*WID2+j*WID+i] + dfdt * dev_Ddt[remappedCellIdx];
   const bool lessZero = (NewCellValue < 0.0);
   NewCellValue = lessZero ? 0.0 : NewCellValue;
   dev_cellValues[totalBlockIndex*WID3+k*WID2+j*WID+i] = NewCellValue;
}

__global__ void computeDerivativesCFLDdt_kernel(
   size_t *dev_smallCellIdxArray, Real *dev_dVbins, int *dev_cRight, int *dev_cLeft,
   Realf *dev_fmu, Real *dev_nu0Values, Realf *dev_dfdt_mu, int *dev_cellIdxKeys,
   Real *dev_potentialDdtValues, Realf *dev_sparsity, const Real dmubins, const Real epsilon,
   Realf PADCFL, int nbins_v, int nbins_mu, int blocksPerVelocityCell, int lastBlockSize
   ){

   int spatialBlockIndex = blockIdx.x/blocksPerVelocityCell; // Corresponds to index spatial and velocity blocks
   int indexInsideBlock = blockIdx.x%blocksPerVelocityCell;
   int nextIndexInsideBlock = (blockIdx.x+1)%blocksPerVelocityCell;
   int idx = indexInsideBlock*blockDim.x+threadIdx.x;
   int threadIndex = threadIdx.x;
   
   // Initiate shared memory values
   extern __shared__ Real localDdtValues[];
   localDdtValues[threadIndex] = std::numeric_limits<Real>::max();

   if(nextIndexInsideBlock == 0 && threadIndex >= lastBlockSize){return;}

   int indv = idx%nbins_v;
   int indmu = (idx/nbins_v)%nbins_mu;
   size_t cellIdx = dev_smallCellIdxArray[spatialBlockIndex];

   const Real Vmu = dev_dVbins[cellIdx] * (float(indv)+0.5);
   Realf dfdmu = 0.0;
   Realf dfdmu2 = 0.0;
   // Compute spatial derivatives
   if( (GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) == 0) && (GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu) != 0) ) {
      dfdmu = (GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu + GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu)) - GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu - GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)))
               /((GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) + GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu))*dmubins) ;
      dfdmu2 = 0.0;
   } else if( (GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu) == 0) && (GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) != 0) ) {
      dfdmu = (GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu + GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu)) - GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu - GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)))
               /((GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) + GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu))*dmubins) ;
      dfdmu2 = 0.0;
   } else if( (GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu) == 0) && (GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) == 0) ) {
      dfdmu = 0.0;
      dfdmu2 = 0.0;
   } else {
      dfdmu = (  GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu + GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu)) - GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu - GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)))
               /((GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) + GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu))*dmubins) ;
      dfdmu2 = ( (GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu + GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu)) - GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu))/(GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu)*dmubins)
               - (GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) - GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu - GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)))
               /(GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)*dmubins) ) / (0.5 * dmubins * (GPUCELLMUSPACE(dev_cRight,cellIdx,indv,indmu) + GPUCELLMUSPACE(dev_cLeft,cellIdx,indv,indmu)));
   }

   // Compute time derivative
   const Realf mu    = (indmu+0.5)*dmubins - 1.0;
   const Realf Dmumu = dev_nu0Values[cellIdx]/2.0 * ( abs(mu)/(1.0 + abs(mu)) + epsilon ) * (1.0 - mu*mu);
   const Realf dDmu  = dev_nu0Values[cellIdx]/2.0 * ( (mu/abs(mu)) * ((1.0 - mu*mu)/((1.0 + abs(mu))*(1.0 + abs(mu)))) - 2.0*mu*( abs(mu)/(1.0 + abs(mu)) + epsilon));
   // We divide dfdt_mu by the normalization factor 2pi*v^2 already here.
   const Realf dfdt_mu_val = ( dDmu * dfdmu + Dmumu * dfdmu2 ) / (2.0 * M_PI * Vmu*Vmu);
   GPUCELLMUSPACE(dev_dfdt_mu,cellIdx,indv,indmu) = dfdt_mu_val;

   // Only consider CFL for non-negative phase-space cells above the sparsity threshold
   const Realf CellValue = GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) / (2.0 * M_PI * Vmu*Vmu);
   const Realf absdfdt = abs(dfdt_mu_val); // Already scaled
   
   // Save calculated Ddt value
   if (absdfdt > 0.0 && CellValue > dev_sparsity[cellIdx]) {
      localDdtValues[threadIndex] = CellValue * PADCFL * (1.0/absdfdt);
   }

   // Reduction in shared memory
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIndex < s) {
         localDdtValues[threadIndex] = min(localDdtValues[threadIndex], localDdtValues[threadIndex + s]);
      }
      __syncthreads();
   }

   // Write the result from the first thread of each block
   if (threadIndex == 0) {
      dev_potentialDdtValues[blockIdx.x] = localDdtValues[0];
      dev_cellIdxKeys[blockIdx.x] = cellIdx;
   }
}

__global__ void dividefByCount_kernel(
   size_t *dev_smallCellIdxArray, Realf *dev_fmu, int *dev_fcount, int nbins_v, int nbins_mu, int maxThreadIndex
   ){

   int idx = blockIdx.x*blockDim.x + threadIdx.x;

   if(idx >= maxThreadIndex){return;}

   int indv = idx%nbins_v;
   int indmu = (idx/nbins_v)%nbins_mu;
   int spatialBlockIndex = idx/(nbins_v*nbins_mu); // Corresponds to index spatial and velocity blocks
   size_t cellIdx = dev_smallCellIdxArray[spatialBlockIndex];

   if (GPUCELLMUSPACE(dev_fcount,cellIdx,indv,indmu) == 0 || GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) <= 0.0) {
      GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) = 0;
   } else {
      GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) = GPUCELLMUSPACE(dev_fmu,cellIdx,indv,indmu) / GPUCELLMUSPACE(dev_fcount,cellIdx,indv,indmu);
   }
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

   size_t numberOfLocalCells = LocalCells.size();

   std::vector<Real> host_bValues (3*numberOfLocalCells, 0.0);
   std::vector<Real> host_nu0Values (numberOfLocalCells, 0.0);
   std::vector<Realf> host_sparsity (numberOfLocalCells, 0.0);
   std::vector<Realf> density_pre_adjust (numberOfLocalCells, 0.0);
   
   std::vector<bool> spatialLoopComplete(numberOfLocalCells, false);

   for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells

      const auto CellID                  = LocalCells[CellIdx];
      SpatialCell& cell                  = *mpiGrid[CellID];
      const Real* parameters             = cell.get_block_parameters(popID);
      const size_t meshID = getObjectWrapper().particleSpecies[popID].velocityMesh;
      const vmesh::MeshParameters& vMesh = vmesh::getMeshWrapper()->velocityMeshes->at(meshID);
      
      host_sparsity[CellIdx]   = 0.01 * cell.getVelocityBlockMinValue(popID);

      // Ensure mass conservation
      if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
         //TODO: parallelize, tho this not usually used (I think) and probably not that expensive
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
      host_bValues[3*CellIdx] = b[0];
      host_bValues[3*CellIdx+1] = b[1];
      host_bValues[3*CellIdx+2] = b[2];

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

      host_nu0Values[CellIdx] = nu0;

      // Enable nu0 disk output; skip cells where diffusion is not required (or diffusion coefficient is very small).
      cell.parameters[CellParams::NU0] = nu0;
      if (nu0 <= 0.001) {
         spatialLoopComplete[CellIdx] = true;
      }
   } // End spatial cell loop

   // Create device arrays
   Real *dev_bValues, *dev_nu0Values, *dev_sparsity;

   // Allocate memory
   CHK_ERR( gpuMalloc((void**)&dev_bValues, 3*numberOfLocalCells*sizeof(Real)) );
   CHK_ERR( gpuMalloc((void**)&dev_nu0Values, numberOfLocalCells*sizeof(Real)) );
   CHK_ERR( gpuMalloc((void**)&dev_sparsity, numberOfLocalCells*sizeof(Realf)) );

   // Copy data to device
   CHK_ERR( gpuMemcpy(dev_bValues, host_bValues.data(), 3*numberOfLocalCells*sizeof(Real), gpuMemcpyHostToDevice) );
   CHK_ERR( gpuMemcpy(dev_nu0Values, host_nu0Values.data(), numberOfLocalCells*sizeof(Real), gpuMemcpyHostToDevice) );
   CHK_ERR( gpuMemcpy(dev_sparsity, host_sparsity.data(), numberOfLocalCells*sizeof(Realf), gpuMemcpyHostToDevice) );

   std::vector<int>   host_fcount (numberOfLocalCells*nbins_v*nbins_mu,0); // Array to count number of f stored for each spatial cells
   std::vector<int> host_cRight (numberOfLocalCells*nbins_v*nbins_mu);
   std::vector<int> host_cLeft (numberOfLocalCells*nbins_v*nbins_mu);
   std::vector<Realf> host_dfdt_mu (numberOfLocalCells*nbins_v*nbins_mu,0.0); // Array to store dfdt_mu for each spatial cells

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
      // Initialised at each substep
      std::fill(host_fcount.begin(), host_fcount.end(), 0);

      // Compute maximum indices
      int maxBlockIndex = 0;
      int maxCellIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];

         maxBlockIndex += cell.get_number_of_velocity_blocks(popID);
         maxCellIndex++;
      } // End spatial cell loop

      int maxGPUIndex = maxBlockIndex*WID3;

      int maxThreadsPerBlock = 1024;
      int blocksPerVelocityCell = (nbins_v*nbins_mu+maxThreadsPerBlock-1)/maxThreadsPerBlock;

      // Construct cellIdx arrays
      std::vector<size_t> host_cellIdxArray(maxBlockIndex);
      std::vector<size_t> host_smallCellIdxArray(maxCellIndex);
      std::vector<size_t> host_remappedCellIdxArray(maxBlockIndex); // The position of the cell index in the sequence instead of the actual index
      // And load CPU data
      std::vector<Real> host_dVbins (maxCellIndex);
      std::vector<Real> host_bulkVX (maxCellIndex);
      std::vector<Real> host_bulkVY (maxCellIndex);
      std::vector<Real> host_bulkVZ (maxCellIndex);

      // Compute certain values on host
      int remappedCellIdx = 0;
      int totalBlockIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];
         const Real* parameters             = cell.get_block_parameters(popID);
         const size_t meshID = getObjectWrapper().particleSpecies[popID].velocityMesh;
         const vmesh::MeshParameters& vMesh = vmesh::getMeshWrapper()->velocityMeshes->at(meshID);

         host_bulkVX[CellIdx] = cell.parameters[CellParams::VX];
         host_bulkVY[CellIdx] = cell.parameters[CellParams::VY];
         host_bulkVZ[CellIdx] = cell.parameters[CellParams::VZ];

         const Real Vmax   = 2*sqrt(3)*vMesh.meshLimits[1];
         host_dVbins[CellIdx] = Vmax/nbins_v;

         vmesh::LocalID numberOfVelocityBlocks = cell.get_number_of_velocity_blocks(popID);
         
         // Add elements to cellIdx arrays
         std::fill(host_cellIdxArray.begin() + totalBlockIndex, host_cellIdxArray.begin() + totalBlockIndex + numberOfVelocityBlocks, CellIdx);
         std::fill(host_remappedCellIdxArray.begin() + totalBlockIndex, host_remappedCellIdxArray.begin() + totalBlockIndex + numberOfVelocityBlocks, remappedCellIdx);
         host_smallCellIdxArray[remappedCellIdx] = CellIdx;

         remappedCellIdx++;
         totalBlockIndex += numberOfVelocityBlocks;
      } // End spatial cell loop

      // Load cell values
      Real *host_cellValues = new Real[maxGPUIndex];
      totalBlockIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];

         vmesh::LocalID numberOfVelocityBlocks = cell.get_number_of_velocity_blocks(popID);

         std::memcpy(&host_cellValues[totalBlockIndex*WID3], cell.get_data(popID), numberOfVelocityBlocks * WID3 * sizeof(Real));

         totalBlockIndex += numberOfVelocityBlocks;
      } // End spatial cell loop

      // Load parameters
      Real *host_parameters = new Real[maxBlockIndex*BlockParams::N_VELOCITY_BLOCK_PARAMS];
      totalBlockIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; ++CellIdx) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID = LocalCells[CellIdx];
         SpatialCell& cell = *mpiGrid[CellID];
         const Real* cellParameters = cell.get_block_parameters(popID);

         vmesh::LocalID numberOfVelocityBlocks = cell.get_number_of_velocity_blocks(popID);

         std::memcpy(&host_parameters[totalBlockIndex*BlockParams::N_VELOCITY_BLOCK_PARAMS], cellParameters, numberOfVelocityBlocks * BlockParams::N_VELOCITY_BLOCK_PARAMS * sizeof(Real));
         totalBlockIndex += numberOfVelocityBlocks;
      } // End spatial cell loop

      std::vector<Real> host_Ddt (maxCellIndex);

      // Create device arrays
      size_t *dev_cellIdxArray, *dev_smallCellIdxArray, *dev_remappedCellIdxArray;
      Real *dev_dVbins, *dev_bulkVX, *dev_bulkVY, *dev_bulkVZ, *dev_parameters, *dev_Ddt, *dev_cellValues, *dev_potentialDdtValues;
      Realf *dev_fmu, *dev_dfdt_mu;
      int *dev_fcount, *dev_cRight, *dev_cLeft, *dev_cellIdxKeys;

      // Allocate memory
      CHK_ERR( gpuMalloc((void**)&dev_cellIdxArray, maxBlockIndex*sizeof(size_t)) );
      CHK_ERR( gpuMalloc((void**)&dev_smallCellIdxArray, maxCellIndex*sizeof(size_t)) );
      CHK_ERR( gpuMalloc((void**)&dev_remappedCellIdxArray, maxBlockIndex*sizeof(size_t)) );
      CHK_ERR( gpuMalloc((void**)&dev_dVbins, maxCellIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_bulkVX, maxCellIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_bulkVY, maxCellIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_bulkVZ, maxCellIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_Ddt, maxCellIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_parameters, maxBlockIndex*BlockParams::N_VELOCITY_BLOCK_PARAMS*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_cellValues, maxGPUIndex*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_potentialDdtValues, maxCellIndex*blocksPerVelocityCell*sizeof(Real)) );
      CHK_ERR( gpuMalloc((void**)&dev_fmu, numberOfLocalCells*nbins_v*nbins_mu*sizeof(Realf)) );
      CHK_ERR( gpuMalloc((void**)&dev_dfdt_mu, numberOfLocalCells*nbins_v*nbins_mu*sizeof(Realf)) );
      CHK_ERR( gpuMalloc((void**)&dev_fcount, numberOfLocalCells*nbins_v*nbins_mu*sizeof(int)) );
      CHK_ERR( gpuMalloc((void**)&dev_cRight, numberOfLocalCells*nbins_v*nbins_mu*sizeof(int)) );
      CHK_ERR( gpuMalloc((void**)&dev_cLeft, numberOfLocalCells*nbins_v*nbins_mu*sizeof(int)) );
      CHK_ERR( gpuMalloc((void**)&dev_cellIdxKeys, maxCellIndex*blocksPerVelocityCell*sizeof(int)) );

      // Copy data to device
      CHK_ERR( gpuMemcpy(dev_cellIdxArray, host_cellIdxArray.data(), maxBlockIndex*sizeof(size_t), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_smallCellIdxArray, host_smallCellIdxArray.data(), maxCellIndex*sizeof(size_t), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_remappedCellIdxArray, host_remappedCellIdxArray.data(), maxBlockIndex*sizeof(size_t), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_dVbins, host_dVbins.data(), maxCellIndex*sizeof(Real), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_bulkVX, host_bulkVX.data(), maxCellIndex*sizeof(Real), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_bulkVY, host_bulkVY.data(), maxCellIndex*sizeof(Real), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_bulkVZ, host_bulkVZ.data(), maxCellIndex*sizeof(Real), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_parameters, host_parameters, maxBlockIndex*BlockParams::N_VELOCITY_BLOCK_PARAMS*sizeof(Real), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_cellValues, host_cellValues, maxGPUIndex*sizeof(Real), gpuMemcpyHostToDevice) );

      // Initialize with zero values
      CHK_ERR( gpuMemset(dev_fmu, 0.0, numberOfLocalCells*nbins_v*nbins_mu*sizeof(Realf)) );
      CHK_ERR( gpuMemset(dev_dfdt_mu, 0.0, numberOfLocalCells*nbins_v*nbins_mu*sizeof(Realf)) );
      CHK_ERR( gpuMemset(dev_fcount, 0, numberOfLocalCells*nbins_v*nbins_mu*sizeof(int)) );

      // Run the kernel
      dim3 threadsPerBlock(WID, WID, WID);
      int totalThreadsPerBlock = WID3;
      int blocksPerGrid = (maxGPUIndex+totalThreadsPerBlock-1)/totalThreadsPerBlock;

      build2dArrayOfFvmu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
         dev_cellIdxArray, dev_parameters, dev_bulkVX, dev_bulkVY, dev_bulkVZ,
         dev_bValues, dev_dVbins, dev_cellValues, dev_fmu,
         dev_fcount, dmubins, nbins_v, nbins_mu, maxBlockIndex
      );
      
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuDeviceSynchronize() );

      // Copy computed data to CPU
      CHK_ERR( gpuMemcpy(host_fcount.data(), dev_fcount, numberOfLocalCells*nbins_v*nbins_mu*sizeof(int), gpuMemcpyDeviceToHost) );

      // TODO: The following region includes significant divergence in forms of index based if else conditions and
      // data dependent while loops with unpredictable size and hence it's currently parallelized with CPU
      // possible GPU parallelization should be looked into

      #pragma omp parallel for
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         // Search limits for how many cells in mu-direction should be max evaluated when searching for a near neighbour?
         // Assuming some oversampling; changing these values may result in method breaking at very small plasma frame velocities.
         const int rlimit = nbins_mu-1;
         const int llimit = 0;

         for (int indv = 0; indv < nbins_v; indv++) {
            for(int indmu = 0; indmu < nbins_mu; indmu++) {
               if (indmu == 0) {
                  CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  = 0;
                  CELLMUSPACE(host_cRight,CellIdx,indv,indmu) = 1;
                  while( (CELLMUSPACE(host_fcount,CellIdx,indv,indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu)) == 0) && (indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu) < rlimit) )  { CELLMUSPACE(host_cRight,CellIdx,indv,indmu) += 1; }
                  if(    (CELLMUSPACE(host_fcount,CellIdx,indv,indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu)) == 0) && (indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu) == rlimit) ) { CELLMUSPACE(host_cRight,CellIdx,indv,indmu)  = 0; }
               } else if (indmu == nbins_mu-1) {
                  CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  = 1;
                  CELLMUSPACE(host_cRight,CellIdx,indv,indmu) = 0;
                  while( (CELLMUSPACE(host_fcount,CellIdx,indv,indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)) == 0) && (indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu) > llimit) )  { CELLMUSPACE(host_cLeft,CellIdx,indv,indmu) += 1; }
                  if(    (CELLMUSPACE(host_fcount,CellIdx,indv,indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)) == 0) && (indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu) == llimit) ) { CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  = 0; }
               } else {
                  CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  = 1;
                  CELLMUSPACE(host_cRight,CellIdx,indv,indmu) = 1;
                  while( (CELLMUSPACE(host_fcount,CellIdx,indv,indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu)) == 0) && (indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu) < rlimit) )  { CELLMUSPACE(host_cRight,CellIdx,indv,indmu) += 1; }
                  if(    (CELLMUSPACE(host_fcount,CellIdx,indv,indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu)) == 0) && (indmu + CELLMUSPACE(host_cRight,CellIdx,indv,indmu) == rlimit) ) { CELLMUSPACE(host_cRight,CellIdx,indv,indmu)  = 0; }
                  while( (CELLMUSPACE(host_fcount,CellIdx,indv,indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu) ) == 0) && (indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  > llimit) )           { CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  += 1; }
                  if(    (CELLMUSPACE(host_fcount,CellIdx,indv,indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu) ) == 0) && (indmu - CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)  == llimit) )          { CELLMUSPACE(host_cLeft,CellIdx,indv,indmu)   = 0; }
               }
            }
         }
      } // End spatial cell loop

      // Run the kernel
      totalThreadsPerBlock = 512;
      int maxThreadIndex = numberOfLocalCells*nbins_v*nbins_mu;
      blocksPerGrid = (maxThreadIndex+totalThreadsPerBlock-1)/totalThreadsPerBlock;
      dividefByCount_kernel<<<blocksPerGrid, totalThreadsPerBlock>>>(
         dev_smallCellIdxArray, dev_fmu, dev_fcount, nbins_v, nbins_mu, maxThreadIndex
      );
      
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuDeviceSynchronize() );

      CHK_ERR( gpuMemcpy(dev_cRight, host_cRight.data(), numberOfLocalCells*nbins_v*nbins_mu*sizeof(int), gpuMemcpyHostToDevice) );
      CHK_ERR( gpuMemcpy(dev_cLeft, host_cLeft.data(), numberOfLocalCells*nbins_v*nbins_mu*sizeof(int), gpuMemcpyHostToDevice) );
      
      // Run the kernel
      int lastBlockSize = nbins_v*nbins_mu-(blocksPerVelocityCell-1)*maxThreadsPerBlock;
      if(blocksPerVelocityCell == 1){
         totalThreadsPerBlock = nextPowerOfTwo(nbins_v*nbins_mu);
      }else{
         totalThreadsPerBlock = maxThreadsPerBlock;
      }
      blocksPerGrid = numberOfLocalCells*blocksPerVelocityCell;
      int sharedMemorySize = totalThreadsPerBlock * sizeof(Real);

      computeDerivativesCFLDdt_kernel<<<blocksPerGrid, totalThreadsPerBlock, sharedMemorySize>>>(
         dev_smallCellIdxArray, dev_dVbins, dev_cRight, dev_cLeft,
         dev_fmu, dev_nu0Values, dev_dfdt_mu, dev_cellIdxKeys,
         dev_potentialDdtValues, dev_sparsity, dmubins, epsilon,
         Parameters::PADCFL, nbins_v, nbins_mu, blocksPerVelocityCell, lastBlockSize
      );
      
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuDeviceSynchronize() );

      // Find minimum values with thrust

      int* dev_out_keys;
      Real* dev_out_values;
      CHK_ERR( cudaMalloc(&dev_out_keys, maxCellIndex * sizeof(int)) );
      CHK_ERR( cudaMalloc(&dev_out_values, maxCellIndex * sizeof(Real)) );
      thrust::device_ptr<int> out_keys(dev_out_keys);
      thrust::device_ptr<Real> out_values(dev_out_values);
      thrust::device_ptr<int> in_keys(dev_cellIdxKeys);
      thrust::device_ptr<Real> in_values(dev_potentialDdtValues);

      // Run reduce_by_key
      auto new_end = thrust::reduce_by_key(
         in_keys, in_keys + maxCellIndex*blocksPerVelocityCell,        // keys
         in_values,                    // values
         out_keys, out_values,     // output
         thrust::equal_to<int>(),              // binary predicate for keys
         thrust::minimum<Real>()              // reduction operator
      );
      
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuDeviceSynchronize() );

      Real* dev_Ddt_values = thrust::raw_pointer_cast(out_values);
      cudaMemcpy(host_Ddt.data(), dev_Ddt_values, maxCellIndex * sizeof(Real), cudaMemcpyDeviceToHost);

      // Compute Ddt
      
      remappedCellIdx = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }
         const Real RemainT  = Parameters::dt - dtTotalDiff[CellIdx]; //Remaining time before reaching simulation time step
         if (host_Ddt[remappedCellIdx] > RemainT) {
            host_Ddt[remappedCellIdx] = RemainT;
         }
         dtTotalDiff[CellIdx] += host_Ddt[remappedCellIdx];
         remappedCellIdx++;
      } // End spatial cell loop

      CHK_ERR( gpuMemcpy(dev_Ddt, host_Ddt.data(), numberOfLocalCells*sizeof(Real), gpuMemcpyHostToDevice) );

      // Run the kernel
      threadsPerBlock = dim3(WID, WID, WID);

      totalThreadsPerBlock = WID3;
      blocksPerGrid = (maxGPUIndex+totalThreadsPerBlock-1)/totalThreadsPerBlock;

      computeNewCellValues_kernel<<<blocksPerGrid, threadsPerBlock>>>(
         dev_cellIdxArray, dev_remappedCellIdxArray, dev_parameters,
         dev_bulkVX, dev_bulkVY, dev_bulkVZ, dev_bValues,
         dev_dVbins, dev_cellValues, dev_dfdt_mu, dev_Ddt,
         dmubins, nbins_v, nbins_mu, maxBlockIndex
      );
      
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuDeviceSynchronize() );

      // Copy computed data to CPU
      CHK_ERR( gpuMemcpy(host_cellValues, dev_cellValues, maxGPUIndex*sizeof(Real), gpuMemcpyDeviceToHost) );

      totalBlockIndex = 0;
      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         if(spatialLoopComplete[CellIdx]){
            continue;
         }

         const auto CellID                  = LocalCells[CellIdx];
         SpatialCell& cell                  = *mpiGrid[CellID];

         vmesh::LocalID numberOfVelocityBlocks = cell.get_number_of_velocity_blocks(popID);
         
         std::memcpy(cell.get_data(popID), &host_cellValues[totalBlockIndex*WID3], numberOfVelocityBlocks * WID3 * sizeof(Real));

         totalBlockIndex += numberOfVelocityBlocks;
      } // End spatial cell loop
      
      // Free memory
      CHK_ERR( gpuFree(dev_cellIdxArray) );
      CHK_ERR( gpuFree(dev_smallCellIdxArray) );
      CHK_ERR( gpuFree(dev_remappedCellIdxArray) );
      CHK_ERR( gpuFree(dev_dVbins) );
      CHK_ERR( gpuFree(dev_bulkVX) );
      CHK_ERR( gpuFree(dev_bulkVY) );
      CHK_ERR( gpuFree(dev_bulkVZ) );
      CHK_ERR( gpuFree(dev_Ddt) );
      CHK_ERR( gpuFree(dev_parameters) );
      CHK_ERR( gpuFree(dev_cellValues) );
      CHK_ERR( gpuFree(dev_potentialDdtValues) );
      CHK_ERR( gpuFree(dev_fmu) );
      CHK_ERR( gpuFree(dev_dfdt_mu) );
      CHK_ERR( gpuFree(dev_fcount) );
      CHK_ERR( gpuFree(dev_cRight) );
      CHK_ERR( gpuFree(dev_cLeft) );
      CHK_ERR( gpuFree(dev_cellIdxKeys) );
      CHK_ERR( cudaFree(dev_out_keys) );
      CHK_ERR( cudaFree(dev_out_values) );
      delete[] host_cellValues;
      delete[] host_parameters;

      for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells
         allSpatialCellTimeLoopsComplete = true;
         if(dtTotalDiff[CellIdx] < Parameters::dt){
            allSpatialCellTimeLoopsComplete = false;
         }else{
            spatialLoopComplete[CellIdx] = true;
         }
      }

   } // End Time loop

   // Free memory
   CHK_ERR( gpuFree(dev_bValues) );
   CHK_ERR( gpuFree(dev_nu0Values) );
   CHK_ERR( gpuFree(dev_sparsity) );

   for (size_t CellIdx = 0; CellIdx < numberOfLocalCells; CellIdx++) { // Iterate over all spatial cells

      const auto CellID                  = LocalCells[CellIdx];
      SpatialCell& cell                  = *mpiGrid[CellID];

      // Ensure mass conservation
      if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
         //TODO: parallelize, tho this not usually used (I think) and probably not that expensive
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