/*
 * This file is part of Vlasiator.
 * Copyright 2010-2024 Finnish Meteorological Institute
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

#include "compression.h"
#include <vector>

extern "C" Real compress_and_reconstruct_vdf(Real* vx, Real* vy, Real* vz, Realf* vspace, std::size_t size,
                                             Realf* new_vspace, std::size_t max_epochs, std::size_t fourier_order,
                                             size_t* hidden_layers, size_t n_hidden_layers, Real sparsity, Real tol);

// These tools  are fwd declared here and implemented at the end of the file for better clarity
auto overwrite_pop_spatial_cell_vdf(SpatialCell* sc, uint popID, const std::vector<Realf>& new_vspace) -> void;

auto extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID, std::vector<Real>& vx_coord,
                                       std::vector<Real>& vy_coord, std::vector<Real>& vz_coord,
                                       std::vector<Realf>& vspace) -> std::array<Real, 6>;

/*
   Here we do a 3 step process which compresses and reconstructs the VDFs using
   the Fourier MLP method in Asterix
   First we collect the VDF and vspace coords into buffers.
   Then we call the reconstruction method which compresses reconstructs and writes the reconstructed VDF in a separate
   buffer. Finally we overwrite the original vdf with the previous buffer. Then we wait for it to explode! NOTES: This
   is a thread safe operation but will OOM easily. All scaling operations should be handled in the compression code. The
   original VDF leaves Vlasiator untouched. The new VDF should be returned in a sane state.
*/
void ASTERIX::compress_vdfs_fourier_mlp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                        size_t number_of_spatial_cells) {
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   float local_compression_achieved = 0.0;
   float global_compression_achieved = 0.0;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
      // Vlasiator boilerplate
      const auto& local_cells = getLocalCells();
#pragma omp parallel for reduction(+ : local_compression_achieved)
      for (auto& cid : local_cells) { // loop over spatial cells
         SpatialCell* sc = mpiGrid[cid];
         assert(sc && "Invalid Pointer to Spatial Cell !");

         // (1) Extract and Collect the VDF of this cell
         std::vector<Real> vx_coord, vy_coord, vz_coord;
         std::vector<Realf> vspace;
         auto vspace_extent = extract_pop_vdf_from_spatial_cell(sc, popID, vx_coord, vy_coord, vz_coord, vspace);
         // Min Max normaloze Vspace Coords
         for (std::size_t i = 0; i < vx_coord.size(); ++i) {
            vx_coord[i] = (vx_coord[i] - vspace_extent[0]) / (vspace_extent[3] - vspace_extent[0]);
            vy_coord[i] = (vy_coord[i] - vspace_extent[1]) / (vspace_extent[4] - vspace_extent[1]);
            vz_coord[i] = (vz_coord[i] - vspace_extent[2]) / (vspace_extent[5] - vspace_extent[2]);
         }

         // TODO: fix this
         static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

         // (2) Do the compression for this VDF
         // Create spave for the reconstructed VDF
         std::vector<Realf> new_vspace(vspace.size(), Realf(0));
         float ratio = compress_and_reconstruct_vdf(
             vx_coord.data(), vy_coord.data(), vz_coord.data(), vspace.data(), vspace.size(), new_vspace.data(),
             P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(), P::mlp_arch.size(), 1e-16, P::mlp_tollerance);
         local_compression_achieved += ratio;

         // (3) Overwrite the VDF of this cell
         overwrite_pop_spatial_cell_vdf(sc, popID, new_vspace);

      } // loop over all spatial cells
   }    // loop over all populations
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&local_compression_achieved, &global_compression_achieved, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   float realized_compression = global_compression_achieved / (float)number_of_spatial_cells;
   if (myRank == MASTER_RANK) {
      logFile << "(INFO): Compression Ratio = " << realized_compression << std::endl;
   }
   return;
}

/*
Extracts VDF from spatial cell
std::vectors (vx_coord,vy_coord,vz_coord,vspace) coming in do **not** need to be properly sized;
 */
std::array<Real, 6> extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID, std::vector<Real>& vx_coord,
                                                      std::vector<Real>& vy_coord, std::vector<Real>& vz_coord,
                                                      std::vector<Realf>& vspace) {

   assert(sc && "Invalid Pointer to Spatial Cell !");
   vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
   const size_t total_blocks = blockContainer.size();
   const Real* max_v_lims = sc->get_velocity_grid_max_limits(popID);
   const Real* min_v_lims = sc->get_velocity_grid_min_limits(popID);
   const Real* blockParams = sc->get_block_parameters(popID);
   Realf* data = blockContainer.getData();
   assert(max_v_lims && "Invalid Pointre to max_v_limits");
   assert(min_v_lims && "Invalid Pointre to min_v_limits");
   assert(data && "Invalid Pointre block container data");
   vx_coord.resize(blockContainer.size() * WID3, Real(0));
   vy_coord.resize(blockContainer.size() * WID3, Real(0));
   vz_coord.resize(blockContainer.size() * WID3, Real(0));
   vspace.resize(blockContainer.size() * WID3, Realf(0));

   // xmin,ymin,zmin,xmax,ymax,zmax;
   std::array<Real, 6> vlims{std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::max(),
                             std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::lowest(),
                             std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest()};

   std::size_t cnt = 0;
   for (std::size_t n = 0; n < total_blocks; ++n) {
      auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
      const Realf* vdf_data = &data[n * WID3];
      for (uint k = 0; k < WID; ++k) {
         for (uint j = 0; j < WID; ++j) {
            for (uint i = 0; i < WID; ++i) {
               const Real vx = bp[BlockParams::VXCRD] + (i + 0.5) * bp[BlockParams::DVX];
               const Real vy = bp[BlockParams::VYCRD] + (j + 0.5) * bp[BlockParams::DVY];
               const Real vz = bp[BlockParams::VZCRD] + (k + 0.5) * bp[BlockParams::DVZ];
               vlims[0] = std::min(vlims[0], vx);
               vlims[1] = std::min(vlims[1], vy);
               vlims[2] = std::min(vlims[2], vz);
               vlims[3] = std::max(vlims[3], vx);
               vlims[4] = std::max(vlims[4], vy);
               vlims[5] = std::max(vlims[5], vz);
               Realf vdf_val = vdf_data[cellIndex(i, j, k)];
               vx_coord[cnt] = vx;
               vy_coord[cnt] = vy;
               vz_coord[cnt] = vz;
               vspace[cnt] = vdf_val;
               cnt++;
            }
         }
      }
   } // over blocks
   return vlims;
}

// Simply overwrites the VDF of this population for the give spatial cell with a new vspace
void overwrite_pop_spatial_cell_vdf(SpatialCell* sc, uint popID, const std::vector<Realf>& new_vspace) {

   assert(sc && "Invalid Pointer to Spatial Cell !");
   vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
   const size_t total_blocks = blockContainer.size();
   const Real* max_v_lims = sc->get_velocity_grid_max_limits(popID);
   const Real* min_v_lims = sc->get_velocity_grid_min_limits(popID);
   const Real* blockParams = sc->get_block_parameters(popID);
   Realf* data = blockContainer.getData();
   assert(max_v_lims && "Invalid Pointre to max_v_limits");
   assert(min_v_lims && "Invalid Pointre to min_v_limits");
   assert(data && "Invalid Pointre block container data");

   std::size_t cnt = 0;
   for (std::size_t n = 0; n < total_blocks; ++n) {
      auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
      Realf* vdf_data = &data[n * WID3];
      for (uint k = 0; k < WID; ++k) {
         for (uint j = 0; j < WID; ++j) {
            for (uint i = 0; i < WID; ++i) {
               vdf_data[cellIndex(i, j, k)] = new_vspace[cnt];
               cnt++;
            }
         }
      }
   } // over blocks
   return;
}
