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
#include "zfp/array1.hpp"
#include <concepts>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <zfp.h>

#include "../object_wrapper.h"
#include "../spatial_cell_wrapper.hpp"
#include "../velocity_blocks.h"

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#endif
#ifdef __HIP__
#include <hip/hip_runtime_api.h>
#endif

constexpr float ZFP_TOLL = 1e-12;

extern "C" {
Real compress_and_reconstruct_vdf(std::array<Real, 3>* vcoords, Realf* vspace, std::size_t size,
                                  std::array<Real, 3>* inference_vcoords, Realf* new_vspace, std::size_t inference_size,
                                  std::size_t max_epochs, std::size_t fourier_order, size_t* hidden_layers,
                                  size_t n_hidden_layers, Real sparsity, Real tol, Real* weights,
                                  std::size_t weight_size, bool use_input_weights);

std::size_t probe_network_size(std::array<Real, 3>* vcoords, Realf* vspace, std::size_t size,
                               std::array<Real, 3>* inference_vcoords, Realf* new_vspace, std::size_t inference_size,
                               std::size_t max_epochs, std::size_t fourier_order, size_t* hidden_layers,
                               size_t n_hidden_layers, Real sparsity, Real tol);

Real compress_and_reconstruct_vdf_2(std::array<Real, 3>* vcoords, Realf* vspace, std::size_t size,
                                    std::array<Real, 3>* inference_vcoords, Realf* new_vspace,
                                    std::size_t inference_size, std::size_t max_epochs, std::size_t fourier_order,
                                    size_t* hidden_layers, size_t n_hidden_layers, Real sparsity, Real tol,
                                    Real* weights, std::size_t weight_size, bool use_input_weights,uint32_t downsampling_factor);

Real compress_and_reconstruct_vdf_2_multi(std::size_t nVDFS, std::array<Real, 3>* vcoords, Realf* vspace,
                                          std::size_t size, std::array<Real, 3>* inference_vcoords, Realf* new_vspace,
                                          std::size_t inference_size, std::size_t max_epochs, std::size_t fourier_order,
                                          size_t* hidden_layers, size_t n_hidden_layers, Real sparsity, Real tol,
                                          Real* weights, std::size_t weight_size, bool use_input_weights,uint32_t downsampling_factor);

std::size_t probe_network_size_2(std::array<Real, 3>* vcoords, Realf* vspace, std::size_t size,
                                 std::array<Real, 3>* inference_vcoords, Realf* new_vspace, std::size_t inference_size,
                                 std::size_t max_epochs, std::size_t fourier_order, size_t* hidden_layers,
                                 size_t n_hidden_layers, Real sparsity, Real tol);
}

// These tools  are fwd declared here and implemented at the end of the file for
// better clarity. They are not for external usage and as such they do not go
// into the header file

struct VCoords {
   Real vx, vy, vz;
};

auto overwrite_pop_spatial_cell_vdf(SpatialCell* sc, uint popID, const std::vector<Realf>& new_vspace) -> void;

auto extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID, std::vector<std::array<Real, 3>>& vcoords,
                                       std::vector<Realf>& vspace) -> std::array<Real, 6>;

auto extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID, std::vector<Realf>& vspace) -> void;

auto extract_union_pop_vdfs_from_cids(const std::vector<CellID>& cids, uint popID,
                                      const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                      std::vector<std::array<Real, 3>>& vcoords, std::vector<Realf>& vspace)
    -> std::tuple<std::array<Real, 6>, std::unordered_map<vmesh::LocalID, std::size_t>>;

auto compress_vdfs_fourier_mlp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                               size_t number_of_spatial_cells, bool update_weightsu,uint32_t downsampling_factor) -> void;

auto compress_vdfs_fourier_mlp_multi(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                     size_t number_of_spatial_cells, bool update_weights,uint32_t downsampling_factor) -> void;

auto compress_vdfs_zfp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, size_t number_of_spatial_cells)
    -> void;

auto compress(float* array, size_t arraySize, size_t& compressedSize) -> std::vector<char>;

auto compress(double* array, size_t arraySize, size_t& compressedSize) -> std::vector<char>;

auto decompressArrayDouble(char* compressedData, size_t compressedSize, size_t arraySize) -> std::vector<double>;

auto decompressArrayFloat(char* compressedData, size_t compressedSize, size_t arraySize) -> std::vector<float>;

auto extract_pop_vdf_from_spatial_cell_ordered_min_bbox_zoomed(SpatialCell* sc, uint popID, std::vector<Realf>& vspace,
                                                               int zoom) -> void;

constexpr auto isPow2(std::unsigned_integral auto val) -> bool { return (val & (val - 1)) == 0; };

// Main driver, look at header file  for documentation
void ASTERIX::compress_vdfs(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                            size_t number_of_spatial_cells, P::ASTERIX_COMPRESSION_METHODS method,
                            bool update_weights,uint32_t downsampling_factor/*=1*/) {
   
   if (downsampling_factor<1){
      throw std::runtime_error("Requested downsampling factor in VDF compression makes no sense!");
   }
   switch (method) {
   case P::ASTERIX_COMPRESSION_METHODS::MLP:
      compress_vdfs_fourier_mlp(mpiGrid, number_of_spatial_cells, update_weights,downsampling_factor);
      break;
   case P::ASTERIX_COMPRESSION_METHODS::MLP_MULTI:
      compress_vdfs_fourier_mlp_multi(mpiGrid, number_of_spatial_cells, update_weights,downsampling_factor);
      break;
   case P::ASTERIX_COMPRESSION_METHODS::ZFP:
      compress_vdfs_zfp(mpiGrid, number_of_spatial_cells);
      break;
   default:
      throw std::runtime_error("This is bad!. Improper Asterix method detected!");
      break;
   };
}

// Detail implementations
/*
   Here we do a 3 step process which compresses and reconstructs the VDFs using
   the Fourier MLP method in Asterix
   First we collect the VDF and vspace coords into buffers.
   Then we call the reconstruction method which compresses reconstructs and
   writes the reconstructed VDF in a separate buffer. Finally we overwrite the
   original vdf with the previous buffer. Then we wait for it to explode! NOTES:
   This is a thread safe operation but will OOM easily. All scaling operations
   should be handled in the compression code. The original VDF leaves Vlasiator
   untouched. The new VDF should be returned in a sane state.

   mpiGrid: Grid with all local spatial cells
   number_of_spatial_cells:
      Used to reduce the global comrpession achieved
   update_weights:
      If the flag is set to true the method will create a feedback loop where
   the weights of the MLP are stored and then re used for the next training
   session. This will? lead to faster convergence down the road. If the flag is
   set to false then every training session starts from randomized weights. I
   think this is what ML people call transfer learning (together with freezing
   and adding extra neuron which we do not do here).
*/
void compress_vdfs_fourier_mlp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                               size_t number_of_spatial_cells, bool update_weights,uint32_t downsampling_factor) {
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   int deviceCount = 0;

   float local_compression_achieved = 0.0;
   float global_compression_achieved = 0.0;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
      Real sparse = getObjectWrapper().particleSpecies[popID].sparseMinValue;
      // Vlasiator boilerplate
      const auto& local_cells = getLocalCells();
#pragma omp parallel for reduction(+ : local_compression_achieved)
      for (auto& cid : local_cells) { // loop over spatial cells
         SpatialCell* sc = mpiGrid[cid];
         assert(sc && "Invalid Pointer to Spatial Cell !");

         // (1) Extract and Collect the VDF of this cell
         std::vector<std::array<Real, 3>> vcoords;
         std::vector<Realf> vspace;
         auto vspace_extent = extract_pop_vdf_from_spatial_cell(sc, popID, vcoords, vspace);

         // Min Max normalize Vspace Coords
         auto normalize_vspace_coords = [&]() {
            std::ranges::for_each(vcoords, [vspace_extent](std::array<Real, 3>& x) {
               x[0] = (x[0] - vspace_extent[0]) / (vspace_extent[3] - vspace_extent[0]);
               x[1] = (x[1] - vspace_extent[1]) / (vspace_extent[4] - vspace_extent[1]);
               x[2] = (x[2] - vspace_extent[2]) / (vspace_extent[5] - vspace_extent[2]);
            });
         };
         normalize_vspace_coords();

         // TODO: fix this
         static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

         // (2) Do the compression for this VDF
         // Create spave for the reconstructed VDF
         std::vector<Realf> new_vspace(vspace.size(), Realf(0));
         bool use_input_weights = update_weights;
         if (sc->fmlp_weights.size() == 0 && use_input_weights) {
            // This is lazilly done. The first time that we have no weights the MLP
            // is overwritten. Subsequent calls use the weights and update them at
            // the end
            size_t sz = probe_network_size(vcoords.data(), vspace.data(), vspace.size(), vcoords.data(), vspace.data(),
                                           vspace.size(), P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(),
                                           P::mlp_arch.size(), sparse, P::mlp_tollerance);
            sc->fmlp_weights.resize(sz / sizeof(Real));
            use_input_weights = false; // do not use this on the first pass;
         }

         float ratio = compress_and_reconstruct_vdf_2(vcoords.data(), vspace.data(), vspace.size(), vcoords.data(),
                                                      new_vspace.data(), vspace.size(), P::mlp_max_epochs,
                                                      P::mlp_fourier_order, P::mlp_arch.data(), P::mlp_arch.size(),
                                                      sparse, P::mlp_tollerance, nullptr, 0, false,downsampling_factor);
         local_compression_achieved += ratio;

         // (3) Overwrite the VDF of this cell
         overwrite_pop_spatial_cell_vdf(sc, popID, new_vspace);

      } // loop over all spatial cells
   }    // loop over all populations
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&local_compression_achieved, &global_compression_achieved, 1, MPI_FLOAT, MPI_SUM, MASTER_RANK,
              MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   float realized_compression = global_compression_achieved / (float)number_of_spatial_cells;
   if (myRank == MASTER_RANK) {
      logFile << "(INFO): Compression Ratio = " << realized_compression << std::endl;
   }
   return;
}

void overwrite_cellids_vdfs(const std::vector<CellID>& cids, uint popID,
                            dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                            const std::vector<std::array<Real, 3>>& vcoords, const std::vector<Realf>& vspace_union,
                            const std::unordered_map<vmesh::LocalID, std::size_t>& map_exists_id) {
   const std::size_t nrows = vcoords.size();
   const std::size_t ncols = cids.size();
   // This will be used further down for indexing into the vspace_union
   auto index_2d = [nrows, ncols](std::size_t row, std::size_t col) -> std::size_t { return row * ncols + col; };

   for (std::size_t cc = 0; cc < cids.size(); ++cc) {
      const auto& cid = cids[cc];
      SpatialCell* sc = mpiGrid[cid];
      vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
      const size_t total_blocks = blockContainer.size();
      Realf* data = blockContainer.getData();
      const Real* blockParams = sc->get_block_parameters(popID);
      for (std::size_t n = 0; n < total_blocks; ++n) {
         const auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
         const vmesh::GlobalID gid = sc->get_velocity_block_global_id(n, popID);
         const auto it = map_exists_id.find(gid);
         const bool exists = it != map_exists_id.end();
         assert(exists && "Someone has a buuuug!");
         const auto index = it->second;
         Realf* vdf_data = &data[n * WID3];
         size_t cnt = 0;
         for (uint k = 0; k < WID; ++k) {
            for (uint j = 0; j < WID; ++j) {
               for (uint i = 0; i < WID; ++i) {
                  const std::size_t index = it->second;
                  vdf_data[cellIndex(i, j, k)] = vspace_union[index_2d(index + cnt, cc)];
                  cnt++;
               }
            }
         }
      }
   }
   return;
}

void compress_vdfs_fourier_mlp_multi(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                     size_t number_of_spatial_cells, bool update_weights,uint32_t downsampling_factor) {
   int myRank;
   int mpiProcs;
   MPI_Comm_size(MPI_COMM_WORLD, &mpiProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   float local_compression_achieved = 0.0;
   float global_compression_achieved = 0.0;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {

      Real sparse = getObjectWrapper().particleSpecies[popID].sparseMinValue;
      const std::vector<CellID>& local_cells = getLocalCells();
      std::vector<std::array<Real, 3>> vcoords;
      std::vector<Realf> vspace;
      auto retval = extract_union_pop_vdfs_from_cids(local_cells, popID, mpiGrid, vcoords, vspace);
      auto vspace_extent = std::get<0>(retval);
      auto map_exists = std::get<1>(retval);

      // Min Max normalize Vspace Coords
      auto normalize_vspace_coords = [&]() {
         std::ranges::for_each(vcoords, [vspace_extent](std::array<Real, 3>& x) {
            x[0] = (x[0] - vspace_extent[0]) / (vspace_extent[3] - vspace_extent[0]);
            x[1] = (x[1] - vspace_extent[1]) / (vspace_extent[4] - vspace_extent[1]);
            x[2] = (x[2] - vspace_extent[2]) / (vspace_extent[5] - vspace_extent[2]);
         });
      };
      normalize_vspace_coords();

      // TODO: fix this
      static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

      // (2) Do the compression for this VDF
      // Create space for the reconstructed VDF
      std::vector<Realf> new_vspace(vspace.size(), Realf(0));

      float ratio = compress_and_reconstruct_vdf_2_multi(
          local_cells.size(), vcoords.data(), vspace.data(), vcoords.size(), vcoords.data(), new_vspace.data(),
          vcoords.size(), P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(), P::mlp_arch.size(), sparse,
          P::mlp_tollerance, nullptr, 0, false,downsampling_factor);
      local_compression_achieved += ratio;

      // (3) Overwrite the VDF of this cell
      overwrite_cellids_vdfs(local_cells, popID, mpiGrid, vcoords, new_vspace, map_exists);

   } // loop over all populations
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&local_compression_achieved, &global_compression_achieved, 1, MPI_FLOAT, MPI_SUM, MASTER_RANK,
              MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   float realized_compression = global_compression_achieved / (float)mpiProcs;
   if (myRank == MASTER_RANK) {
      logFile << "(INFO): Compression Ratio = " << realized_compression << std::endl;
   }
   return;
}

// Just probes the needed size to store the weights of the MLP. Kinda stupid
// interface but this is all we have now.
std::size_t ASTERIX::probe_network_size_in_bytes(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                                 size_t number_of_spatial_cells) {
   abort();
   std::size_t network_size = 0;
   uint popID = 0;
   const auto& local_cells = getLocalCells();
   auto cid = local_cells.front();
   SpatialCell* sc = mpiGrid[cid];
   assert(sc && "Invalid Pointer to Spatial Cell !");

   // (1) Extract and Collect the VDF of this cell
   std::vector<std::array<Real, 3>> vcoords;
   std::vector<Realf> vspace;
   auto vspace_extent = extract_pop_vdf_from_spatial_cell(sc, popID, vcoords, vspace);

   // TODO: fix this
   static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

   // (2) Probe network size
   std::vector<Realf> new_vspace(vspace.size(), Realf(0));
   network_size = probe_network_size(vcoords.data(), vspace.data(), vspace.size(), vcoords.data(), new_vspace.data(),
                                     new_vspace.size(), P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(),
                                     P::mlp_arch.size(), 0.0, P::mlp_tollerance);
   MPI_Barrier(MPI_COMM_WORLD);
   return network_size;
}

// Compresses and reconstucts VDFs using ZFP
void compress_vdfs_zfp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, size_t number_of_spatial_cells) {
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
         std::vector<Realf> vspace;
         extract_pop_vdf_from_spatial_cell(sc, popID, vspace);

         // (2) Do the compression for this VDF
         // Create spave for the reconstructed VDF
         size_t ss{0};
         std::vector<char> compressedState = compress(vspace.data(), vspace.size(), ss);
         std::vector<Realf> new_vspace = decompressArrayFloat(compressedState.data(), ss, vspace.size());
         float ratio = static_cast<float>(vspace.size() * sizeof(Realf)) / static_cast<float>(ss);
         local_compression_achieved += ratio;

         // (3) Overwrite the VDF of this cell
         overwrite_pop_spatial_cell_vdf(sc, popID, new_vspace);

      } // loop over all spatial cells
   }    // loop over all populations
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&local_compression_achieved, &global_compression_achieved, 1, MPI_FLOAT, MPI_SUM, MASTER_RANK,
              MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   float realized_compression = global_compression_achieved / (float)number_of_spatial_cells;
   if (myRank == MASTER_RANK) {
      logFile << "(INFO): Compression Ratio = " << realized_compression << std::endl;
   }
   return;
}

/*
Extracts VDF from spatial cell
std::vectors (vx_coord,vy_coord,vz_coord,vspace) coming in do **not** need to be
properly sized;
 */
std::array<Real, 6> extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID,
                                                      std::vector<std::array<Real, 3>>& vcoords,
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
   vcoords.resize(blockContainer.size() * WID3, {Real(0), Real(0), Real(0)});
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
               vcoords[cnt] = {vx, vy, vz};
               vspace[cnt] = vdf_val;
               cnt++;
            }
         }
      }
   } // over blocks
   return vlims;
}

std::tuple<std::array<Real, 6>, std::unordered_map<vmesh::LocalID, std::size_t>>
extract_union_pop_vdfs_from_cids(const std::vector<CellID>& cids, uint popID,
                                 const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                 std::vector<std::array<Real, 3>>& vcoords_union, std::vector<Realf>& vspace_union) {

   // Let's find out which of these cellids has the largest VDF
   std::size_t max_cid_block_size = 0;
   for (const auto& cid : cids) {
      SpatialCell* sc = mpiGrid[cid];
      const vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
      const size_t total_size = blockContainer.size();
      max_cid_block_size = std::max(total_size, max_cid_block_size);
   }
   const std::size_t nrows = max_cid_block_size * WID3;
   const std::size_t ncols = cids.size();

   // This will be used further down for indexing into the vspace_union
   auto index_2d = [nrows, ncols](std::size_t row, std::size_t col) -> std::size_t { return row * ncols + col; };

   // Resize to fit the union of vspace coords and vspace density
   vcoords_union.resize(max_cid_block_size * WID3, {Real(0), Real(0), Real(0)});
   vspace_union = std::vector<Realf>(max_cid_block_size * WID3 * cids.size(), Realf(0));

   // xmin,ymin,zmin,xmax,ymax,zmax;
   std::array<Real, 6> vlims{std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::max(),
                             std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::lowest(),
                             std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest()};

   std::unordered_map<vmesh::LocalID, std::size_t> map_exists_id;
   std::size_t last_row = 0;
   for (std::size_t cc = 0; cc < cids.size(); ++cc) {
      const auto& cid = cids[cc];
      SpatialCell* sc = mpiGrid[cid];
      vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
      const size_t total_blocks = blockContainer.size();
      Realf* data = blockContainer.getData();
      const Real* blockParams = sc->get_block_parameters(popID);
      for (std::size_t n = 0; n < total_blocks; ++n) {
         const auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
         const vmesh::GlobalID gid = sc->get_velocity_block_global_id(n, popID);
         const Realf* vdf_data = &data[n * WID3];
         std::size_t cnt = 0;
         const auto it = map_exists_id.find(gid);
         const bool block_exists = it != map_exists_id.end();
         for (uint k = 0; k < WID; ++k) {
            for (uint j = 0; j < WID; ++j) {
               for (uint i = 0; i < WID; ++i) {

                  const VCoords coords = {bp[BlockParams::VXCRD] + (i + 0.5) * bp[BlockParams::DVX],
                                          bp[BlockParams::VYCRD] + (j + 0.5) * bp[BlockParams::DVY],
                                          bp[BlockParams::VZCRD] + (k + 0.5) * bp[BlockParams::DVZ]};

                  vlims[0] = std::min(vlims[0], coords.vx);
                  vlims[1] = std::min(vlims[1], coords.vy);
                  vlims[2] = std::min(vlims[2], coords.vz);
                  vlims[3] = std::max(vlims[3], coords.vx);
                  vlims[4] = std::max(vlims[4], coords.vy);
                  vlims[5] = std::max(vlims[5], coords.vz);
                  const Realf vdf_val = vdf_data[cellIndex(i, j, k)];

                  const std::size_t index = (block_exists) ? (it->second + cnt) : (last_row + cnt);
                  if (!block_exists) {
                     vcoords_union[last_row + cnt] = {coords.vx, coords.vy, coords.vz};
                  }
                  vspace_union[index_2d(index, cc)] = vdf_val;
                  cnt++;
               }
            }
         }
         if (!block_exists) {
            map_exists_id[gid] = last_row;
            last_row += WID3;
         }
      }
   }
   return {vlims, map_exists_id};
}

/*
Extracts VDF from spatial cell. This overload returns only the vspave
std::vector (vspace) coming in does **not** need to be properly sized;
 */
void extract_pop_vdf_from_spatial_cell(SpatialCell* sc, uint popID, std::vector<Realf>& vspace) {
   assert(sc && "Invalid Pointer to Spatial Cell !");
   vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
   const size_t total_blocks = blockContainer.size();
   const Real* blockParams = sc->get_block_parameters(popID);
   Realf* data = blockContainer.getData();
   vspace.resize(blockContainer.size() * WID3, Realf(0));

   std::size_t cnt = 0;
   for (std::size_t n = 0; n < total_blocks; ++n) {
      auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
      const Realf* vdf_data = &data[n * WID3];
      for (uint k = 0; k < WID; ++k) {
         for (uint j = 0; j < WID; ++j) {
            for (uint i = 0; i < WID; ++i) {
               Realf vdf_val = vdf_data[cellIndex(i, j, k)];
               vspace[cnt] = vdf_val;
               cnt++;
            }
         }
      }
   } // over blocks
}

// Extracts VDF in a cartesian C ordered mesh in a minimum BBOX and with a zoom level used for upsampling/downsampling
void extract_pop_vdf_from_spatial_cell_ordered_min_bbox_zoomed(SpatialCell* sc, uint popID, std::vector<Realf>& vspace,
                                                               int zoom) {
   assert(sc && "Invalid Pointer to Spatial Cell !");
   vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = sc->get_velocity_blocks(popID);
   const size_t total_blocks = blockContainer.size();
   const Real* blockParams = sc->get_block_parameters(popID);

   // xmin,ymin,zmin,xmax,ymax,zmax;
   std::array<Real, 6> vlims{std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::max(),
                             std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::lowest(),
                             std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest()};

   // This pass is computing the active vmesh limits
   // Store dvx,dvy,dvz here
   const Real dvx = (blockParams + BlockParams::N_VELOCITY_BLOCK_PARAMS)[BlockParams::DVX];
   const Real dvy = (blockParams + BlockParams::N_VELOCITY_BLOCK_PARAMS)[BlockParams::DVY];
   const Real dvz = (blockParams + BlockParams::N_VELOCITY_BLOCK_PARAMS)[BlockParams::DVZ];
   for (std::size_t n = 0; n < total_blocks; ++n) {
      const auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
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
            }
         }
      }
   } // over blocks

   assert(isPow2(static_cast<size_t>(std::abs(zoom))));
   float ratio = (zoom > 0) ? static_cast<float>(std::abs(zoom)) : 1.0 / static_cast<float>(std::abs(zoom));
   assert(ratio > 0);

   const Real target_dvx = dvx * ratio;
   const Real target_dvy = dvy * ratio;
   const Real target_dvz = dvz * ratio;
   std::size_t nx = std::ceil((vlims[3] - vlims[0]) / target_dvx);
   std::size_t ny = std::ceil((vlims[4] - vlims[1]) / target_dvy);
   std::size_t nz = std::ceil((vlims[5] - vlims[2]) / target_dvz);
   printf("VDF min box is %zu , %zu %zu \n ", nx, ny, nz);

   Realf* data = blockContainer.getData();
   vspace.resize(nx * ny * nz, Realf(0));
   for (std::size_t n = 0; n < total_blocks; ++n) {
      const auto bp = blockParams + n * BlockParams::N_VELOCITY_BLOCK_PARAMS;
      const Realf* vdf_data = &data[n * WID3];
      for (uint k = 0; k < WID; ++k) {
         for (uint j = 0; j < WID; ++j) {
            for (uint i = 0; i < WID; ++i) {
               const Real vx = bp[BlockParams::VXCRD] + (i + 0.5) * bp[BlockParams::DVX];
               const Real vy = bp[BlockParams::VYCRD] + (j + 0.5) * bp[BlockParams::DVY];
               const Real vz = bp[BlockParams::VZCRD] + (k + 0.5) * bp[BlockParams::DVZ];
               const size_t bbox_i = std::min(static_cast<size_t>(std::floor((vx - vlims[0]) / target_dvx)), nx - 1);
               const size_t bbox_j = std::min(static_cast<size_t>(std::floor((vy - vlims[1]) / target_dvy)), ny - 1);
               const size_t bbox_k = std::min(static_cast<size_t>(std::floor((vz - vlims[2]) / target_dvz)), nz - 1);

               // Averaging
               if (ratio >= 1.0) {
                  const size_t index = bbox_i * (ny * nz) + bbox_j * nz + bbox_k;
                  vspace.at(index) += vdf_data[cellIndex(i, j, k)] / ratio;
               } else {
                  // Same value in all bins
                  int max_off = 1 / ratio;
                  for (int off_z = 0; off_z <= max_off; off_z++) {
                     for (int off_y = 0; off_y <= max_off; off_y++) {
                        for (int off_x = 0; off_x <= max_off; off_x++) {
                           const size_t index = (bbox_i + off_x) * (ny * nz) + (bbox_j + off_y) * nz + (bbox_k + off_z);
                           if (index < vspace.size()) {
                              vspace.at(index) = vdf_data[cellIndex(i, j, k)];
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   } // over blocks
}

// Simply overwrites the VDF of this population for the give spatial cell with a
// new vspace
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

std::vector<char> compress(float* array, size_t arraySize, size_t& compressedSize) {
   // Allocate memory for compressed data

   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_field* field = zfp_field_1d(array, zfp_type_float, arraySize);
   size_t maxSize = zfp_stream_maximum_size(zfp, field);
   std::vector<char> compressedData(maxSize);

   // Initialize ZFP compression
   zfp_stream_set_accuracy(zfp, ZFP_TOLL);
   bitstream* stream = stream_open(compressedData.data(), compressedSize);
   zfp_stream_set_bit_stream(zfp, stream);
   zfp_stream_rewind(zfp);

   // Compress the array
   compressedSize = zfp_compress(zfp, field);
   compressedData.erase(compressedData.begin() + compressedSize, compressedData.end());
   zfp_field_free(field);
   zfp_stream_close(zfp);
   stream_close(stream);
   return compressedData;
}

// Function to decompress a compressed array of floats using ZFP
std::vector<float> decompressArrayFloat(char* compressedData, size_t compressedSize, size_t arraySize) {
   // Allocate memory for decompresseFloatd data
   std::vector<float> decompressedArray(arraySize);

   // Initialize ZFP decompression
   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_stream_set_accuracy(zfp, ZFP_TOLL);
   bitstream* stream_decompress = stream_open(compressedData, compressedSize);
   zfp_stream_set_bit_stream(zfp, stream_decompress);
   zfp_stream_rewind(zfp);

   // Decompress the array
   zfp_field* field_decompress = zfp_field_1d(decompressedArray.data(), zfp_type_float, decompressedArray.size());
   size_t retval = zfp_decompress(zfp, field_decompress);
   (void)retval;
   zfp_field_free(field_decompress);
   zfp_stream_close(zfp);
   stream_close(stream_decompress);

   return decompressedArray;
}

// Function to compress a 1D array of doubles using ZFP
std::vector<char> compress(double* array, size_t arraySize, size_t& compressedSize) {
   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_field* field = zfp_field_1d(array, zfp_type_double, arraySize);
   size_t maxSize = zfp_stream_maximum_size(zfp, field);
   std::vector<char> compressedData(maxSize);

   zfp_stream_set_accuracy(zfp, ZFP_TOLL);
   bitstream* stream = stream_open(compressedData.data(), compressedSize);
   zfp_stream_set_bit_stream(zfp, stream);
   zfp_stream_rewind(zfp);

   compressedSize = zfp_compress(zfp, field);
   compressedData.erase(compressedData.begin() + compressedSize, compressedData.end());
   zfp_field_free(field);
   zfp_stream_close(zfp);
   stream_close(stream);
   return compressedData;
}

// Function to decompress a compressed array of doubles using ZFP
std::vector<double> decompressArrayDouble(char* compressedData, size_t compressedSize, size_t arraySize) {
   // Allocate memory for decompressed data
   std::vector<double> decompressedArray(arraySize);

   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_stream_set_accuracy(zfp, ZFP_TOLL);
   bitstream* stream_decompress = stream_open(compressedData, compressedSize);
   zfp_stream_set_bit_stream(zfp, stream_decompress);
   zfp_stream_rewind(zfp);

   zfp_field* field_decompress = zfp_field_1d(decompressedArray.data(), zfp_type_double, decompressedArray.size());
   size_t retval = zfp_decompress(zfp, field_decompress);
   (void)retval;
   zfp_field_free(field_decompress);
   zfp_stream_close(zfp);
   stream_close(stream_decompress);
   return decompressedArray;
}
