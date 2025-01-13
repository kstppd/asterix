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
#include "compression_tools.h"
#include "zfp/array1.hpp"
#include <concepts>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <zfp.h>

#include "../object_wrapper.h"
#include "../spatial_cell_wrapper.hpp"
#include "../velocity_blocks.h"

// #define LUMI_FALLBACK
constexpr float ZFP_TOLL = 1e-12;

using namespace ASTERIX;

extern "C" {

size_t compress_and_reconstruct_vdf(std::size_t nVDFS, std::array<Real, 3>* vcoords, Realf* vspace, std::size_t size,
                                    std::array<Real, 3>* inference_vcoords, Realf* new_vspace,
                                    std::size_t inference_size, std::size_t max_epochs, std::size_t fourier_order,
                                    size_t* hidden_layers, size_t n_hidden_layers, Real sparsity, Real tol,
                                    Real* weights, std::size_t weight_size, bool use_input_weights,
                                    uint32_t downsampling_factor, float& error, int& status);

size_t compress_vdf_union(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr, std::size_t size,
                          std::size_t max_epochs, std::size_t fourier_order, size_t* hidden_layers_ptr,
                          size_t n_hidden_layers, Real sparsity, Real tol, Real* weights_ptr, std::size_t weight_size,
                          bool use_input_weights, uint32_t downsampling_factor, float& error, int& status);

void uncompress_vdf_union(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr, std::size_t size,
                          std::size_t fourier_order, size_t* hidden_layers_ptr, size_t n_hidden_layers,
                          Real* weights_ptr, std::size_t weight_size, bool use_input_weights);
}

auto compress_vdfs_fourier_mlp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                               size_t number_of_spatial_cells, bool update_weights, uint32_t downsampling_factor)
    -> float;

auto compress_vdfs_fourier_mlp_clustered(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                         size_t number_of_spatial_cells, bool update_weights,
                                         uint32_t downsampling_factor) -> float;

auto compress_vdfs_zfp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, size_t number_of_spatial_cells)
    -> float;

auto compress_vdfs_octree(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, size_t number_of_spatial_cells)
    -> float;

auto compress(float* array, size_t arraySize, size_t& compressedSize) -> std::vector<char>;

auto compress(double* array, size_t arraySize, size_t& compressedSize) -> std::vector<char>;

auto decompressArrayDouble(char* compressedData, size_t compressedSize, size_t arraySize) -> std::vector<double>;

auto decompressArrayFloat(char* compressedData, size_t compressedSize, size_t arraySize) -> std::vector<float>;

// Main driver, look at header file  for documentation
void ASTERIX::compress_vdfs(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                            size_t number_of_spatial_cells, P::ASTERIX_COMPRESSION_METHODS method, bool update_weights,
                            uint32_t downsampling_factor /*=1*/) {

   const auto& local_cells = getLocalCells();
#pragma omp parallel for
   for (auto& cid : local_cells) {
      // std::string fname = "vdf_" + std::to_string(cid) + "_pre.bin";
      // dump_vdf_to_binary_file(fname.c_str(), cid, mpiGrid);
   }

   if (downsampling_factor < 1) {
      throw std::runtime_error("Requested downsampling factor in VDF compression makes no sense!");
   }

   float local_compression_ratio = 0.0;
   switch (method) {
   case P::ASTERIX_COMPRESSION_METHODS::MLP:
      local_compression_ratio =
          compress_vdfs_fourier_mlp(mpiGrid, number_of_spatial_cells, update_weights, downsampling_factor);
      break;
   case P::ASTERIX_COMPRESSION_METHODS::MLP_MULTI:
      local_compression_ratio =
          compress_vdfs_fourier_mlp_clustered(mpiGrid, number_of_spatial_cells, update_weights, downsampling_factor);
      // local_compression_ratio=compress_vdfs_fourier_mlp(mpiGrid, number_of_spatial_cells, update_weights,
      // downsampling_factor);
      break;
   case P::ASTERIX_COMPRESSION_METHODS::ZFP:
      local_compression_ratio = compress_vdfs_zfp(mpiGrid, number_of_spatial_cells);
      break;
   case P::ASTERIX_COMPRESSION_METHODS::OCTREE:
      local_compression_ratio = compress_vdfs_octree(mpiGrid, number_of_spatial_cells);
      break;
   default:
      throw std::runtime_error("This is bad!. Improper Asterix method detected!");
      break;
   };

   // Reduce global compression ratio
   int myRank;
   int mpiProcs;
   float global_compression_ratio = 0.0;
   MPI_Comm_size(MPI_COMM_WORLD, &mpiProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&local_compression_ratio, &global_compression_ratio, 1, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);

   if (myRank == MASTER_RANK) {
      logFile << "(VDF COMPRESSION INFO): Compression Ratio = "
              << global_compression_ratio / static_cast<float>(mpiProcs) << std::endl;
   }

#pragma omp parallel for
   for (auto& cid : local_cells) {
      // std::string fname = "vdf_" + std::to_string(cid) + "_post.bin";
      // dump_vdf_to_binary_file(fname.c_str(), cid, mpiGrid);
   }
}

float compress_vdfs_fourier_mlp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                size_t number_of_spatial_cells, bool update_weights, uint32_t downsampling_factor) {

   float local_compression_achieved = 0.0;
   std::size_t total_samples = 0;
   const std::size_t max_span_size = P::max_vdfs_per_nn;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {

      Real sparse = getObjectWrapper().particleSpecies[popID].sparseMinValue;
      const std::vector<CellID>& local_cells = getLocalCells();
#pragma omp parallel for reduction(+ : local_compression_achieved)
      for (std::size_t sample = 0; sample < local_cells.size(); sample += max_span_size) {

#pragma omp atomic
         total_samples++;

         // Extract this span of VDFs as a union
         std::size_t span_size = std::min(max_span_size, local_cells.size() - sample);
         const std::span<const CellID> span(local_cells.data() + sample, span_size);
         VDFUnion vdf_union = extract_union_pop_vdfs_from_cids(span, popID, mpiGrid, true);

         // Min Max normalize Vspace Coords
         auto normalize_vspace_coords = [](VDFUnion& some_vdf_union) {
            std::ranges::for_each(some_vdf_union.vcoords_union, [&some_vdf_union](std::array<Real, 3>& x) {
               x[0] = 2.0 * ((x[0] - some_vdf_union.v_limits[0]) /
                             (some_vdf_union.v_limits[3] - some_vdf_union.v_limits[0])) -
                      1.0;
               x[1] = 2.0 * ((x[1] - some_vdf_union.v_limits[1]) /
                             (some_vdf_union.v_limits[4] - some_vdf_union.v_limits[1])) -
                      1.0;
               x[2] = 2.0 * ((x[2] - some_vdf_union.v_limits[2]) /
                             (some_vdf_union.v_limits[5] - some_vdf_union.v_limits[2])) -
                      1.0;
            });
         };
         normalize_vspace_coords(vdf_union);
         vdf_union.scale(sparse);
         vdf_union.normalize_union();

         // TODO: fix this
         static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

         // (2) Do the compression for this VDF
         float error = std::numeric_limits<float>::max();
         int status = 0;
         
         // Allocate spaced for weights
         auto network_size = calculate_total_size_bytes<double>(P::mlp_arch, P::mlp_fourier_order, vdf_union.cids.size());
         vdf_union.network_weights = (double*)malloc(network_size);
         vdf_union.n_weights=network_size/sizeof(double);

         std::size_t nn_mem_footprint_bytes = compress_vdf_union(
             span.size(), vdf_union.vcoords_union.data(), vdf_union.vspace_union.data(), vdf_union.vcoords_union.size(),
             P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(), P::mlp_arch.size(), sparse, P::mlp_tollerance,
             vdf_union.network_weights, network_size, false, downsampling_factor, error, status);

         assert(network_size==nn_mem_footprint_bytes && "Mismatch betweeen estimated and actual network size!!!");

         uncompress_vdf_union(span.size(), vdf_union.vcoords_union.data(), vdf_union.vspace_union.data(),
                              vdf_union.vcoords_union.size(), P::mlp_fourier_order, P::mlp_arch.data(),
                              P::mlp_arch.size(), vdf_union.network_weights, network_size, true);

         local_compression_achieved += vdf_union.size_in_bytes / static_cast<float>(nn_mem_footprint_bytes);
         vdf_union.unormalize_union();
         vdf_union.unscale(sparse);
         vdf_union.sparsify(sparse);
         free(vdf_union.network_weights);

         // (3) Overwrite the VDF of this cell
         overwrite_cellids_vdfs(span, popID, mpiGrid, vdf_union.vcoords_union, vdf_union.vspace_union, vdf_union.map);
      }
   } // loop over all populations
   return local_compression_achieved / static_cast<float>(total_samples);
}

std::vector<std::vector<std::pair<CellID, Real>>>
clusterVDFs(const std::vector<CellID>& local_cells, const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
            uint popID) {
   std::vector<Real> non_maxwellianity(local_cells.size(), 0.0);
   std::transform(local_cells.begin(), local_cells.end(), non_maxwellianity.begin(),
                  [&](const auto& cid) { return get_Non_MaxWellianity(mpiGrid[cid], popID); });
   std::vector<std::pair<CellID, Real>> sorted_vdf(local_cells.size());
   for (std::size_t i = 0; i < local_cells.size(); ++i) {
      sorted_vdf[i] = std::pair<CellID, Real>{local_cells[i], non_maxwellianity[i]};
   }
   std::sort(sorted_vdf.begin(), sorted_vdf.end(),
             [=](std::pair<CellID, Real>& a, std::pair<CellID, Real>& b) { return a.second < b.second; });
   std::vector<std::vector<std::pair<CellID, Real>>> clusters;
   std::vector<std::pair<CellID, Real>> current_cluster;
   for (const auto& pair : sorted_vdf) {
      if (current_cluster.empty()) {
         current_cluster.push_back(pair);
      } else {
         Real last_value = current_cluster.back().second;
         Real margin = 0.2f * std::max(last_value, pair.second);
         if (std::fabs(last_value - pair.second) <= margin) {
            current_cluster.push_back(pair);
         } else {
            clusters.push_back(current_cluster);
            current_cluster.clear();
            current_cluster.push_back(pair);
         }
      }
   }
   if (!current_cluster.empty()) {
      clusters.push_back(current_cluster);
   }
   return clusters;
}

float compress_vdfs_fourier_mlp_clustered(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                          size_t number_of_spatial_cells, bool update_weights,
                                          uint32_t downsampling_factor) {
   float local_compression_achieved = 0.0;
   std::size_t total_samples = 0;
   const std::size_t max_span_size = P::max_vdfs_per_nn;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
      Real sparse = getObjectWrapper().particleSpecies[popID].sparseMinValue;
      const std::vector<CellID>& local_cells = getLocalCells();
      const auto clusters = clusterVDFs(local_cells, mpiGrid, popID);
      std::cout << "Generated " << clusters.size() << " clusters" << std::endl;

#pragma omp parallel for reduction(+ : local_compression_achieved)
      for (const auto& cluster : clusters) {

#pragma omp atomic
         total_samples++;

         std::vector<CellID> cids(cluster.size());
         std::transform(cluster.begin(), cluster.end(), cids.begin(), [](const auto& pair) { return pair.first; });

         // Extract this span of VDFs as a union
         const std::span<const CellID> span(cids.data(), cids.size());
         VDFUnion vdf_union = extract_union_pop_vdfs_from_cids(span, popID, mpiGrid, true);

         // Min Max normalize Vspace Coords
         auto normalize_vspace_coords = [](VDFUnion& some_vdf_union) {
            std::ranges::for_each(some_vdf_union.vcoords_union, [&some_vdf_union](std::array<Real, 3>& x) {
               x[0] = 2.0 * ((x[0] - some_vdf_union.v_limits[0]) /
                             (some_vdf_union.v_limits[3] - some_vdf_union.v_limits[0])) -
                      1.0;
               x[1] = 2.0 * ((x[1] - some_vdf_union.v_limits[1]) /
                             (some_vdf_union.v_limits[4] - some_vdf_union.v_limits[1])) -
                      1.0;
               x[2] = 2.0 * ((x[2] - some_vdf_union.v_limits[2]) /
                             (some_vdf_union.v_limits[5] - some_vdf_union.v_limits[2])) -
                      1.0;
            });
         };
         normalize_vspace_coords(vdf_union);
         vdf_union.scale(sparse);
         vdf_union.normalize_union();

         // TODO: fix this
         static_assert(sizeof(Real) == 8 and sizeof(Realf) == 4);

         // (2) Do the compression for this VDF
         float error = std::numeric_limits<float>::max();
         int status = 0;

         // Allocate spaced for weights
         auto network_size = calculate_total_size_bytes<double>(P::mlp_arch, P::mlp_fourier_order, vdf_union.cids.size());
         vdf_union.network_weights = (double*)malloc(network_size);
         vdf_union.n_weights=network_size/sizeof(double);

         std::size_t nn_mem_footprint_bytes = compress_vdf_union(
             span.size(), vdf_union.vcoords_union.data(), vdf_union.vspace_union.data(), vdf_union.vcoords_union.size(),
             P::mlp_max_epochs, P::mlp_fourier_order, P::mlp_arch.data(), P::mlp_arch.size(), sparse, P::mlp_tollerance,
             vdf_union.network_weights, network_size, false, downsampling_factor, error, status);

         assert(network_size==nn_mem_footprint_bytes && "Mismatch betweeen estimated and actual network size!!!");

         std::vector<unsigned char> bytes(vdf_union.total_serialized_size_bytes());
         // vdf_union.serialize_into(&bytes[0]);
         // new_union.deserialize_from(&bytes[0]);
         
         uncompress_vdf_union(span.size(), vdf_union.vcoords_union.data(), vdf_union.vspace_union.data(),
                              vdf_union.vcoords_union.size(), P::mlp_fourier_order, P::mlp_arch.data(),
                              P::mlp_arch.size(), vdf_union.network_weights, network_size, true);

         free(vdf_union.network_weights);
         local_compression_achieved += vdf_union.size_in_bytes / static_cast<float>(nn_mem_footprint_bytes);

         vdf_union.unormalize_union();
         vdf_union.unscale(sparse);
         vdf_union.sparsify(sparse);

         // (3) Overwrite the VDF of this cell
         overwrite_cellids_vdfs(span, popID, mpiGrid, vdf_union.vcoords_union, vdf_union.vspace_union, vdf_union.map);
      }
   } // loop over all populations
   return local_compression_achieved / static_cast<float>(total_samples);
}

// Compresses and reconstucts VDFs using ZFP
float compress_vdfs_zfp(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, size_t number_of_spatial_cells) {
   float local_compression_achieved = 0.0;
   std::size_t total_samples = 0;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
      // Vlasiator boilerplate
      const auto& local_cells = getLocalCells();
#pragma omp parallel for reduction(+ : local_compression_achieved)
      for (auto& cid : local_cells) { // loop over spatial cells
         SpatialCell* sc = mpiGrid[cid];
         assert(sc && "Invalid Pointer to Spatial Cell !");

#pragma omp atomic
         total_samples++;

         // (1) Extract and Collect the VDF of this cell
         UnorderedVDF vdf = extract_pop_vdf_from_spatial_cell(sc, popID);

         // (2) Do the compression for this VDF
         // Create spave for the reconstructed VDF
         size_t ss{0};
         std::vector<char> compressedState = compress(vdf.vdf_vals.data(), vdf.vdf_vals.size(), ss);
         std::vector<Realf> new_vdf = decompressArrayFloat(compressedState.data(), ss, vdf.vdf_vals.size());
         float ratio = static_cast<float>(vdf.vdf_vals.size() * sizeof(Realf)) / static_cast<float>(ss);
         local_compression_achieved += ratio;

         // (3) Overwrite the VDF of this cell
         overwrite_pop_spatial_cell_vdf(sc, popID, new_vdf);

      } // loop over all spatial cells
   }    // loop over all populations
   return local_compression_achieved / static_cast<float>(total_samples);
}

// Compresses and reconstucts VDFs using ZFP
float compress_vdfs_octree(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                           size_t number_of_spatial_cells) {
   int total_bytes = 0;
   int global_total_bytes = 0;
   float local_compression_achieved = 0.0;
   std::size_t total_samples = 0;
   for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
      // Vlasiator boilerplate
      const auto& local_cells = getLocalCells();
#pragma omp parallel for reduction(+ : total_bytes, local_compression_achieved)
      for (auto& cid : local_cells) { // loop over spatial cells
         SpatialCell* sc = mpiGrid[cid];
         assert(sc && "Invalid Pointer to Spatial Cell !");

         // (1) Extract and Collect the VDF of this cell
         OrderedVDF vdf = extract_pop_vdf_from_spatial_cell_ordered_min_bbox_zoomed(sc, popID, 1);

#pragma omp atomic
         total_samples++;

         // (2) Do the compression for this VDF
         /* float ratio = 0.0; */
         uint8_t* bytes = nullptr;
         std::size_t n_bytes;

         /* 0. cmake build system for tinyAI3
          * 1. 4x4x4_blocks -> dense -> bytes+n_bytes -> to_disk -> from disk -> bytes -> dense
          * 2. asterix_hack_3 + siren merge in vlasiator
          * 3. link vlasiator to tinyAI3 library instead of building lib.cu inside vlasiator
          *
          *  iowrite line 57: writeVelocityDistributionData
          *
          *  create dense vdf iterator from byte array and offsets
          * */

         constexpr std::size_t maxiter = 5000;
         compress_with_toctree_method(vdf.vdf_vals.data(), vdf.shape[0], vdf.shape[1], vdf.shape[2],
                                      P::octree_tolerance, &bytes, &n_bytes, maxiter);

         uncompress_with_toctree_method(vdf.vdf_vals.data(), vdf.shape[0], vdf.shape[1], vdf.shape[2], bytes, n_bytes);

         if (bytes != NULL) {
            free(bytes);
         }
         total_bytes += n_bytes;
         local_compression_achieved += vdf.sparse_vdf_bytes / static_cast<float>(n_bytes);

         // (3) Overwrite the VDF of this cell
         overwrite_pop_spatial_cell_vdf(sc, popID, vdf);

      } // loop over all spatial cells
   }    // loop over all populations
   return local_compression_achieved / static_cast<float>(total_samples);
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
