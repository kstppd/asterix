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
   
   std::cout << "Running\n";
}