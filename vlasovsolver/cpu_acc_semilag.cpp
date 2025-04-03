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

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "cpu_acc_semilag.hpp"
#include "cpu_acc_transform.hpp"
#include "cpu_acc_intersections.hpp"
#include "cpu_acc_map.hpp"

using namespace std;
using namespace spatial_cell;
using namespace Eigen;

/*!
  Propagates the distribution function in velocity space of given real
  space cell.

  Based on SLICE-3D algorithm: Zerroukat, M., and T. Allen. "A
  three‐dimensional monotone and conservative semi‐Lagrangian scheme
  (SLICE‐3D) for transport problems." Quarterly Journal of the Royal
  Meteorological Society 138.667 (2012): 1640-1651.

 * @param spatial_cell Spatial cell containing the accelerated population.
 * @param popID ID of the accelerated particle species.
 * @param vmesh Velocity mesh.
 * @param blockContainer Velocity block data container.
 * @param map_order Order in which vx,vy,vz mappings are performed. 
 * @param dt Time step of one subcycle.
*/

void cpu_accelerate_cell(SpatialCell* spatial_cell,
                         const uint popID,     
                         const uint map_order,
                         const Real& dt,
                         int intersections_id, // Phiprof Timer IDs
                         int mappings_id
   ) {

   compute_cell_intersections(spatial_cell, popID, map_order, dt, intersections_id);

   phiprof::Timer mappingsTimer {mappings_id};
   switch(map_order){
      case 0: {
         //Map order XYZ
         map_1d(spatial_cell, popID, spatial_cell->intersection_x,
                spatial_cell->intersection_x_di,spatial_cell->intersection_x_dj,spatial_cell->intersection_x_dk,0); // map along x
         map_1d(spatial_cell, popID, spatial_cell->intersection_y,
                spatial_cell->intersection_y_di,spatial_cell->intersection_y_dj,spatial_cell->intersection_y_dk,1); // map along y
         map_1d(spatial_cell, popID, spatial_cell->intersection_z,
                spatial_cell->intersection_z_di,spatial_cell->intersection_z_dj,spatial_cell->intersection_z_dk,2); // map along z
         break;
      }
      case 1: {
         //Map order YZX
         map_1d(spatial_cell, popID, spatial_cell->intersection_y,
                spatial_cell->intersection_y_di,spatial_cell->intersection_y_dj,spatial_cell->intersection_y_dk,1); // map along y
         map_1d(spatial_cell, popID, spatial_cell->intersection_z,
                spatial_cell->intersection_z_di,spatial_cell->intersection_z_dj,spatial_cell->intersection_z_dk,2); // map along z
         map_1d(spatial_cell, popID, spatial_cell->intersection_x,
                spatial_cell->intersection_x_di,spatial_cell->intersection_x_dj,spatial_cell->intersection_x_dk,0); // map along x
         break;
      }
      case 2: {
         //Map order Z X Y
         map_1d(spatial_cell, popID, spatial_cell->intersection_z,
                spatial_cell->intersection_z_di,spatial_cell->intersection_z_dj,spatial_cell->intersection_z_dk,2); // map along z
         map_1d(spatial_cell, popID, spatial_cell->intersection_x,
                spatial_cell->intersection_x_di,spatial_cell->intersection_x_dj,spatial_cell->intersection_x_dk,0); // map along x
         map_1d(spatial_cell, popID, spatial_cell->intersection_y,
                spatial_cell->intersection_y_di,spatial_cell->intersection_y_dj,spatial_cell->intersection_y_dk,1); // map along y
         break;
      }
   }
   mappingsTimer.stop();
}
