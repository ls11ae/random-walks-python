#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <assert.h>

#include "math/Point2D.h"
#include "c_walk.h"
#include "b_walk.h"
#include <stdlib.h>
#include <stdio.h>


Point2DArray* mixed_walk(ssize_t W, ssize_t H, TerrainMap* spatial_map,
                         KernelsMap3D* tensor_map, Tensor* c_kernel, ssize_t T, const Point2DArray* steps);

Tensor** m_walk(ssize_t W, ssize_t H, const TerrainMap* terrain_map,
                const KernelsMap3D* kernels_map, ssize_t T, ssize_t start_x,
                ssize_t start_y);

Point2DArray* m_walk_backtrace(Tensor** DP_Matrix, const ssize_t T,
                               KernelsMap3D* tensor_map, TerrainMap* terrain, ssize_t end_x, ssize_t end_y,
                               ssize_t dir);

Tensor** mixed_walk_time(ssize_t W, ssize_t H,
                         TerrainMap* terrain_map,
                         KernelsMap4D* kernels_map, ssize_t T,
                         ssize_t start_x,
                         ssize_t start_y);

Point2DArray* backtrace_time_walk(Tensor** DP_Matrix, const ssize_t T, const TerrainMap* terrain,
                                  const KernelsMap4D* kernels_map, ssize_t end_x, ssize_t end_y, ssize_t dir);


Point2DArray* time_walk_geo(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                            int grid_x, int grid_y,
                            Point2D start, Point2D goal);

Point2DArray* time_walk_geo_multi(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                                  int grid_x, int grid_y,
                                  Point2DArray* steps);

#ifdef __cplusplus
}
#endif
