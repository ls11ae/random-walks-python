#pragma once

#include "matrix/matrix.h"
#include "matrix/tensor.h"
//#include "walk_data.h"
#include "math/Point2D.h"
#include "matrix/ScalarMapping.h"
#include "parsers/terrain_parser.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    size_t x;
    size_t y;
    size_t d;
} TEMP;

#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)

Matrix* generate_chi_kernel(ssize_t size, ssize_t subsample_size, int k, int d);

Tensor** dp_calculation(ssize_t W, ssize_t H, const Tensor* kernel, ssize_t T, ssize_t start_x, ssize_t start_y);

Point2DArray* backtrace(Tensor** DP_Matrix, ssize_t T, const Tensor* kernel,
                        TerrainMap* terrain, KernelsMap3D* tensor_map, ssize_t end_x, ssize_t end_y, ssize_t dir,
                        ssize_t D);

void dp_calculation_low_ram(ssize_t W, ssize_t H, const Tensor* kernel, const ssize_t T, const ssize_t start_x,
                            const ssize_t start_y, const char* output_folder);

void c_walk_init_terrain_low_ram(ssize_t W, ssize_t H, const Tensor* kernel, const TerrainMap* terrain_map,
                                 const KernelsMap3D* kernels_map, const ssize_t T, const ssize_t start_x,
                                 const ssize_t start_y, const char* output_folder);

Point2DArray* backtrace_low_ram(const char* dp_folder, const ssize_t T, const Tensor* kernel,
                                KernelsMap3D* tensor_map, ssize_t end_x, ssize_t end_y, ssize_t dir, ssize_t D);


Point2DArray* c_walk_backtrace_multiple(ssize_t T, ssize_t W, ssize_t H, Tensor* kernel, TerrainMap* terrain,
                                        KernelsMap3D* kernels_map,
                                        const Point2DArray* steps);

Tensor** c_walk_init_terrain(ssize_t W, ssize_t H, const Tensor* kernel, const TerrainMap* terrain_map,
                             const KernelsMap3D* kernels_map, ssize_t T, ssize_t start_x, ssize_t start_y);

    Point2DArray* c_walk_backtrace_multiple_no_terrain(ssize_t T_c, ssize_t W_c, ssize_t H_c, Tensor* kernel_c,
                                                       Point2DArray* steps_c);

Tensor* generate_kernels(ssize_t dirs, ssize_t size);

Matrix* assign_sectors_matrix(ssize_t width, ssize_t height, ssize_t D);

Tensor* assign_sectors_tensor(ssize_t width, ssize_t height, int D);

#ifdef __cplusplus
}
#endif
