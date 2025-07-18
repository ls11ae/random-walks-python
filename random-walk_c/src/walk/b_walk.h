#pragma once
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walk.h"

#ifdef __cplusplus
extern "C" {
#endif

    Tensor *brownian_walk_init(ssize_t T,
                               ssize_t W,
                               ssize_t H,
                               ssize_t start_x,
                               ssize_t start_y,
                               Matrix *kernel);

    Tensor *brownian_walk_terrain_init(ssize_t T,
                                       ssize_t W,
                                       ssize_t H,
                                       ssize_t start_x,
                                       ssize_t start_y,
                                       Matrix *kernel,
                                       const TerrainMap *terrain_map,
                                       KernelsMap *kernels_map);


Point2DArray* brownian_backtrace(const Tensor* dp_tensor, Matrix* kernel, ssize_t end_x, ssize_t end_y);

Point2DArray* brownian_backtrace_terrain(Tensor* dp_tensor, Matrix* kernel, KernelsMap* kernels_map, ssize_t end_x,
                                         ssize_t end_y);

Tensor* b_walk_A_init(Matrix* matrix_start, Matrix* matrix_kernel, ssize_t T);

Tensor* b_walk_init_terrain(const Matrix* matrix_start, const Matrix* matrix_kernel, const TerrainMap* terrain_map,
                            const KernelsMap* kernels_map, ssize_t T);

Tensor* get_brownian_kernel(ssize_t M, double sigma, double scale);

Point2DArray* b_walk_backtrace(const Tensor* tensor, Matrix* kernel, KernelsMap* kernels_map,
                               ssize_t x, ssize_t y);

Point2DArray* b_walk_backtrace_multiple(ssize_t T, ssize_t W, ssize_t H, Matrix* kernel,
                                        KernelsMap* kernels_map,
                                        const Point2DArray* steps);


double calculate_ram_mib(ssize_t D, ssize_t W, ssize_t H, ssize_t T, bool terrain_map);

double get_mem_available_mib();

#ifdef __cplusplus
}
#endif
