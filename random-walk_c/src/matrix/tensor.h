#pragma once

#include "math/Point2D.h"
#include "matrix/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Point2D** data;
    Point2D* grid_cells;
    size_t* sizes;
    size_t count;
} Vector2D;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    Matrix** data;
    Vector2D* dir_kernel;
} Tensor;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    size_t max_D;
    Tensor** data;
    Vector2D** grid_cells;
} TensorSet;

Tensor* tensor_new(size_t width, size_t height, size_t depth);

TensorSet* tensor_set_new(size_t count, Tensor** tensors);

void tensor_set_free(TensorSet* set);

bool tensor_equals(const Tensor* t1, const Tensor* t2);

Vector2D* get_dir_kernel(ssize_t D, ssize_t size);

void free_Vector2D(Vector2D* vec);

void tensor_free(Tensor* tensor);

Tensor* tensor_copy(const Tensor* original);

void tensor_fill(Tensor* tensor, double value);

int tensor_in_bounds(Tensor* tensor, size_t x, size_t y, size_t z);

size_t tensor_save(Tensor* tensor, const char* foldername);

Tensor* tensor_load(const char* foldername);

typedef struct {
    size_t len_data;
    Tensor** data;
} Tensor4D;

void tensor4D_free(Tensor** tensor, ssize_t T);


#ifdef __cplusplus
}
#endif
