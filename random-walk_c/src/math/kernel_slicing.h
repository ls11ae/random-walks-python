#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "matrix/matrix.h"
#include "matrix/tensor.h"

typedef struct {
    double x;
    double y;
} point;

typedef struct {
    size_t count;
    size_t *sizes;
    point **data;
} vec2;

double compute_angle_ks(double x, double y);

void compute_overlap_percentages(int W, int D, Tensor *tensor);

#ifdef __cplusplus
    }
#endif
