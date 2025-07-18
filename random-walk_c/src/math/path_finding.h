#pragma once

#include <stdio.h>

#include "parsers/terrain_parser.h"
#include "matrix/matrix.h"


#ifdef __cplusplus
extern "C" {
#endif


Matrix *get_reachability_kernel(ssize_t x, ssize_t y, ssize_t kernel_size, const TerrainMap *terrain);

#ifdef __cplusplus
}
#endif
