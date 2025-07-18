//
// Created by omar on 30.06.25.
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Assuming all your typedefs are declared above this function

void serialize_kernels_map_3d(const KernelsMap3D* map, const char* filename);

KernelsMap3D* deserialize_kernels_map_3d(const char* filename);

#ifdef __cplusplus
}
#endif
