#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>

#include "move_bank_parser.h"


typedef struct CacheEntry {
    uint64_t hash;

    union {
        Tensor* array; // For tensor_map_new
        Matrix* single; // For kernels_map_new
    } data;

    bool is_array;
    ssize_t array_size;
    struct CacheEntry* next;
} CacheEntry;

typedef struct {
    CacheEntry** buckets;
    size_t num_buckets;
} Cache;

typedef struct {
    Matrix*** kernels;
    ssize_t width, height;
    Cache* cache;
} KernelsMap;

typedef struct {
    Tensor*** kernels; // 3D [y][x][d]
    ssize_t width, height, max_D;
    Cache* cache;
} KernelsMap3D;

typedef struct {
    Tensor**** kernels; // 4D array [y][x][t][d]
    ssize_t width, height, timesteps, max_D;
    Cache* cache;
} KernelsMap4D;

KernelsMap* kernels_map_new(const TerrainMap* terrain, const Matrix* kernel);

KernelsMap3D* tensor_map_new(const TerrainMap* terrain, const Tensor* kernels);

KernelsMap3D* tensor_map_mixed(const TerrainMap* terrain, TensorSet* tensor_set);

KernelsMap4D* tensor_map_terrain_weather(TerrainMap* terrain, const WeatherGrid* weather_grid);

KernelsMap4D* tensor_map_terrain_biased(TerrainMap* terrain, Point2DArray* biases);

KernelsMap4D* tensor_map_terrain_biased_grid(TerrainMap* terrain, Point2DArrayGrid* biases);

KernelsMap3D* tensor_map_terrain(TerrainMap* terrain);

Matrix* kernel_at(const KernelsMap* kernels_map, ssize_t x, ssize_t y);

void kernels_map_free(KernelsMap* kernels_map);

void tensor_map_free(KernelsMap** tensor_map, size_t D);

void kernels_map3d_free(KernelsMap3D* kernels_map);

void kernels_map4d_free(KernelsMap4D* map);

TerrainMap* get_terrain_map(const char* file, char delimiter);

int terrain_at(ssize_t x, ssize_t y, const TerrainMap* terrain_map);

void terrain_set(const TerrainMap* terrain_map, ssize_t x, ssize_t y, int value);

TerrainMap* terrain_map_new(ssize_t width, ssize_t height);

void terrain_map_free(TerrainMap* terrain_map);

int parse_terrain_map(const char* filename, TerrainMap* map, char delimiter);

    TerrainMap* create_terrain_map(const char* filename, char delimiter);

bool kernels_maps_equal(const KernelsMap3D* kmap3d, const KernelsMap4D* kmap4d);

#ifdef __cplusplus
}
#endif
