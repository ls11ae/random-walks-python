#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "terrain_parser.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <asm-generic/errno-base.h>

#include "move_bank_parser.h"
#include "math/path_finding.h"
#include "walk/c_walk.h"


uint64_t compute_matrix_hash(const Matrix* m) {
    uint64_t h = 146527;
    for (size_t i = 0; i < m->len; i++) {
        uint64_t bits;
        memcpy(&bits, &m->data[i], sizeof(bits));
        h ^= bits + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

uint64_t compute_parameters_hash(const KernelParameters* params) {
    uint64_t h = 14695981039346656037ULL;
    h = (h ^ (params->is_brownian)) * 1099511628211ULL;
    h = (h ^ params->S) * 1099511628211ULL;
    h = (h ^ params->D) * 1099511628211ULL;
    // Hash float values by their bit pattern
    uint64_t bits;
    memcpy(&bits, &params->diffusity, sizeof(bits));
    h = (h ^ bits) * 1099511628211ULL;

    h = (h ^ params->bias_x) * 1099511628211ULL;
    h = (h ^ params->bias_y) * 1099511628211ULL;

    return h;
}

uint32_t hash_bytes(const void* key, size_t length) {
    const uint8_t* data = (const uint8_t*)key;
    uint32_t hash = 0;
    for (size_t i = 0; i < length; i++) {
        hash += data[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

uint32_t weather_entry_hash(const WeatherEntry* entry) {
    uint32_t hash = 0;
    // Hash each field individually and combine them
    hash = hash_bytes(&entry->temperature, sizeof(entry->temperature));
    hash ^= hash_bytes(&entry->humidity, sizeof(entry->humidity));
    hash ^= hash_bytes(&entry->precipitation, sizeof(entry->precipitation));
    hash ^= hash_bytes(&entry->wind_speed, sizeof(entry->wind_speed));
    hash ^= hash_bytes(&entry->wind_direction, sizeof(entry->wind_direction));
    hash ^= hash_bytes(&entry->snow_fall, sizeof(entry->snow_fall));
    hash ^= hash_bytes(&entry->weather_code, sizeof(entry->weather_code));
    hash ^= hash_bytes(&entry->cloud_cover, sizeof(entry->cloud_cover));

    // Final mixing
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}


Cache* cache_create(size_t num_buckets) {
    Cache* cache = (Cache*)malloc(sizeof(Cache));
    cache->num_buckets = num_buckets;
    cache->buckets = (CacheEntry**)calloc(num_buckets, sizeof(CacheEntry*));
    return cache;
}

CacheEntry* cache_lookup_entry(Cache* cache, uint64_t hash) {
    size_t bucket = hash % cache->num_buckets;

    CacheEntry* entry = cache->buckets[bucket];

    while (entry != NULL) {
        if (entry->hash == hash) {
            return entry;
        }

        entry = entry->next;
    }

    return NULL;
}

TensorSet* generate_correlated_tensors() {
    const int terrain_count = 11;
    Tensor** tensors = malloc(terrain_count * sizeof(Tensor*));
    const enum landmarkType landmarkTypes[11] = {
        TREE_COVER, SHRUBLAND, GRASSLAND, CROPLAND, BUILT_UP, SPARSE_VEGETATION, SNOW_AND_ICE, WATER,
        HERBACEOUS_WETLAND, MANGROVES,
        MOSS_AND_LICHEN
    };
    for (int i = 0; i < terrain_count; i++) {
        KernelParameters* parameters = kernel_parameters_terrain(landmarkTypes[i]);
        ssize_t t_D = parameters->D;
        ssize_t M = parameters->S * 2 + 1;
        tensors[i] = generate_kernels(t_D, M);
        free(parameters);
    }
    TensorSet* correlated_kernels = tensor_set_new(terrain_count, tensors);
    return correlated_kernels;
}

void cache_insert(Cache* cache, uint64_t hash, void* data, bool is_array, ssize_t array_size) {
    size_t bucket = hash % cache->num_buckets;
    CacheEntry* entry = malloc(sizeof(CacheEntry));
    entry->hash = hash;
    entry->is_array = is_array;
    entry->array_size = array_size;
    if (is_array) {
        entry->data.array = (Tensor*)data;
    }
    else {
        entry->data.single = (Matrix*)data;
    }
    entry->next = cache->buckets[bucket];
    cache->buckets[bucket] = entry;
}

void cache_free(Cache* cache) {
    for (size_t i = 0; i < cache->num_buckets; i++) {
        CacheEntry* entry = cache->buckets[i];
        while (entry != NULL) {
            CacheEntry* next = entry->next;
            if (entry->is_array) {
                tensor_free(entry->data.array);
            }
            else {
                matrix_free(entry->data.single);
            }
            free(entry);
            entry = next;
        }
    }
    free(cache->buckets);
    free(cache);
}

uint64_t hash_combine(uint64_t a, uint64_t b) {
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

KernelsMap* kernels_map_new(const TerrainMap* terrain, const Matrix* kernel) {
    KernelsMap* kernels_map = malloc(sizeof(KernelsMap));
    kernels_map->kernels = malloc(terrain->height * sizeof(Matrix**));
    for (ssize_t y = 0; y < terrain->height; y++) {
        kernels_map->kernels[y] = malloc(terrain->width * sizeof(Matrix*));
    }
    kernels_map->width = terrain->width;
    kernels_map->height = terrain->height;
    kernels_map->cache = cache_create(1024);

    const uint64_t kernel_hash = compute_matrix_hash(kernel);
    const ssize_t kernel_size = kernel->width;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t y = 0; y < terrain->height; y++) {
        for (ssize_t x = 0; x < terrain->width; x++) {
            if (terrain_at(x, y, terrain) == 0) continue;
            Matrix* reachable = get_reachability_kernel(x, y, kernel_size, terrain);
            const uint64_t reachable_hash = compute_matrix_hash(reachable);
            const uint64_t combined_hash = reachable_hash ^ kernel_hash;

            CacheEntry* entry = cache_lookup_entry(kernels_map->cache, combined_hash);
            Matrix* current = NULL;
            if (entry && !entry->is_array) {
                current = entry->data.single;
            }
            else {
                current = matrix_elementwise_mul(kernel, reachable);
                matrix_normalize_L1(current);
                cache_insert(kernels_map->cache, combined_hash, current, false, 0);
            }
            kernels_map->kernels[y][x] = current;
            matrix_free(reachable);
        }
    }
    return kernels_map;
}

KernelsMap3D* tensor_map_new(const TerrainMap* terrain, const Tensor* kernels) {
    ssize_t D = (ssize_t)kernels->len;
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    ssize_t M = (ssize_t)kernels->data[0]->width;

    KernelsMap3D* kernels_map = (KernelsMap3D*)malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;

    Cache* cache = cache_create(1024);

    // Precompute tensor hash using all kernels
    uint64_t tensor_hash = 0;
    for (ssize_t d = 0; d < D; d++) {
        tensor_hash ^= compute_matrix_hash(kernels->data[d]);
    }

    // Precompute combined hashes and populate cache
    uint64_t** hash_grid = (uint64_t**)malloc(terrain_height * sizeof(uint64_t*));
    for (ssize_t y = 0; y < terrain_height; y++) {
        hash_grid[y] = (uint64_t*)malloc(terrain_width * sizeof(uint64_t));
        for (ssize_t x = 0; x < terrain_width; x++) {
            if (terrain_at(x, y, terrain) == 0) continue;
            Matrix* reachable = get_reachability_kernel(x, y, M, terrain);
            uint64_t reachable_hash = compute_matrix_hash(reachable);
            uint64_t combined_hash = reachable_hash ^ tensor_hash;
            hash_grid[y][x] = combined_hash;

            // Check cache with combined hash
            CacheEntry* entry = cache_lookup_entry(cache, combined_hash);
            Tensor* kernels_arr = NULL;

            if (entry && entry->is_array && entry->array_size == D) {
                kernels_arr = entry->data.array;
            }
            else {
                // Compute and cache if not found
                kernels_arr = tensor_new(M, M, D);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix* current = matrix_elementwise_mul(kernels->data[d], reachable);
                    matrix_normalize_L1(current);
                    kernels_arr->data[d] = current;
                }
                cache_insert(cache, combined_hash, kernels_arr, true, D);
            }
            matrix_free(reachable);
        }
    }

    // Build kernels map from cache
    kernels_map->kernels = (Tensor***)malloc(terrain_height * sizeof(Tensor**));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = (Tensor**)malloc(terrain_width * sizeof(Tensor*));
        for (ssize_t x = 0; x < terrain_width; x++) {
            if (terrain_at(x, y, terrain) == 0) continue;
            CacheEntry* entry = cache_lookup_entry(cache, hash_grid[y][x]);
            if (entry) {
                kernels_map->kernels[y][x] = entry->data.array;
            }
            else {
                // Fallback
                fprintf(stderr, "Critical cache miss at (%zd, %zd)\n", x, y);
                exit(EXIT_FAILURE);
            }
        }
        free(hash_grid[y]);
    }
    free(hash_grid);
    kernels_map->cache = cache;

    return kernels_map;
}

KernelsMap3D* tensor_map_mixed(const TerrainMap* terrain, TensorSet* tensor_set) {
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;

    KernelsMap3D* kernels_map = (KernelsMap3D*)malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->kernels = malloc(terrain->height * sizeof(Tensor**));
    for (ssize_t y = 0; y < terrain->height; y++) {
        kernels_map->kernels[y] = malloc(terrain->width * sizeof(Tensor*));
    }

    Cache* cache = cache_create(4096);

    // Precompute tensor hash using all kernels
    uint64_t* tensor_hashes = (uint64_t*)malloc(tensor_set->len * sizeof(uint64_t));
    for (ssize_t i = 1; i < tensor_set->len; i++) {
        size_t D = tensor_set->data[i]->len;
        uint64_t tensor_hash = 0;
        for (ssize_t d = 0; d < D; d++) {
            tensor_hash ^= compute_matrix_hash(tensor_set->data[i]->data[d]);
        }
        tensor_hashes[i] = tensor_hash;
    }
    size_t recomputed = 0;
    // Precompute combined hashes and populate cache
    uint64_t** hash_grid = (uint64_t**)malloc(terrain_height * sizeof(uint64_t*));
    for (ssize_t y = 0; y < terrain_height; y++) {
        hash_grid[y] = (uint64_t*)malloc(terrain_width * sizeof(uint64_t));
        for (ssize_t x = 0; x < terrain_width; x++) {
            const size_t terrain_val = terrain_at(x, y, terrain);
            if (terrain_val == 0) continue;
            const size_t tensor_index = terrain_val - 1; // Convert to 0-based index

            if (tensor_index >= tensor_set->len) continue;

            const ssize_t M = tensor_set->data[tensor_index]->data[0]->width;
            const ssize_t D = (ssize_t)tensor_set->data[tensor_index]->len;
            Matrix* reachable = get_reachability_kernel(x, y, M, terrain);
            uint64_t reachable_hash = compute_matrix_hash(reachable);
            uint64_t combined_hash = reachable_hash;
            combined_hash ^= tensor_hashes[tensor_index] + (D * 0x9e3779b97f4a7c15ULL) + (combined_hash << 6) + (
                combined_hash >> 2);
            hash_grid[y][x] = combined_hash;

            // Check cache with combined hash
            CacheEntry* entry = cache_lookup_entry(cache, combined_hash);
            Tensor* kernels_arr = NULL;

            if (entry && entry->is_array && entry->array_size == D) {
                kernels_arr = entry->data.array;
            }
            else {
                recomputed++;
                // Compute and cache if not found
                kernels_arr = tensor_new(M, M, D);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix* current = matrix_elementwise_mul(tensor_set->data[tensor_index]->data[d], reachable);
                    matrix_normalize_L1(current);
                    kernels_arr->data[d] = current;
                }
                cache_insert(cache, combined_hash, kernels_arr, true, D);
            }
            matrix_free(reachable);
            kernels_map->kernels[y][x] = kernels_arr;
        }
    }

    // Build kernels map from cache
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            const size_t terrain_val = terrain_at(x, y, terrain);
            if (terrain_val == 0) continue;
            const size_t tensor_index = terrain_val - 1; // Convert to 0-based index

            if (tensor_index >= tensor_set->len) continue;
            const ssize_t D = (ssize_t)tensor_set->data[tensor_index]->len;
            CacheEntry* entry = cache_lookup_entry(cache, hash_grid[y][x]);
            if (entry) {
                kernels_map->kernels[y][x] = entry->data.array;
            }
            else {
                // Fallback
                fprintf(stderr, "Critical cache miss at (%zd, %zd)\n", x, y);
                exit(EXIT_FAILURE);
            }
        }
        free(hash_grid[y]);
    }
    free(hash_grid);
    kernels_map->cache = cache;
    printf("Recomputed: %zd\n", recomputed);
    return kernels_map;
}


Tensor* generate_tensor(const KernelParameters* p, int terrain_value, bool full_bias, TensorSet* correlated_tensors) {
    size_t M = p->S * 2 + 1;
    if (p->is_brownian) {
        float scale, sigma;
        get_gaussian_parameters(p->diffusity, terrain_value, &scale, &sigma);
        Matrix* kernel;
        if (full_bias)
            kernel = matrix_generator_gaussian_pdf(M, M, (double)sigma, (double)scale, p->bias_x, p->bias_y);
        else
            kernel = matrix_gaussian_pdf_alpha(M, M, (double)sigma, (double)scale, p->bias_x, p->bias_y);

        Tensor* result = tensor_new(M, M, 1);
        Vector2D* dir_kernel = get_dir_kernel(1, M);
        result->dir_kernel = dir_kernel;
        result->len = 1;
        result->data[0] = kernel;
        return result;
    }

    int index;
    if (terrain_value == MANGROVES) index = 9;
    else index = terrain_value / 10 - 1;
    Tensor* result = correlated_tensors->data[index];
    assert(result);
    return result;
}

KernelsMap4D* tensor_map_terrain_weather(TerrainMap* terrain, const WeatherGrid* weather_grid) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather* tensor_set = get_kernels_terrain_weather(terrain, weather_grid);
    const ssize_t terrain_width = 100; //terrain->width;
    const ssize_t terrain_height = 100; //terrain->height;
    const ssize_t time_steps = (ssize_t)tensor_set->time;

    printf("kernel parameters set\n");

    // 2) Map und Cache anlegen
    KernelsMap4D* kernels_map = malloc(sizeof(KernelsMap4D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->timesteps = time_steps;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor***));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor**));
        for (ssize_t x = 0; x < terrain_width; x++) {
            kernels_map->kernels[y][x] = malloc(time_steps * sizeof(Tensor*));
        }
    }


    Cache* cache = cache_create(4096);

    TensorSet* c_kernels = generate_correlated_tensors();

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            for (ssize_t t = 0; t < tensor_set->time; t++)
                if ((size_t)tensor_set->data[i][j][t]->D > maxD)
                    maxD = tensor_set->data[i][j][t]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == WATER) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }
                // weather indices to terrain grid
                size_t weather_x = (x * weather_grid->width) / terrain->width;
                size_t weather_y = (y * weather_grid->height) / terrain->height;

                // paranoia-check (clamping)
                if (weather_x >= weather_grid->width) weather_x = weather_grid->width - 1;
                if (weather_y >= weather_grid->height) weather_y = weather_grid->height - 1;

                WeatherTimeline* timeline = weather_grid->entries[weather_y][weather_x];
                WeatherEntry* w_entry = timeline->data[t];
                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                uint64_t w_params = weather_entry_hash(w_entry);
                Matrix* reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                uint64_t h_reach = compute_matrix_hash(reach_mat);
                uint64_t pre_combined = hash_combine(h_params, h_reach);
                uint64_t combined = hash_combine(pre_combined, w_params);

                // b) Cache‐Lookup
                CacheEntry* entry = cache_lookup_entry(cache, combined);
                Tensor* arr;
                if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                    arr = entry->data.array;
                }
                else {
                    // c) Cache‐Miss → neu berechnen und einfügen
                    recomputed++;
                    ssize_t D = tensor_set->data[y][x][t]->D;
                    arr = generate_tensor(tensor_set->data[y][x][t], (int)terrain_val, true, c_kernels);
                    for (ssize_t d = 0; d < D; d++) {
                        Matrix* m = matrix_elementwise_mul(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(m);
                        arr->data[d] = m;
                    }
                    cache_insert(cache, combined, arr, true, D);
                }

                // d) Aufräumen und Zuordnung
                matrix_free(reach_mat);
                kernels_map->kernels[y][x][t] = arr;
            }
        }
    }

    // 5) Abschluss
    printf("Recomputed: %i / %zu\n", recomputed, terrain_width * terrain->height * time_steps);
    kernels_map->cache = cache;
    kernel_parameters_mixed_free(tensor_set);
    return kernels_map;
}

KernelsMap4D* tensor_map_terrain_biased(TerrainMap* terrain, Point2DArray* biases) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather* tensor_set = get_kernels_terrain_biased(terrain, biases);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t)tensor_set->time;

    printf("kernel parameters set\n");

    // 2) Map und Cache anlegen
    KernelsMap4D* kernels_map = malloc(sizeof(KernelsMap4D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->timesteps = time_steps;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor***));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor**));
        for (ssize_t x = 0; x < terrain_width; x++) {
            kernels_map->kernels[y][x] = malloc(time_steps * sizeof(Tensor*));
        }
    }


    Cache* cache = cache_create(4096);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            for (ssize_t t = 0; t < tensor_set->time; t++)
                if ((size_t)tensor_set->data[i][j][t]->D > maxD)
                    maxD = tensor_set->data[i][j][t]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;
    TensorSet* ck = generate_correlated_tensors();

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == WATER) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }


                Point2D bias = biases->points[t];
                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                uint64_t w_params = ((uint64_t)(bias.x) << 32) | (uint32_t)(bias.y);
                Matrix* reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                uint64_t h_reach = compute_matrix_hash(reach_mat);
                uint64_t pre_combined = hash_combine(h_params, h_reach);
                uint64_t combined = hash_combine(pre_combined, w_params);

                // b) Cache‐Lookup
                CacheEntry* entry = cache_lookup_entry(cache, combined);
                Tensor* arr;
                if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                    arr = entry->data.array;
                }
                else {
                    // c) Cache‐Miss → neu berechnen und einfügen
                    recomputed++;
                    ssize_t D = tensor_set->data[y][x][t]->D;
                    arr = generate_tensor(tensor_set->data[y][x][t], (int)terrain_val, true, ck);
                    for (ssize_t d = 0; d < D; d++) {
                        Matrix* m = matrix_elementwise_mul(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(m);
                        arr->data[d] = m;
                    }
                    cache_insert(cache, combined, arr, true, D);
                }

                // d) Aufräumen und Zuordnung
                matrix_free(reach_mat);
                kernels_map->kernels[y][x][t] = arr;
            }
        }
    }

    // 5) Abschluss
    printf("Recomputed: %i / %zu\n", recomputed, terrain_width * terrain->height * time_steps);
    kernels_map->cache = cache;
    kernel_parameters_mixed_free(tensor_set);
    return kernels_map;
}

KernelsMap4D* tensor_map_terrain_biased_grid(TerrainMap* terrain, Point2DArrayGrid* biases) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather* tensor_set = get_kernels_terrain_biased_grid(terrain, biases);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t)tensor_set->time;


    printf("kernel parameters set\n");

    // 2) Map und Cache anlegen
    KernelsMap4D* kernels_map = malloc(sizeof(KernelsMap4D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->timesteps = time_steps;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor***));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor**));
        for (ssize_t x = 0; x < terrain_width; x++) {
            kernels_map->kernels[y][x] = malloc(time_steps * sizeof(Tensor*));
        }
    }

    TensorSet* correlated_kernels = generate_correlated_tensors();

    Cache* cache = cache_create(20000);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            for (ssize_t t = 0; t < tensor_set->time; t++)
                if ((size_t)tensor_set->data[i][j][t]->D > maxD)
                    maxD = tensor_set->data[i][j][t]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        //printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == WATER) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }

                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                Matrix* reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                uint64_t h_reach = compute_matrix_hash(reach_mat);
                uint64_t pre_combined = hash_combine(h_params, h_reach);

                // b) Cache‐Lookup
                CacheEntry* entry = cache_lookup_entry(cache, pre_combined);
                Tensor* arr;
                if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                    arr = entry->data.array;
                }
                else {
                    // c) Cache‐Miss → neu berechnen und einfügen
                    recomputed++;
                    ssize_t D = tensor_set->data[y][x][t]->D;
                    arr = generate_tensor(tensor_set->data[y][x][t], (int)terrain_val, true, correlated_kernels);
                    for (ssize_t d = 0; d < D; d++) {
                        Matrix* m = matrix_elementwise_mul(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(m);
                        arr->data[d] = m;
                    }
                    cache_insert(cache, pre_combined, arr, true, D);
                }

                // d) Aufräumen und Zuordnung
                matrix_free(reach_mat);
                kernels_map->kernels[y][x][t] = arr;
            }
        }
    }

    // 5) Abschluss
    printf("Recomputed: %i / %zu\n", recomputed, terrain_width * terrain->height * time_steps);
    kernels_map->cache = cache;
    kernel_parameters_mixed_free(tensor_set);
    return kernels_map;
}


KernelsMap3D* tensor_map_terrain(TerrainMap* terrain) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrain* tensor_set = get_kernels_terrain(terrain);
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;

    // 2) Map und Cache anlegen
    KernelsMap3D* kernels_map = malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor**));
    for (ssize_t y = 0; y < terrain_height; y++)
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor*));

    Cache* cache = cache_create(4096);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    size_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            if ((size_t)tensor_set->data[i][j]->D > maxD)
                maxD = tensor_set->data[i][j]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;
    TensorSet* correlated_kernels = generate_correlated_tensors();


    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("%zd\n", y);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            if (terrain_val == WATER) {
                kernels_map->kernels[y][x] = NULL;
                continue;
            }

            // a) Einzel-Hashes
            uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x]);
            Matrix* reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x]->S + 1, terrain);
            uint64_t h_reach = compute_matrix_hash(reach_mat);
            uint64_t combined = hash_combine(h_params, h_reach);
            combined = hash_combine(combined, tensor_set->data[y][x]->D);

            // b) Cache‐Lookup
            CacheEntry* entry = cache_lookup_entry(cache, combined);
            Tensor* arr;
            if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x]->D) {
                arr = entry->data.array;
            }
            else {
                // c) Cache‐Miss → neu berechnen und einfügen
                recomputed++;
                ssize_t D = tensor_set->data[y][x]->D;
                arr = generate_tensor(tensor_set->data[y][x], (int)terrain_val, false, correlated_kernels);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix* m = matrix_elementwise_mul(
                        arr->data[d],
                        reach_mat
                    );
                    matrix_normalize_L1(m);
                    arr->data[d] = m;
                }
                cache_insert(cache, combined, arr, true, D);
            }

            // d) Aufräumen und Zuordnung
            matrix_free(reach_mat);
            kernels_map->kernels[y][x] = arr;
        }
    }

    // 5) Abschluss
    printf("Recomputed: %d\n", recomputed);
    kernels_map->cache = cache;
    kernel_parameters_terrain_free(tensor_set);
    tensor_set_free(correlated_kernels);
    return kernels_map;
}


void kernels_map_free(KernelsMap* kernels_map) {
    if (kernels_map == NULL) { return; }
    for (ssize_t y = 0; y < kernels_map->height; y++) {
        for (ssize_t x = 0; x < kernels_map->width; x++) {
            if (kernels_map->kernels[y][x])
                matrix_free(kernels_map->kernels[y][x]);
        }
        free(kernels_map->kernels[y]);
    }
    free(kernels_map->kernels);
    free(kernels_map);
}

void kernels_map3d_free(KernelsMap3D* map) {
    cache_free(map->cache);
    for (int i = 0; i < map->height; ++i) {
        for (int j = 0; j < map->width; ++j) {
            tensor_free(map->kernels[i][j]);
        }
    }
    free(map->kernels);
}

void kernels_map4d_free(KernelsMap4D* map) {
    cache_free(map->cache);
    for (int i = 0; i < map->height; ++i) {
        for (int j = 0; j < map->width; ++j) {
            for (int t = 0; t < map->timesteps; ++t) {
                if (map->kernels[i][j][t] != NULL)
                    tensor_free(map->kernels[i][j][t]);
            }
            free(map->kernels[i][j]);
        }
        free(map->kernels[i]);
    }
    free(map->kernels);
}


void tensor_map_free(KernelsMap** tensor_map, const size_t D) {
    if (tensor_map == NULL) { return; }
    for (size_t d = 0; d < D; d++) {
        kernels_map_free(tensor_map[d]);
    }
    free(tensor_map);
}

Matrix* kernel_at(const KernelsMap* kernels_map, ssize_t x, ssize_t y) {
    assert(x < kernels_map->width && y < kernels_map->height&& x >= 0 && y >= 0);
    return kernels_map->kernels[y][x];
}

TerrainMap* get_terrain_map(const char* file, const char delimiter) {
    TerrainMap* terrain_map = malloc(sizeof(TerrainMap));
    if (parse_terrain_map(file, terrain_map, delimiter) != 0) {
        fprintf(stderr, "Failed to parse terrain map file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    return terrain_map;
}

int terrain_at(const ssize_t x, const ssize_t y, const TerrainMap* terrain_map) {
    assert(x >= 0 && y >= 0 && x < terrain_map->width && y < terrain_map->height);
    return terrain_map->data[y][x];
}

void terrain_set(const TerrainMap* terrain_map, ssize_t x, ssize_t y, int value) {
    assert(terrain_map != NULL);
    terrain_map->data[y][x] = value;
}

TerrainMap* terrain_map_new(const ssize_t width, const ssize_t height) {
    TerrainMap* map = malloc(sizeof(TerrainMap));
    if (!map) return NULL;
    map->width = width;
    map->height = height;

    map->data = malloc(height * sizeof(int*));
    if (!map->data) {
        free(map);
        return NULL;
    }

    for (ssize_t y = 0; y < height; ++y) {
        map->data[y] = malloc(width * sizeof(int));
        if (!map->data[y]) {
            for (ssize_t i = 0; i < y; ++i) free(map->data[i]);
            free(map->data);
            free(map);
            return NULL;
        }
    }

    return map;
}


void terrain_map_free(TerrainMap* terrain_map) {
    if (terrain_map == NULL) return;
    for (size_t y = 0; y < terrain_map->height; y++) {
        free(terrain_map->data[y]);
    }
    free(terrain_map->data);
    free(terrain_map);
}

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 8192
#endif

int parse_terrain_map(const char* filename, TerrainMap* map, char delimiter) {
    FILE* file = NULL;
    char line_buffer[MAX_LINE_LENGTH];
    char delim_str[2]; // For strtok, which requires a null-terminated string

    if (filename == NULL || map == NULL) {
        return -1; // Invalid arguments
    }

    // Initialize map to a safe, empty state
    map->data = NULL;
    map->width = 0;
    map->height = 0;

    delim_str[0] = delimiter;
    delim_str[1] = '\0';

    file = fopen(filename, "r");
    if (file == NULL) {
        // perror("Error opening file"); // Uncomment for debug messages
        return -2; // File open error
    }

    // --- Pass 1: Determine width and height ---
    ssize_t calculated_width = 0;
    ssize_t calculated_height = 0;

    // Read the first line to attempt to determine width
    if (fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char* temp_line_for_width = strdup(line_buffer); // strtok modifies the string
        if (temp_line_for_width == NULL) {
            fclose(file);
            return -3; // Memory allocation error for strdup
        }

        char* current_pos_in_line = temp_line_for_width;
        // Skip any leading whitespace on the line before tokenizing
        while (*current_pos_in_line && isspace((unsigned char)*current_pos_in_line)) {
            current_pos_in_line++;
        }

        if (*current_pos_in_line == '\0') { // First line is effectively empty (all whitespace or truly empty)
            free(temp_line_for_width);
            // Check if the rest of the file is also empty
            if (fgets(line_buffer, sizeof(line_buffer), file) == NULL && feof(file)) {
                fclose(file); // Successfully parsed an empty map (file was empty or one empty line)
                return 0;
            }
            else {
                // First line was empty, but file has more content or a read error occurred.
                // This is considered a malformed map.
                fclose(file);
                return -5; // Invalid dimensions (malformed: first line empty in non-empty file)
            }
        }

        // First line has content; tokenize it to determine the width
        char* token = strtok(current_pos_in_line, delim_str);
        while (token) {
            calculated_width++;
            token = strtok(NULL, delim_str);
        }
        free(temp_line_for_width);

        if (calculated_width == 0) {
            // No tokens found on the first line (e.g., "abc" with space delimiter, or ",," with comma delimiter)
            fclose(file);
            return -5; // Invalid dimensions (no parsable tokens on the first potentially data-bearing line)
        }
        calculated_height = 1; // Counted the first non-empty line

        // Count remaining non-empty lines to determine the total height
        while (fgets(line_buffer, sizeof(line_buffer), file)) {
            char* p = line_buffer;
            while (*p && isspace((unsigned char)*p)) p++; // Skip leading whitespace
            // Consider a line non-empty if it has any non-whitespace characters
            if (*p != '\0' && *p != '\r' && *p != '\n') {
                calculated_height++;
            }
        }
    }
    else { // fgets failed for the very first line attempt
        if (feof(file)) { // File is completely empty
            fclose(file);
            return 0; // Successfully parsed an empty map
        }
        else { // A read error occurred on the first line
            // perror("Error reading file for dimensions"); // Uncomment for debug
            fclose(file);
            return -4; // File read error
        }
    }

    // If dimensions are zero at this point, it implies an empty map was processed (returned 0)
    // or a specific malformed case led to this state.
    if (calculated_width == 0 && calculated_height == 0) {
        // This path should ideally be covered by the "empty file" return 0.
        // If reached, it implies the file was effectively empty.
        if (file) fclose(file); // Ensure file is closed
        return 0;
    }
    // If only one dimension is zero, it's an error (e.g. content-less lines after a valid first line).
    if (calculated_width == 0 || calculated_height == 0) {
        fclose(file);
        return -5; // Invalid dimensions
    }

    map->width = calculated_width;
    map->height = calculated_height;

    // --- Memory Allocation for map data ---
    map->data = malloc((size_t)map->height * sizeof(int*));
    if (map->data == NULL) {
        fclose(file);
        terrain_map_free(map); // Reset map struct
        return -3; // Memory allocation error
    }
    // Initialize row pointers to NULL for safer cleanup in case of partial column allocation
    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = NULL;
    }

    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = malloc((size_t)map->width * sizeof(int));
        if (map->data[i] == NULL) {
            fclose(file);
            terrain_map_free(map); // Frees successfully allocated parts
            return -3; // Memory allocation error
        }
    }

    // --- Pass 2: Populate data ---
    rewind(file); // Go back to the beginning of the file to read data
    ssize_t current_row = 0;
    long val;
    char* endptr; // For strtol error checking

    while (current_row < map->height && fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char* line_content_start = line_buffer;
        // Skip leading whitespace to find actual content start
        while (*line_content_start && isspace((unsigned char)*line_content_start)) {
            line_content_start++;
        }

        if (*line_content_start == '\0') {
            // This line is effectively empty.
            // Height calculation only counted non-empty lines. So, if we encounter
            // an empty line here, it was not part of the expected `map->height` data lines.
            // We can skip it. The final check `current_row != map->height` will catch
            // if there are fewer actual data lines than determined.
            continue;
        }

        char* token = strtok(line_content_start, delim_str); // Start tokenizing from actual content

        for (ssize_t current_col = 0; current_col < map->width; current_col++) {
            if (token == NULL) { // Not enough tokens in the current line
                fclose(file);
                terrain_map_free(map);
                return -7; // Row width mismatch (too few values)
            }

            errno = 0; // Reset errno before calling strtol
            val = strtol(token, &endptr, 10); // Base 10 conversion

            if (errno == ERANGE) { // Value out of range for 'long'
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of long range)
            }
            if (endptr == token || *endptr != '\0') {
                // No digits were converted, or there were non-numeric trailing characters in the token
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (invalid number format in token)
            }
            // Check if the parsed 'long' value fits into an 'int'
            if (val < INT_MIN || val > INT_MAX) {
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of int range)
            }

            map->data[current_row][current_col] = (int)val;
            token = strtok(NULL, delim_str); // Get the next token
        }

        // After iterating through the expected number of columns, check if there are more tokens
        if (token != NULL) { // Extra tokens found on the line
            fclose(file);
            terrain_map_free(map);
            return -7; // Row width mismatch (too many values)
        }
        current_row++; // Successfully parsed a row
    }

    fclose(file); // Close the file after processing

    // Final check: ensure the number of rows processed matches the expected height
    if (current_row != map->height) {
        // This implies that fewer valid data rows were found than expected.
        // (e.g., file ended prematurely, or more blank lines than anticipated by parsing logic)
        terrain_map_free(map); // The map is incomplete or invalid
        return -8; // Row count mismatch
    }

    return 0; // Success!
}

TerrainMap* create_terrain_map(const char* filename, char delimiter) {
    TerrainMap* terrain_map = malloc(sizeof(TerrainMap));
    if (terrain_map == NULL) {
        free(terrain_map);
        printf("terrain map failed\n");
    }
    parse_terrain_map(filename, terrain_map, delimiter);
    return terrain_map;
}

bool kernels_maps_equal(const KernelsMap3D* kmap3d, const KernelsMap4D* kmap4d) {
    // Check if either pointer is NULL
    if (!kmap3d || !kmap4d) {
        return false;
    }

    // Check basic dimensions match
    if (kmap3d->width != kmap4d->width || kmap3d->height != kmap4d->height ||
        kmap3d->max_D != kmap4d->max_D) {
        return false;
    }

    // Check each x,y position
    for (ssize_t y = 0; y < kmap3d->height; y++) {
        for (ssize_t x = 0; x < kmap3d->width; x++) {
            // Get the 3D kernel at (x,y)
            Tensor* kernel3d = kmap3d->kernels[y][x];

            // Check all timesteps in 4D map against the 3D kernel
            for (ssize_t t = 0; t < kmap4d->timesteps; t++) {
                Tensor* kernel4d = kmap4d->kernels[y][x][t];

                if (!tensor_equals(kernel3d, kernel4d)) {
                    return false;
                }
            }
        }
    }

    return true;
}
