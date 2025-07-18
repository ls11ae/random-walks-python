//
// Created by omar on 30.06.25.
//

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "terrain_parser.h"

// Assuming all your typedefs are declared above this function

void serialize_kernels_map_3d(const KernelsMap3D* map, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    // Write basic dimensions
    fwrite(&map->width, sizeof(ssize_t), 1, file);
    fwrite(&map->height, sizeof(ssize_t), 1, file);
    fwrite(&map->max_D, sizeof(ssize_t), 1, file);

    for (ssize_t y = 0; y < map->height; ++y) {
        for (ssize_t x = 0; x < map->width; ++x) {
            for (ssize_t d = 0; d < map->max_D; ++d) {
                Tensor* tensor = &map->kernels[y][x][d];
                if (!tensor) {
                    ssize_t zero = 0;
                    fwrite(&zero, sizeof(ssize_t), 1, file);
                    continue;
                }

                // Write Tensor length
                fwrite(&tensor->len, sizeof(size_t), 1, file);

                // Write each Matrix
                for (size_t i = 0; i < tensor->len; ++i) {
                    Matrix* mat = tensor->data[i];
                    fwrite(&mat->width, sizeof(ssize_t), 1, file);
                    fwrite(&mat->height, sizeof(ssize_t), 1, file);
                    fwrite(&mat->len, sizeof(ssize_t), 1, file);
                    fwrite(mat->data, sizeof(double), mat->len, file);
                }

                // Write Vector2D
                Vector2D* vec = tensor->dir_kernel;
                fwrite(&vec->count, sizeof(size_t), 1, file);
                fwrite(vec->sizes, sizeof(size_t), vec->count, file);
                // grid_cells
                fwrite(vec->grid_cells, sizeof(Point2D), vec->count, file);
                // data: assume a 2D layout [count][sizes[i]]
                for (size_t i = 0; i < vec->count; ++i) {
                    fwrite(vec->data[i], sizeof(Point2D), vec->sizes[i], file);
                }
            }
        }
    }

    fclose(file);
}

KernelsMap3D* deserialize_kernels_map_3d(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return NULL;
    }

    KernelsMap3D* map = (KernelsMap3D*)malloc(sizeof(KernelsMap3D));
    fread(&map->width, sizeof(ssize_t), 1, file);
    fread(&map->height, sizeof(ssize_t), 1, file);
    fread(&map->max_D, sizeof(ssize_t), 1, file);

    map->kernels = (Tensor***)malloc(map->height * sizeof(Tensor**));
    for (ssize_t y = 0; y < map->height; ++y) {
        map->kernels[y] = (Tensor**)malloc(map->width * sizeof(Tensor*));
        for (ssize_t x = 0; x < map->width; ++x) {
            map->kernels[y][x] = (Tensor*)malloc(map->max_D * sizeof(Tensor));
            for (ssize_t d = 0; d < map->max_D; ++d) {
                size_t len = 0;
                fread(&len, sizeof(size_t), 1, file);

                if (len == 0) {
                    //map->kernels[y][x][d] = NULL;
                    continue;
                }

                Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
                tensor->len = len;
                tensor->data = (Matrix**)malloc(len * sizeof(Matrix*));

                for (size_t i = 0; i < len; ++i) {
                    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
                    fread(&mat->width, sizeof(ssize_t), 1, file);
                    fread(&mat->height, sizeof(ssize_t), 1, file);
                    fread(&mat->len, sizeof(ssize_t), 1, file);
                    mat->data = (double*)malloc(mat->len * sizeof(double));
                    fread(mat->data, sizeof(double), mat->len, file);
                    tensor->data[i] = mat;
                }

                Vector2D* vec = (Vector2D*)malloc(sizeof(Vector2D));
                fread(&vec->count, sizeof(size_t), 1, file);
                vec->sizes = (size_t*)malloc(vec->count * sizeof(size_t));
                fread(vec->sizes, sizeof(size_t), vec->count, file);
                vec->grid_cells = (Point2D*)malloc(vec->count * sizeof(Point2D));
                fread(vec->grid_cells, sizeof(Point2D), vec->count, file);

                vec->data = (Point2D**)malloc(vec->count * sizeof(Point2D*));
                for (size_t i = 0; i < vec->count; ++i) {
                    vec->data[i] = (Point2D*)malloc(vec->sizes[i] * sizeof(Point2D));
                    fread(vec->data[i], sizeof(Point2D), vec->sizes[i], file);
                }

                tensor->dir_kernel = vec;
                map->kernels[y][x][d] = *tensor;
            }
        }
    }

    fclose(file);
    return map;
}


#endif //SERIALIZATION_H
