//
// Created by omar on 24.03.25.
//
#include "Point2D.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/move_bank_parser.h"
#include "parsers/types.h"

Point2D* point_2d_new(const ssize_t x, const ssize_t y) {
    Point2D* result = malloc(sizeof(Point2D));
    result->x = x;
    result->y = y;
    return result;
}

void point_2d_free(Point2D* p) {
    free(p);
}


Point2DArray* point_2d_array_new(Point2D* points, size_t length) {
    Point2DArray* result = (Point2DArray*)malloc(sizeof(Point2DArray));
    if (!result) return NULL;

    result->points = (Point2D*)malloc(length * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }

    // Copy data from input `points` to the new array
    memcpy(result->points, points, length * sizeof(Point2D)); // <-- Critical fix

    result->length = length;
    return result;
}

Point2DArray* point_2d_array_new_empty(size_t length) {
    Point2DArray* result = (Point2DArray*)malloc(sizeof(Point2DArray));
    if (!result) return NULL;

    result->points = (Point2D*)malloc(length * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }

    result->length = length;
    return result;
}

Point2DArrayGrid* point_2d_array_grid_new(size_t width, size_t height, size_t times) {
    Point2DArrayGrid* result = (Point2DArrayGrid*)malloc(sizeof(Point2DArrayGrid));
    if (!result) return NULL;

    Point2DArray*** data = (Point2DArray***)malloc(sizeof(Point2DArray**) * height);
    if (!data) {
        free(result);
        return NULL;
    }

    for (size_t i = 0; i < height; i++) {
        data[i] = (Point2DArray**)malloc(sizeof(Point2DArray*) * width);
        if (!data[i]) {
            // Cleanup previously allocated memory
            for (size_t k = 0; k < i; k++) {
                for (size_t j = 0; j < width; j++) {
                    point2d_array_free(data[k][j]);
                }
                free(data[k]);
            }
            free(data);
            free(result);
            return NULL;
        }

        for (size_t j = 0; j < width; j++) {
            data[i][j] = point_2d_array_new_empty(times);
            if (!data[i][j]) {
                // Cleanup previously allocated memory
                for (size_t k = 0; k <= i; k++) {
                    for (size_t l = 0; l < (k == i ? j : width); l++) {
                        point2d_array_free(data[k][l]);
                    }
                    free(data[k]);
                }
                free(data);
                free(result);
                return NULL;
            }
        }
    }

    result->height = height;
    result->width = width;
    result->data = data;
    return result;
}

// Print all points in the Point2DArray
void point2d_array_print(const Point2DArray* array) {
    if (!array || !array->points) {
        printf("Invalid Point2DArray\n");
        fflush(stdout);
        return;
    }
    printf("%zu\n", array->length);
    for (size_t i = 0; i < array->length; ++i) {
        printf("(%zd, %zd),\n", array->points[i].x, array->points[i].y);
        fflush(stdout);
    }
}

// Free the Point2DArray and its internal points array
void point2d_array_free(Point2DArray* array) {
    if (array) {
        free(array->points); // Free the points data
        free(array); // Free the struct itself
    }
}

void point_2d_array_grid_free(Point2DArrayGrid* grid) {
    if (!grid) return;

    for (size_t i = 0; i < grid->height; i++) {
        for (size_t j = 0; j < grid->width; j++) {
            point2d_array_free(grid->data[i][j]);
        }
        free(grid->data[i]);
    }

    free(grid->data);
    free(grid);
}

char* read_file_to_string(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    long len = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(len + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, len, file);
    buffer[len] = '\0';
    fclose(file);
    return buffer;
}

Point2DArray* bias_from_csv(const char* file_content, ssize_t max_bias) {
    // Parse CSV content
    int num_entries;
    WeatherEntry* entries = parse_csv(file_content, &num_entries);

    Point2D* points2 = malloc(sizeof(Point2D) * num_entries);
    for (int i = 0; i < num_entries; i++) {
        points2[i] = *weather_entry_to_bias(&entries[i], max_bias);
    }
    Point2DArray* biases = point_2d_array_new(points2, num_entries);
    return biases;
}

Point2DArrayGrid* load_weather_grid(const char* filename_base, int grid_y, int grid_x, int times) {
    Point2DArrayGrid* grid = point_2d_array_grid_new(grid_y, grid_x, times);
    if (!grid) return NULL;

    char filename[512];

    for (int i = 0; i < grid_y; ++i) {
        for (int j = 0; j < grid_x; ++j) {
            snprintf(filename, sizeof(filename), "%s/weather_grid_y%d_x%d.csv", filename_base, i, j);
            char* file_content = read_file_to_string(filename);
            if (!file_content) {
                fprintf(stderr, "Failed to open or read file: %s\n", filename);
                return NULL;
            }

            Point2DArray* biases = bias_from_csv(file_content, 5);
            free(file_content);

            grid->data[i][j] = biases;
        }
    }

    return grid;
}
