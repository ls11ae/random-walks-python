#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>

typedef struct {
    ssize_t x;
    ssize_t y;
} Point2D;

typedef struct {
    Point2D* points;
    size_t length;
} Point2DArray;

typedef struct {
    Point2DArray*** data;
    size_t width;
    size_t height;
    size_t times;
} Point2DArrayGrid;


Point2D* point_2d_new(ssize_t x, ssize_t y);

void point_2d_free(Point2D* p);

Point2DArray* point_2d_array_new(Point2D* points, size_t length);

Point2DArrayGrid* point_2d_array_grid_new(size_t width, size_t height, size_t times);

Point2DArray* point_2d_array_new_empty(size_t length);

void point2d_array_print(const Point2DArray* array);

void point2d_array_free(Point2DArray* array);

void point_2d_array_grid_free(Point2DArrayGrid* grid);

Point2DArrayGrid* load_weather_grid(const char* filename_base, int grid_y, int grid_x, int times);

#ifdef __cplusplus
}
#endif
