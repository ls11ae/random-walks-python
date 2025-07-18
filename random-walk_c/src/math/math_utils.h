#pragma once
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x;
    int y;
} Point;

// Funktion, um einen Punkt um einen beliebigen Mittelpunkt (offset_x, offset_y) zu drehen
Point rotate_point(Point p, double theta);

ssize_t weighted_random_index(const double *array, size_t length);

double to_radians(double angle);

double compute_angle(ssize_t x, ssize_t y);

size_t angle_to_direction(double angle, double angle_step_size);

double find_closest_angle(double angle, double angle_step_size);

double alpha(int i, int j, double rotation_angle);

double euclid(ssize_t f_x, ssize_t f_y, ssize_t s_x, ssize_t s_y);

double euclid_origin(int i, int j);

double euclid_sqr(ssize_t point1_x, ssize_t point1_y, ssize_t point2_x, ssize_t point2_y);
#ifdef __cplusplus
}
#endif
