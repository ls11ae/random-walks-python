#include <math.h>  // for atan2, round, and M_PI
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

ssize_t weighted_random_index(const double* array, size_t len) {
    // Seed the random number generator with the current time
    static int seeded = 0;
    if (!seeded) {
        srand(((unsigned int)time(NULL))); // Seed only once
        seeded = 1;
    }

    const ssize_t length = (ssize_t)len;

    // Calculate the total weight (CDF)
    double total_weight = 0.0;
    for (size_t i = 0; i < length; i++) {
        total_weight += array[i];
    }

    // Generate a random value between 0 and total_weight
    double random_value = (rand() / (double)RAND_MAX) * total_weight;

    // Find the index where the cumulative sum exceeds the random value
    double cumulative_sum = 0.0;
    for (ssize_t i = 0; i < length; i++) {
        cumulative_sum += array[i];
        if (cumulative_sum >= random_value) {
            return i; // Return the index
        }
    }

    // If no index is found, return the last index (in case of very small values)
    return length - 1;
}

Point rotate_point(Point p, double theta) {
    Point result;

    // Berechne den Cosinus und Sinus des Winkels in double
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    // Drehe den Punkt
    result.x = (int)(p.x * cos_theta - p.y * sin_theta);
    result.y = (int)(p.x * sin_theta + p.y * cos_theta);

    return result;
}

double to_radians(const double angle) {
    return angle * M_PI / 180;
}

double compute_angle(ssize_t dx, ssize_t dy) {
    if (dx == 0 && dy == 0) return 0.0; // Handle zero vector

    double radians = atan2(dy, dx);
    double degrees = radians * 180.0 / M_PI;
    // Adjust to 0-360 range
    if (degrees < 0) {
        degrees += 360.0;
    }
    return degrees;
}

size_t angle_to_direction(double angle, double angle_step_size) {
    return (size_t)round(angle / angle_step_size) % ((size_t)(360.0 / angle_step_size));
}


double find_closest_angle(double angle, double angle_step_size) {
    int steps = (int)(360.0 / angle_step_size);
    int num_angles = steps + 1;
    double* angles = (double*)malloc(num_angles * sizeof(double));
    if (!angles) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < steps; ++i) {
        angles[i] = i * angle_step_size;
    }
    angles[steps] = 360.0;

    double closest_angle = angles[0];
    double min_diff = fabs(angles[0] - angle);

    for (int j = 1; j < num_angles; ++j) {
        double current_diff = fabs(angles[j] - angle);
        if (current_diff < min_diff) {
            min_diff = current_diff;
            closest_angle = angles[j];
        }
    }

    free(angles);
    return closest_angle;
}

double alpha(int i, int j, double rotation_angle) {
    double original_alpha = atan2(j, i);
    return original_alpha - rotation_angle;
}

double euclid(ssize_t point1_x, ssize_t point1_y, ssize_t point2_x, ssize_t point2_y) {
    const double delta_x = (double)(point2_x - point1_x);
    const double delta_y = (double)(point2_y - point1_y);
    return sqrt(delta_x * delta_x + delta_y * delta_y);
}

double euclid_sqr(ssize_t point1_x, ssize_t point1_y, ssize_t point2_x, ssize_t point2_y) {
    double delta_x = point2_x - point1_x;
    double delta_y = point2_y - point1_y;
    return delta_x * delta_x + delta_y * delta_y;
}

double euclid_origin(const int i, const int j) {
    return sqrt(i * i + j * j);
}
