#include "path_finding.h"

#include <stdlib.h>

// Bresenham's line algorithm
static int is_path_clear(const TerrainMap* terrain, ssize_t x0, ssize_t y0, ssize_t x1, ssize_t y1) {
    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;

    ssize_t current_x = x0;
    ssize_t current_y = y0;
    int is_first = 1;

    while (1) {
        if (!is_first) {
            if (current_x == x1 && current_y == y1) {
                break;
            }
            if (current_x < 0 || current_x >= terrain->width || current_y < 0 || current_y >= terrain->height) {
                return 0;
            }
            if (terrain_at(current_x, current_y, terrain) == WATER) {
                return 0;
            }
        }
        else {
            is_first = 0;
        }

        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            if (current_x == x1) break;
            error += dy;
            current_x += sx;
        }
        if (e2 <= dx) {
            if (current_y == y1) break;
            error += dx;
            current_y += sy;
        }
    }

    return 1;
}

Matrix* get_reachability_kernel(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                const TerrainMap* terrain) {
    Matrix* result = matrix_new(kernel_size, kernel_size);

    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height) {
        return result;
    }
    if (terrain_at(x, y, terrain) == WATER) {
        return result;
    }

    const ssize_t kernel_center_x = (kernel_size) / 2;
    const ssize_t kernel_center_y = (kernel_size) / 2;

    bool full_reachable = true;
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) {
                continue;
            }

            if (terrain_at(new_x, new_y, terrain) == WATER) {
                full_reachable = false;
                break;
            }
        }
        if (!full_reachable) {
            break;
        }
    }

    if (full_reachable) {
        matrix_fill(result, 1.0);
        return result;
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;

            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;

            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) {
                continue;
            }

            if (terrain_at(new_x, new_y, terrain) == WATER) {
                continue;
            }

            if (is_path_clear(terrain, x, y, new_x, new_y)) {
                matrix_set(result, (size_t)i, (size_t)j, 1.0);
            }
        }
    }

    return result;
}
