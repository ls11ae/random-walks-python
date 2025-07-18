#include "math/kernel_slicing.h"

#define SAMPLES_PER_SIDE 200
#define PI 3.14159265358979323846

double compute_angle_ks(double x, double y) {
    if (x == 0.0 && y == 0.0) return 0.0;
    double radians = atan2(y, x);
    double degrees = radians * 180.0 / M_PI;
    if (degrees < 0.0) {
        degrees += 360.0;
    }
    return degrees;
}

void compute_overlap_percentages(int W, int D, Tensor* tensor) {
    const int S = W / 2;
    const double angle_step = 360.0 / D;

    // Allocate tensor data for D matrices
    tensor->len = D;
    tensor->data = (Matrix**)malloc(D * sizeof(Matrix*));
    if (!tensor->data) return;

    for (int d = 0; d < D; ++d) {
        Matrix* m = (Matrix*)malloc(sizeof(Matrix));
        if (!m) {
            // Cleanup previous allocations on failure
            for (int i = 0; i < d; ++i) {
                free(tensor->data[i]->data);
                free(tensor->data[i]);
            }
            free(tensor->data);
            tensor->data = NULL;
            return;
        }
        m->width = W;
        m->height = W;
        m->len = W * W;
        m->data = (double*)malloc(W * W * sizeof(double));
        if (!m->data) {
            free(m);
            // Cleanup previous allocations on failure
            for (int i = 0; i < d; ++i) {
                free(tensor->data[i]->data);
                free(tensor->data[i]);
            }
            free(tensor->data);
            tensor->data = NULL;
            return;
        }
        memset(m->data, 0, W * W * sizeof(double));
        tensor->data[d] = m;
    }

    const int steps = 100;

#pragma omp parrallel for collapse(2) schedule(dynamic)
    for (int x_center = -S; x_center <= S; ++x_center) {
        for (int y_center = -S; y_center <= S; ++y_center) {
            int grid_x = x_center + S;
            int grid_y = y_center + S;
            for (int d = 0; d < D; ++d) {
                double center_angle = d * angle_step;
                double start_angle = center_angle - angle_step / 2.0;
                double end_angle = center_angle + angle_step / 2.0;

                // Normalize angles to [0, 360)
                start_angle = fmod(start_angle, 360.0);
                if (start_angle < 0) start_angle += 360.0;
                end_angle = fmod(end_angle, 360.0);
                if (end_angle < 0) end_angle += 360.0;

                int count = 0;
                for (int dx = 0; dx < steps; ++dx) {
                    double x = x_center - 0.5 + (dx + 0.5) / steps;
                    for (int dy = 0; dy < steps; ++dy) {
                        double y = y_center - 0.5 + (dy + 0.5) / steps;
                        double theta = compute_angle_ks(x, y);

                        bool inside = false;
                        if (start_angle < end_angle) {
                            inside = (theta >= start_angle) && (theta < end_angle);
                        }
                        else {
                            inside = (theta >= start_angle) || (theta < end_angle);
                        }

                        if (inside) count++;
                    }
                }

                double overlap = (double)count / (steps * steps);
                matrix_set(tensor->data[d], grid_x, grid_y, overlap);
            }
        }
    }
}
