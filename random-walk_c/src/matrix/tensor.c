#include <stdlib.h>  // Für malloc, free, NULL
#include <stdio.h>   // Für fprintf, fwrite
#include "matrix.h"

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)
#endif
#include "../memory_utils.h"

#include "../matrix/tensor.h"

#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>

#include "math/math_utils.h"
#include "walk/c_walk.h"

Tensor* tensor_new(size_t width, size_t height, size_t depth) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "Fehler bei der Speicherzuweisung für den Tensor!\n");
        return NULL;
    }

    t->data = malloc(depth * sizeof(Matrix*));
    if (t->data == NULL) {
        fprintf(stderr, "Fehler bei der Speicherzuweisung für den Matrixpointers!\n");
        free(t);
        return NULL;
    }
    t->len = depth;
    for (size_t i = 0; i < depth; i++) {
        Matrix* m = matrix_new(width, height);
        if (m == NULL) {
            fprintf(stderr, "Fehler bei der Speicherzuweisung für den Matrix!\n");
            for (size_t d = 0; d < i; d++) {
                matrix_free(t->data[d]);
            }
            free(t->data);
            free(t);
            break;
        }
        t->data[i] = m;
    }
    return t;
}

TensorSet* tensor_set_new(const size_t count, Tensor** tensors) {
    if (!tensors || count == 0) {
        return NULL;
    }

    TensorSet* set = (TensorSet*)malloc(sizeof(TensorSet));
    if (!set) {
        return NULL;
    }

    set->data = (Tensor**)malloc(count * sizeof(Tensor*));
    set->grid_cells = (Vector2D**)malloc(count * sizeof(Vector2D*));
    if (!set->data || !set->grid_cells) {
        free(set->data);
        free(set->grid_cells);
        free(set);
        return NULL;
    }

    set->len = count;
    set->max_D = 1;

    for (size_t i = 0; i < count; i++) {
        if (!tensors[i]) {
            for (size_t j = 0; j < i; j++) {
                free(set->grid_cells[j]);
            }
            free(set->data);
            free(set->grid_cells);
            free(set);
            return NULL;
        }

        set->data[i] = tensors[i];
        set->grid_cells[i] = get_dir_kernel(tensors[i]->len, tensors[i]->data[0]->width);

        if (tensors[i]->len > set->max_D) {
            set->max_D = tensors[i]->len;
        }
    }
    return set;
}

bool tensor_equals(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        return false;
    }
    if (t1->len != t2->len) {
        return false;
    }
    for (size_t i = 0; i < t1->len; i++) {
        if (!matrix_equals(t1->data[i], t2->data[i])) {
            return false;
        }
    }
    return true;
}

void tensor_set_free(TensorSet* set) {
    if (set) {
        for (size_t i = 0; i < set->len; i++) {
            tensor_free(set->data[i]);
            free_Vector2D(set->grid_cells[i]);
        }
        free(set->data);
        free(set->grid_cells);
        free(set);
    }
}

Vector2D* get_dir_kernel(const ssize_t D, const ssize_t size) {
    Vector2D* result = (Vector2D*)malloc(sizeof(Vector2D));
    result->count = D;
    result->data = (Point2D**)malloc(D * sizeof(Point2D*));
    result->sizes = (size_t*)calloc(D, sizeof(size_t));
    result->grid_cells = (Point2D*)malloc(size * size * sizeof(Point2D));

    // First pass to count points in each direction
    size_t* counts = (size_t*)calloc(D, sizeof(size_t));
    const ssize_t S = size / 2;
    const double angle_step_size = 360.0 / (double)D;

    int index = 0;
    for (ssize_t i = -S; i <= S; ++i) {
        for (ssize_t j = -S; j <= S; ++j) {
            const Point2D point = {j, i};
            result->grid_cells[index++] = point;
            const double angle = compute_angle(j, i);
            const double closest = find_closest_angle(angle, angle_step_size);
            size_t dir = ((closest == 360.0) ? 0 : angle_to_direction(closest, angle_step_size)) % D;
            assert(dir < D);
            counts[dir]++;
        }
    }

    // Allocate memory for each direction
    for (size_t dir = 0; dir < D; dir++) {
        result->data[dir] = (Point2D*)malloc(counts[dir] * sizeof(Point2D));
        result->sizes[dir] = 0; // Reset counter for second pass
    }

    // Second pass to populate the points
    for (ssize_t i = -S; i <= S; ++i) {
        for (ssize_t j = -S; j <= S; ++j) {
            const double angle = compute_angle(j, i);
            const double closest = find_closest_angle(angle, angle_step_size);
            size_t dir = ((closest == 360.0) ? 0 : angle_to_direction(closest, angle_step_size)) % D;

            size_t idx = result->sizes[dir]++;
            result->data[dir][idx].x = j;
            result->data[dir][idx].y = i;
        }
    }

    free(counts);
    return result;
}

// Helper function to free the Vector2D when done
void free_Vector2D(Vector2D* vec) {
    if (vec == NULL) return;
    for (size_t i = 0; i < vec->count; i++) {
        point_2d_free(vec->data[i]);
    }
    free(vec->grid_cells);
    free(vec->data);
    free(vec->sizes);
}


void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    for (size_t i = 0; i < tensor->len; i++) {
        if (tensor->data && tensor->data[i] != NULL) {
            matrix_free(tensor->data[i]);
            //if (tensor->dir_kernel)
            //  free_Vector2D(tensor->dir_kernel);
        }
    }
    free(tensor->data);
    tensor->len = 0;
    tensor->data = NULL;
    free(tensor);
}


Tensor* tensor_copy(const Tensor* original) {
    if (original == NULL) {
        return NULL;
    }

    // Neuen Tensor erstellen
    Tensor* copy = tensor_new(original->data[0]->width, original->data[0]->height, original->len);
    if (copy == NULL) {
        return NULL;
    }

    // Matrixdaten kopieren
    for (size_t i = 0; i < original->len; i++) {
        memcpy(copy->data[i]->data, original->data[i]->data, sizeof(Matrix*) * original->data[i]->len);
    }

    return copy;
}


void tensor_fill(Tensor* tensor, double value) {
    for (size_t i = 0; i < tensor->len; i++) {
        matrix_fill(tensor->data[i], value);
    }
}

int tensor_in_bounds(Tensor* tensor, size_t x, size_t y, size_t z) {
    return z < tensor->len && matrix_in_bounds(tensor->data[z], x, y);
}

size_t tensor_save(Tensor* tensor, const char* foldername) {
    if (MKDIR(foldername) != 0 && errno != EEXIST) {
        perror("Error creating folder");
        return 0;
    }
    char filePathInfo[256];
    snprintf(filePathInfo, sizeof(filePathInfo), "%s/%s", foldername, "info.txt");
    FILE* file = fopen(filePathInfo, "wb"); // Open the file in binary write mode
    if (file == NULL) {
        perror("Error opening file");
        return 0;
    }
    size_t len = 0;
    len += fwrite(&tensor->len, sizeof(size_t), 1, file);

    for (int i = 0; i < tensor->len; i++) {
        char filePath[256];
        snprintf(filePath, sizeof(filePath), "%s/%d.mem", foldername, i);
        len += matrix_save(tensor->data[i], filePath);
    }

    double size_in_bytes = len * sizeof(double);

    printf("tensor_save: %s=", foldername);
    memory_size_print(size_in_bytes);
    printf("\n");

    fclose(file);
    return len;
}

Tensor* tensor_load(const char* foldername) {
    char filePathInfo[256];
    snprintf(filePathInfo, sizeof(filePathInfo), "%s/%s", foldername, "info.txt");
    FILE* file = fopen(filePathInfo, "rb"); // Open the file in binary read mode
    if (file == NULL) {
        printf("foldername: %s\n", foldername);
        perror("Error opening file");
        return NULL;
    }
    size_t len_t;
    fread(&len_t, sizeof(size_t), 1, file);

    Tensor* t = malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "Fehler bei der Speicherzuweisung für den Tensor!\n");
        fclose(file);
        return NULL;
    }
    t->len = len_t;
    t->data = malloc(t->len * sizeof(Matrix*));
    if (t->data == NULL) {
        fprintf(stderr, "Fehler bei der Speicherzuweisung für den Matirxpointers!\n");
        free(t);
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < t->len; i++) {
        char filePath[256];
        snprintf(filePath, sizeof(filePath), "%s/%d.mem", foldername, i);
        Matrix* m = matrix_load(filePath);
        if (m == NULL) {
            fprintf(stderr, "Fehler bei der Speicherzuweisung für den Matrix!\n");
            for (size_t d = 0; d < i; d++) {
                matrix_free(t->data[d]);
            }
            free(t->data);
            free(t);
            fclose(file);
            break;
        }
        t->data[i] = m;
    }
    fclose(file);
    return t;
}

void* tensor4D_new(size_t width, size_t height, size_t depth);

void tensor4D_free(Tensor** tensor, ssize_t T) {
    for (ssize_t i = 0; i < T; ++i)
        tensor_free(tensor[i]);
    free(tensor);
}

void tensor4D_fill(Tensor* tensor, double value);

int tensor4D_in_bounds(Tensor* tensor, size_t x, size_t y, size_t z);
