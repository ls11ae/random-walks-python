/**
 * @file matrix.h
 * @brief Header file for basic matrix operations and utilities.
 *
 * This library provides functions for creating, manipulating, and saving matrices
 * used in scientific computations. It is compatible with C99 and C++11 or later.
 *
 * @authors [Christian Miklar, Omar Chatila]
 *
 * @version 1.0.0
 * @date 2024-12-07
 *
 * @details
 * This header defines the Matrix structure and its associated functions, such as:
 * - Creating and freeing matrices
 * - Basic mathematical operations (e.g., determinant, inversion)
 * - Input/output utilities for matrices
 *
 * Example:
 * @code
 * Matrix *mat = matrix_new(3, 3);
 * matrix_fill(mat, 1.0);
 * matrix_save(mat, "output.mat");
 * matrix_free(mat);
 * mat = matrix_load("output.mat");
 * matrix_free(mat);
 * @endcode
 *
 * @see matrix.c for implementation details.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>


/**
 * @struct Matrix
 * @brief Represents a 2D matrix.
 */
typedef struct {
    ssize_t width; /**< The number of columns in the matrix. */
    ssize_t height; /**< The number of rows in the matrix. */
    ssize_t len; /**< The total number of elements (width * height). */
    double* data; /**< Pointer to the array of matrix elements. */
} Matrix;

Matrix* matrix_new(ssize_t width, ssize_t height);

void matrix_free(Matrix* matrix);

Matrix* matrix_copy(const Matrix* matrix);

void matrix_convolution(Matrix* input, Matrix* kernel, Matrix* output);

bool matrix_equals(const Matrix* matrix1, const Matrix* matrix2);

void matrix_pooling_avg(Matrix* dst, const Matrix* src);

void matrix_copy_to(Matrix* dest, const Matrix* src);

void get_gaussian_parameters(float diffusity, int terrain_value,
                             float* out_sigma, float* out_scale);

int matrix_in_bounds(const Matrix* m, size_t x, size_t y);

double matrix_get(const Matrix* m, size_t x, size_t y);

void matrix_set(const Matrix* m, size_t x, size_t y, double val);

void matrix_fill(Matrix* matrix, double value);

Matrix* matrix_add(const Matrix* a, const Matrix* b);

int matrix_add_inplace(Matrix* a, const Matrix* b);

Matrix* matrix_sub(const Matrix* a, const Matrix* b);

Matrix* matrix_mul(const Matrix* a, const Matrix* b);

Matrix* matrix_elementwise_mul(const Matrix* a, const Matrix* b);

double matrix_sum(const Matrix* matrix);

void matrix_transpose(Matrix* m);

double matrix_determinant(const Matrix* mat);

Matrix* matrix_invert(const Matrix* input);

void matrix_print(const Matrix* m);

char* matrix_to_string(const Matrix* mat);

size_t matrix_save(const Matrix* mat, const char* filename);

Matrix* matrix_load(const char* filename);

void matrix_normalize(const Matrix* mat, double sum);

void matrix_normalize_01(Matrix* m);

void matrix_normalize_L1(Matrix* m);

Matrix* matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                      ssize_t y_offset);

Matrix* matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                  ssize_t y_offset);

#ifdef __cplusplus
}
#endif
