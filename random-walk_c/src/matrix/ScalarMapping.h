/**
* @file scalar_mapping.h
 * @brief Header file for integer 2D points
 *
 * This library provides functions for creating, manipulating, and saving f64, int pairs
 *
 * @authors [Christian Miklar, Omar Chatila]
 *
 * @version 1.0.0
 * @date 2025-01-16
 *
 * @details
 * This header defines the ScalarMapping structure and its associated functions, such as:
 * - Creating and freeing matrices
 * - Basic mathematical operations (e.g., determinant, inversion)
 * - Input/output utilities for matrices
 *
 * Example:
 * @code
 *
 *
 *
 *
 *
 *
 * @endcode
 *
 * @see scalar_mapping.c for implementation details.
 */

#ifndef scalar_mapping_H
#define scalar_mapping_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdlib.h>  // Für malloc, free, NULL


    typedef struct {
        double value;
        size_t index;
    } ScalarMapping;

    typedef ScalarMapping *scalar_mappingRef;

    ScalarMapping *scalar_mapping_new(double x, size_t y);

    void set_values(ScalarMapping *point, double x, size_t y);

    void scalar_mapping_delete(ScalarMapping *self);

#ifdef __cplusplus
}
#endif //scalar_mapping_H
#endif
