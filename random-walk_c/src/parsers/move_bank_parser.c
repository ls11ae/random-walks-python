#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "move_bank_parser.h"  // Include the header with Point2D and Point2DArray definitions

#include <assert.h>
#include <math.h>


Coordinate_array* coordinate_array_new(Coordinate* coordinates, size_t length) {
    Coordinate_array* result = (Coordinate_array*)malloc(sizeof(Coordinate_array));
    if (!result) return NULL;

    result->points = (Coordinate*)malloc(length * sizeof(Coordinate));
    if (!result->points) {
        free(result);
        return NULL;
    }

    // Copy data from input `points` to the new array
    memcpy(result->points, coordinates, length * sizeof(Coordinate));

    result->length = length;
    return result;
}

Coordinate_array* extractLocationsFromCSV(const char* csv_file_path, const char* animal_id) {
    FILE* file = fopen(csv_file_path, "r");
    if (!file) {
        printf("Could not open file %s\n", csv_file_path);
        return NULL;
    }

    // Skip the header line
    char line[1024];
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return NULL;
    }

    Coordinate* points = NULL;
    size_t capacity = 0;
    size_t count = 0;

    while (fgets(line, sizeof(line), file)) {
        Coordinate point = {0, 0}; // Initialize to zero
        char line_copy[1024];
        strncpy(line_copy, line, sizeof(line_copy));
        line_copy[sizeof(line_copy) - 1] = '\0'; // Ensure null-termination

        int column = 0;
        char* token = strtok(line_copy, ",");
        while (token) {
            if (column == 3 || column == 4) {
                char* endptr;
                errno = 0;
                double val = strtod(token, &endptr);

                if (endptr != token && errno != ERANGE) {
                    if (column == 3) {
                        point.x = val;
                    }
                    else {
                        point.y = val;
                    }
                }
            }

            token = strtok(NULL, ",");
            column++;
        }

        // Add point to dynamic array
        if (count >= capacity) {
            size_t new_capacity = (capacity == 0) ? 16 : capacity * 2;
            Coordinate* new_points = (Coordinate*)realloc(points, new_capacity * sizeof(Coordinate));
            if (!new_points) {
                free(points);
                fclose(file);
                return NULL;
            }
            points = new_points;
            capacity = new_capacity;
        }
        points[count++] = point;
    }

    fclose(file);

    // Create and return Point2DArray
    Coordinate_array* result = coordinate_array_new(points, count);
    free(points); // Free temporary buffer after copying (adjust if needed)
    printf("successfully created coordinate array\n");

    return result;
}

Point2DArray* getNormalizedLocations(const Coordinate_array* path, const size_t W, const size_t H) {
    if (path->length == 0) return NULL;
    printf("normalizing locations\n");

    // Find the min and max values for x and y
    double minX = path->points[0].x, maxX = path->points[0].x;
    double minY = path->points[0].y, maxY = path->points[0].y;

    for (size_t i = 0; i < path->length; ++i) {
        const double x = path->points[i].x, y = path->points[i].y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    // Normalize each point to the range [0, W] and [0, H]
    Point2DArray* normalizedPath = (Point2DArray*)malloc(sizeof(Point2DArray));
    Point2D* points = (Point2D*)malloc(path->length * sizeof(Point2D));
    normalizedPath->points = points;
    normalizedPath->length = path->length;

    for (size_t i = 0; i < path->length; ++i) {
        const double x = path->points[i].x;
        const double y = path->points[i].y;

        const ssize_t normalizedX = (ssize_t)((x - minX) / (maxX - minX) * (double)W);
        const ssize_t normalizedY = (ssize_t)((y - minY) / (maxY - minY) * (double)H);

        const Point2D normalizedPoint = {normalizedX, normalizedY};
        normalizedPath->points[i] = normalizedPoint;
    }

    return normalizedPath;
}

Point2DArray* extractSteps(Point2DArray* path, const size_t step_count) {
    const size_t delta = (path->length - 1) / step_count;
    Point2DArray* gap_path = (Point2DArray*)malloc(sizeof(Point2DArray));
    gap_path->points = (Point2D*)malloc(step_count * sizeof(Point2D));
    gap_path->length = step_count;
    for (int i = 0; i < step_count - 1; i++) {
        gap_path->points[i] = path->points[i * delta];
    }
    gap_path->points[gap_path->length - 1] = path->points[path->length - 1];
    return gap_path;
}

void coordinate_array_free(Coordinate_array* coordinate_array) {
    if (coordinate_array) {
        free(coordinate_array->points);
        free(coordinate_array);
    }
}

KernelParameters* kernel_parameters_new(int terrain_value, WeatherEntry* weather_entry) {
    // KernelParameters* params = malloc(sizeof(KernelParameters));
    // if (!params) {
    //     perror("Failed to allocate memory for KernelParameters");
    //     return NULL;
    // }
    // float base_step_multiplier;
    // switch (terrain_value) {
    // case TREE_COVER: // Value 10
    //     params->is_brownian = 1; // Correlated (paths, navigating around trees)
    //     params->D = 1; // More restricted directions
    //     params->diffusity = 0.9f; // Dense, slow spread
    //     base_step_multiplier = 0.7f; // Small steps
    //     break;
    // case SHRUBLAND: // Value 20
    //     params->is_brownian = 0; // Correlated
    //     params->D = 8; // Fairly open for navigation
    //     params->diffusity = 0.8f; // Moderately slow spread
    //     base_step_multiplier = 0.5f; // Moderate steps
    //     break;
    // case GRASSLAND: // Value 30
    //     params->is_brownian = 1; // Correlated
    //     params->D = 1; // Open movement
    //     params->diffusity = 1.0f; // Easy spread
    //     base_step_multiplier = 1.0f; // Standard steps
    //     break;
    // case CROPLAND: // Value 40
    //     params->is_brownian = 0; // Correlated (movement along rows/edges)
    //     params->D = 4; // Structured movement
    //     params->diffusity = 1.2f; // Moderate spread
    //     base_step_multiplier = 0.7f; // Moderate steps, possible obstacles
    //     break;
    // case BUILT_UP: // Value 50
    //     params->is_brownian = 0; // Correlated (streets, paths)
    //     params->D = 4; // Grid-like or defined paths
    //     params->diffusity = 0.7f; // Many obstacles, slow overall spread
    //     base_step_multiplier = 0.6f; // Smaller steps due to structure
    //     break;
    // case SPARSE_VEGETATION: // Value 60 (Desert-like, open)
    //     params->is_brownian = 0; // Correlated
    //     params->D = 8; // Very open
    //     params->diffusity = 2.5f; // Very easy spread
    //     base_step_multiplier = 1.2f; // Larger steps possible
    //     break;
    // case SNOW_AND_ICE: // Value 70
    //     params->is_brownian = 1; // Brownian (slippery, difficult to maintain course, or deep snow)
    //     params->D = 1; // Convention for Brownian
    //     params->diffusity = 0.4f; // Difficult, slow spread
    //     base_step_multiplier = 0.3f; // Small, careful steps
    //     break;
    // case WATER: // Value 80 (Assuming terrestrial agent, difficult to traverse)
    //     params->is_brownian = 1; // Brownian (swimming/wading difficult without aid)
    //     params->D = 1;
    //     params->diffusity = 0.1f; // Very slow spread/progress
    //     base_step_multiplier = 0.1f; // Very small progress
    //     break;
    // case HERBACEOUS_WETLAND: // Value 90 (Marshes, bogs)
    //     params->is_brownian = 1; // Brownian (slogging, difficult to keep direction)
    //     params->D = 1;
    //     params->diffusity = 0.3f; // Slow spread due to terrain
    //     base_step_multiplier = 0.2f; // Small steps
    //     break;
    // case MANGROVES: // Value 95
    //     params->is_brownian = 1; // Brownian (extremely dense, roots, water)
    //     params->D = 1;
    //     params->diffusity = 0.2f; // Very difficult to move/spread
    //     base_step_multiplier = 0.15f; // Very small, difficult steps
    //     break;
    // case MOSS_AND_LICHEN: // Value 100 (Tundra-like, uneven ground)
    //     params->is_brownian = 1; // Correlated (can navigate but ground may be tricky)
    //     params->D = 8; // Generally open directionally
    //     params->diffusity = 1.0f; // Moderate spread
    //     base_step_multiplier = 0.6f; // Moderate steps, accounting for unevenness
    //     break;
    // default: // Handle unknown terrain_value
    //     // fprintf(stderr, "Warning: Unknown terrain_value %d, using default fallback parameters.\n", terrain_value);
    //     params->is_brownian = 1; // Default to Brownian for unknown/unpredictable terrain
    //     params->D = 1;
    //     params->diffusity = 0.7f; // Assume moderate difficulty
    //     base_step_multiplier = 0.5f; // Assume moderate steps
    //     break;
    // }
    // // Calculate final step size
    // float initial_base_step = 3.0f; // An initial reference step size before terrain modification
    // float calculated_step = initial_base_step * base_step_multiplier;
    // params->S = (ssize_t)fmaxf(roundf(calculated_step), 3.0f); // Ensure step size is at least 3
    //
    // // 6. Calculate wind-driven bias (corrected coordinate system)
    // const float wind_dir_rad = weather_entry->wind_direction * (M_PI / 180.0f);
    // const float bias_x = weather_entry->wind_speed * sinf(wind_dir_rad);
    // const float bias_y = weather_entry->wind_speed * cosf(wind_dir_rad);
    //
    // // Kernel dimensions (assuming kernel is square, adjust if rectangular)
    // const ssize_t kernel_radius = (params->S - 1) / 2;
    // const float max_bias = (float)kernel_radius / 4;
    //
    // params->bias_x = (ssize_t)fmaxf(-max_bias, fminf(bias_x, max_bias));
    // params->bias_y = (ssize_t)fmaxf(-max_bias, fminf(bias_y, max_bias));
    //
    // return params;
    KernelParameters* params = malloc(sizeof(KernelParameters));
    if (!params) return NULL;
    // 5. Calculate diffusivity (environment openness + wind impact)
    params->diffusity = params->is_brownian ? 0.5f : 1.5f;
    params->diffusity += weather_entry->wind_speed * 0.05f;
    // 2. Calculate base step size based on terrain
    float base_step = 5.0f;
    float base_step_multiplier;
    switch (terrain_value) {
    case TREE_COVER: // Value 10
        params->is_brownian = 1; // Correlated (paths, navigating around trees)
        params->D = 1; // More restricted directions
        params->diffusity = 0.5f; // Dense, slow spread
        base_step_multiplier = 0.8f; // Small steps
        break;
    case SHRUBLAND: // Value 20
        params->is_brownian = 0; // Correlated
        params->D = 8; // Fairly open for navigation
        params->diffusity = 0.8f; // Moderately slow spread
        base_step_multiplier = 1.5f; // Moderate steps
        break;
    case GRASSLAND: // Value 30
        params->is_brownian = 1; // Correlated
        params->D = 1; // Open movement
        params->diffusity = 2.0f; // Easy spread
        base_step_multiplier = 1.0f; // Standard steps
        break;
    case CROPLAND: // Value 40
        params->is_brownian = 1; // Correlated (movement along rows/edges)
        params->D = 1; // Structured movement
        params->diffusity = 1.2f; // Moderate spread
        base_step_multiplier = 1.7f; // Moderate steps, possible obstacles
        break;
    case BUILT_UP: // Value 50
        params->is_brownian = 0; // Correlated (streets, paths)
        params->D = 4; // Grid-like or defined paths
        params->diffusity = 0.7f; // Many obstacles, slow overall spread
        base_step_multiplier = 1.6f; // Smaller steps due to structure
        break;
    case SPARSE_VEGETATION: // Value 60 (Desert-like, open)
        params->is_brownian = 0; // Correlated
        params->D = 8; // Very open
        params->diffusity = 2.5f; // Very easy spread
        base_step_multiplier = 1.2f; // Larger steps possible
        break;
    case SNOW_AND_ICE: // Value 70
        params->is_brownian = 1; // Brownian (slippery, difficult to maintain course, or deep snow)
        params->D = 1; // Convention for Brownian
        params->diffusity = 0.4f; // Difficult, slow spread
        base_step_multiplier = 1.3f; // Small, careful steps
        break;
    case WATER: // Value 80 (Assuming terrestrial agent, difficult to traverse)
        params->is_brownian = 1; // Brownian (swimming/wading difficult without aid)
        params->D = 1;
        params->diffusity = 0.1f; // Very slow spread/progress
        base_step_multiplier = 1.1f; // Very small progress
        break;
    case HERBACEOUS_WETLAND: // Value 90 (Marshes, bogs)
        params->is_brownian = 1; // Brownian (slogging, difficult to keep direction)
        params->D = 1;
        params->diffusity = 0.3f; // Slow spread due to terrain
        base_step_multiplier = 1.2f; // Small steps
        break;
    case MANGROVES: // Value 95
        params->is_brownian = 1; // Brownian (extremely dense, roots, water)
        params->D = 1;
        params->diffusity = 0.2f; // Very difficult to move/spread
        base_step_multiplier = 1.15f; // Very small, difficult steps
        break;
    case MOSS_AND_LICHEN: // Value 100 (Tundra-like, uneven ground)
        params->is_brownian = 0; // Correlated (can navigate but ground may be tricky)
        params->D = 4; // Generally open directionally
        params->diffusity = 1.0f; // Moderate spread
        base_step_multiplier = 1.6f; // Moderate steps, accounting for unevenness
        break;
    default: // Handle unknown terrain_value
        // fprintf(stderr, "Warning: Unknown terrain_value %d, using default fallback parameters.\n", terrain_value);
        params->is_brownian = 1; // Default to Brownian for unknown/unpredictable terrain
        params->D = 1;
        params->diffusity = 0.7f; // Assume moderate difficulty
        base_step_multiplier = 1.5f; // Assume moderate steps
        break;
    }
    // 6. Calculate wind-driven bias (corrected coordinate system)
    const float wind_dir_rad = weather_entry->wind_direction * (M_PI / 180.0f);
    const float bias_x = weather_entry->wind_speed * sinf(wind_dir_rad);
    const float bias_y = weather_entry->wind_speed * cosf(wind_dir_rad);
    // Kernel dimensions (assuming kernel is square, adjust if rectangular)
    const ssize_t kernel_radius = (params->S - 1) / 2;
    const float max_bias = (float)kernel_radius / 2;
    params->bias_x = (ssize_t)fmax(-max_bias, fmin(bias_x, max_bias));
    params->bias_y = (ssize_t)fmax(-max_bias, fmin(bias_y, max_bias));
    // params->bias_x = 0; //(ssize_t)fmax(-max_bias, fmin(bias_x, max_bias));
    // params->bias_y = 0; //(ssize_t)fmax(-max_bias, fmin(bias_y, max_bias));

    params->S = (ssize_t)(base_step * base_step_multiplier);

    // 7. Handle extreme weather conditions (optional)
    // if (weather_entry->weather_code >= 95) {
    //     params->S = 0;
    //     params->diffusity = 0.0f;
    //     params->bias_x = 0;
    //     params->bias_y = 0;
    // }

    return params;
}

KernelParameters* kernel_parameters_terrain(int terrain_value) {
    KernelParameters* params = malloc(sizeof(KernelParameters));
    if (!params) {
        perror("Failed to allocate memory for KernelParameters");
        return NULL;
    }

    float base_step_multiplier;

    switch (terrain_value) {
    case TREE_COVER: // Value 10
        params->is_brownian = 1; // Correlated (paths, navigating around trees)
        params->D = 1; // More restricted directions
        params->diffusity = 0.9f; // Dense, slow spread
        base_step_multiplier = 0.7f; // Small steps
        break;
    case SHRUBLAND: // Value 20
        params->is_brownian = 0; // Correlated
        params->D = 8; // Fairly open for navigation
        params->diffusity = 0.8f; // Moderately slow spread
        base_step_multiplier = 0.5f; // Moderate steps
        break;
    case GRASSLAND: // Value 30
        params->is_brownian = 1; // Correlated
        params->D = 1; // Open movement
        params->diffusity = 1.0f; // Easy spread
        base_step_multiplier = 1.0f; // Standard steps
        break;
    case CROPLAND: // Value 40
        params->is_brownian = 0; // Correlated (movement along rows/edges)
        params->D = 8; // Structured movement
        params->diffusity = 1.2f; // Moderate spread
        base_step_multiplier = 0.7f; // Moderate steps, possible obstacles
        break;
    case BUILT_UP: // Value 50
        params->is_brownian = 0; // Correlated (streets, paths)
        params->D = 4; // Grid-like or defined paths
        params->diffusity = 0.7f; // Many obstacles, slow overall spread
        base_step_multiplier = 0.6f; // Smaller steps due to structure
        break;
    case SPARSE_VEGETATION: // Value 60 (Desert-like, open)
        params->is_brownian = 0; // Correlated
        params->D = 8; // Very open
        params->diffusity = 2.5f; // Very easy spread
        base_step_multiplier = 0.8f; // Larger steps possible
        break;
    case SNOW_AND_ICE: // Value 70
        params->is_brownian = 1; // Brownian (slippery, difficult to maintain course, or deep snow)
        params->D = 1; // Convention for Brownian
        params->diffusity = 0.4f; // Difficult, slow spread
        base_step_multiplier = 0.3f; // Small, careful steps
        break;
    case WATER: // Value 80 (Assuming terrestrial agent, difficult to traverse)
        params->is_brownian = 1; // Brownian (swimming/wading difficult without aid)
        params->D = 1;
        params->diffusity = 0.1f; // Very slow spread/progress
        base_step_multiplier = 0.1f; // Very small progress
        break;
    case HERBACEOUS_WETLAND: // Value 90 (Marshes, bogs)
        params->is_brownian = 1; // Brownian (slogging, difficult to keep direction)
        params->D = 1;
        params->diffusity = 0.3f; // Slow spread due to terrain
        base_step_multiplier = 0.2f; // Small steps
        break;
    case MANGROVES: // Value 95
        params->is_brownian = 1; // Brownian (extremely dense, roots, water)
        params->D = 1;
        params->diffusity = 0.2f; // Very difficult to move/spread
        base_step_multiplier = 0.15f; // Very small, difficult steps
        break;
    case MOSS_AND_LICHEN: // Value 100 (Tundra-like, uneven ground)
        params->is_brownian = 1; // Correlated (can navigate but ground may be tricky)
        params->D = 8; // Generally open directionally
        params->diffusity = 1.0f; // Moderate spread
        base_step_multiplier = 0.6f; // Moderate steps, accounting for unevenness
        break;
    default: // Handle unknown terrain_value
        // fprintf(stderr, "Warning: Unknown terrain_value %d, using default fallback parameters.\n", terrain_value);
        params->is_brownian = 1; // Default to Brownian for unknown/unpredictable terrain
        params->D = 1;
        params->diffusity = 0.7f; // Assume moderate difficulty
        base_step_multiplier = 0.5f; // Assume moderate steps
        break;
    }

    // Calculate final step size
    float initial_base_step = 9.0f; // An initial reference step size before terrain modification
    float calculated_step = initial_base_step * base_step_multiplier;
    params->S = (ssize_t)fmaxf(roundf(calculated_step), 3.0f); // Ensure step size is at least 1

    // Bias parameters are ignored for now, can be set to 0 if needed by other parts of code.
    params->bias_x = 0;
    params->bias_y = 0;

    return params;
}

KernelParameters* kernel_parameters_biased(const int terrain_value, Point2D* biases) {
    KernelParameters* terrain_dependant = kernel_parameters_terrain(terrain_value);
    terrain_dependant->bias_x = biases->x;
    terrain_dependant->bias_y = biases->y;
    return terrain_dependant;
}


KernelParametersTerrain* get_kernels_terrain(TerrainMap* terrain) {
    size_t width = terrain->width;
    size_t height = terrain->height;
    KernelParametersTerrain* kernel_parameters = malloc(sizeof(KernelParametersTerrain));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    KernelParameters*** kernel_parameters_per_cell = malloc(sizeof(KernelParameters**) * height);
    for (size_t i = 0; i < height; i++) {
        kernel_parameters_per_cell[i] = (KernelParameters**)malloc(sizeof(KernelParameters*) * width);
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];
            KernelParameters* parameters = kernel_parameters_terrain(terrain_value);
            kernel_parameters_per_cell[y][x] = parameters;
        }
    }
    return kernel_parameters;
}

KernelParametersTerrainWeather* get_kernels_terrain_weather(const TerrainMap* terrain, const WeatherGrid* weather) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;
    const size_t times = weather->entries[0][0]->length;

    KernelParametersTerrainWeather* kernel_parameters = malloc(sizeof(KernelParametersTerrainWeather));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = times;
    KernelParameters**** kernel_parameters_per_cell = malloc(sizeof(KernelParameters***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = (KernelParameters***)malloc(sizeof(KernelParameters**) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters*) * times);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    size_t delta_x = width / weather->width;
    size_t delta_y = width / weather->height;

    assert(delta_x > 0);
    assert(delta_y > 0);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];
            // weather indices to terrain grid
            size_t weather_x = (x * weather->width) / terrain->width;
            size_t weather_y = (y * weather->height) / terrain->height;

            // paranoia-check (clamping)
            if (weather_x >= weather->width) weather_x = weather->width - 1;
            if (weather_y >= weather->height) weather_y = weather->height - 1;

            for (size_t t = 0; t < times; t++) {
                WeatherEntry* weather_entry = weather->entries[weather_y][weather_x]->data[t];
                KernelParameters* parameters = kernel_parameters_new(terrain_value, weather_entry);
                kernel_parameters_per_cell[y][x][t] = parameters;
            }
        }
    }
    return kernel_parameters;
}

KernelParametersTerrainWeather* get_kernels_terrain_biased(const TerrainMap* terrain, const Point2DArray* biases) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;
    const size_t times = biases->length;

    KernelParametersTerrainWeather* kernel_parameters = malloc(sizeof(KernelParametersTerrainWeather));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = times;

    KernelParameters**** kernel_parameters_per_cell = malloc(sizeof(KernelParameters***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = (KernelParameters***)malloc(sizeof(KernelParameters**) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters*) * times);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];

            for (size_t t = 0; t < times; t++) {
                Point2D* bias = &biases->points[t];
                KernelParameters* parameters = kernel_parameters_biased(terrain_value, bias);
                kernel_parameters_per_cell[y][x][t] = parameters;
            }
        }
    }
    return kernel_parameters;
}

KernelParametersTerrainWeather*
get_kernels_terrain_biased_grid(const TerrainMap* terrain, Point2DArrayGrid* biases) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;
    const size_t times = biases->data[0][0]->length;

    const size_t bias_grid_width = biases->width;
    const size_t bias_grid_height = biases->height;

    KernelParametersTerrainWeather* kernel_parameters = malloc(sizeof(KernelParametersTerrainWeather));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = times;

    KernelParameters**** kernel_parameters_per_cell = malloc(sizeof(KernelParameters***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = malloc(sizeof(KernelParameters**) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters*) * times);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Mapping terrain cell (x, y) to grid cell (gx, gy)
            size_t gx = x * bias_grid_width / width;
            size_t gy = y * bias_grid_height / height;

            // Clamp to ensure in bounds due to possible rounding
            if (gx >= bias_grid_width) gx = bias_grid_width - 1;
            if (gy >= bias_grid_height) gy = bias_grid_height - 1;

            const int terrain_value = terrain->data[y][x];

            for (size_t t = 0; t < times; t++) {
                Point2D* bias = &biases->data[gy][gx]->points[t];
                KernelParameters* parameters = kernel_parameters_biased(terrain_value, bias);
                kernel_parameters_per_cell[y][x][t] = parameters;
            }
        }
    }

    return kernel_parameters;
}


WeatherEntry* parse_csv(const char* csv_data, int* num_entries) {
    printf("start parsing\n");
    if (csv_data == NULL || num_entries == NULL) {
        *num_entries = 0;
        printf("file not found");
        return NULL;
    }

    char* data_copy = strdup(csv_data);
    if (data_copy == NULL) {
        *num_entries = 0;
        printf("strdup failed");
        return NULL;
    }

    int capacity = 10;
    int count = 0;
    WeatherEntry* entries = malloc(capacity * sizeof(WeatherEntry));
    if (entries == NULL) {
        free(data_copy);
        *num_entries = 0;
        printf("malloc failed");
        return NULL;
    }

    char* line = strtok(data_copy, "\n");
    if (line != NULL) {
        line = strtok(NULL, "\n");
    }
    else {
        printf("strtok failed");
    }

    while (line != NULL) {
        if (count >= capacity) {
            capacity *= 2;
            WeatherEntry* temp = realloc(entries, capacity * sizeof(WeatherEntry));
            if (temp == NULL) {
                break;
            }
            entries = temp;
        }

        WeatherEntry* entry = &entries[count];
        memset(entry, 0, sizeof(WeatherEntry));
        char* start = line;
        int col = 0;
        while (start && *start) {
            char* token = start;
            char* next_comma = strchr(start, ',');
            if (next_comma) {
                *next_comma = '\0';
                start = next_comma + 1;
            }
            else {
                start = NULL;
            }

            switch (col) {
            case 3:
                entry->temperature = atof(token);
                break;
            case 4:
                entry->humidity = atoi(token);
                break;
            case 5:
                entry->precipitation = atof(token);
                break;
            case 6:
                entry->wind_speed = atof(token);
                break;
            case 7:
                entry->wind_direction = atof(token);
                break;
            case 8:
                entry->snow_fall = atof(token);
                break;
            case 9:
                entry->weather_code = atoi(token);
                break;
            case 10:
                entry->cloud_cover = atoi(token);
                break;
            default:
                break;
            }

            col++;
            if (col > 10) {
                break;
            }
        }

        count++;
        line = strtok(NULL, "\n");
    }

    free(data_copy);
    *num_entries = count;
    return entries;
}


void kernel_parameters_terrain_free(KernelParametersTerrain* kernel_parameters_terrain) {
    size_t width = kernel_parameters_terrain->width;
    size_t height = kernel_parameters_terrain->height;
    KernelParameters*** kernel_parameters_per_cell = kernel_parameters_terrain->data;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            free(kernel_parameters_per_cell[y][x]);
        }
        free(kernel_parameters_per_cell[y]);
    }
    free(kernel_parameters_per_cell);
    free(kernel_parameters_terrain);
}

void kernel_parameters_mixed_free(KernelParametersTerrainWeather* kernel_parameters_terrain) {
    size_t width = kernel_parameters_terrain->width;
    size_t height = kernel_parameters_terrain->height;
    size_t times = kernel_parameters_terrain->time;
    KernelParameters**** kernel_parameters_per_cell = kernel_parameters_terrain->data;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t z = 0; z < times; z++) {
                free(kernel_parameters_per_cell[y][x][z]);
            }
            free(kernel_parameters_per_cell[y][x]);
        }
        free(kernel_parameters_per_cell[y]);
    }
    free(kernel_parameters_terrain);
}


Point2D* weather_entry_to_bias(WeatherEntry* entry, ssize_t max_bias) {
    if (entry == NULL) return NULL;
    // Adjust these parameters based on your data range
    const double MAX_WIND_SPEED = 20.0; // Reduced from 40 since your winds are weaker
    const double MIN_BIAS_THRESHOLD = 0.3; // Minimum bias magnitude to consider
    double wind_speed = entry->wind_speed;
    double wind_direction = entry->wind_direction;
    // Normalize wind speed to 0-5 range based on MAX_WIND_SPEED
    double normalized_magnitude = 2 * 4 * (wind_speed * (double)max_bias) / MAX_WIND_SPEED;
    // Apply threshold - ignore very small biases
    if (normalized_magnitude < MIN_BIAS_THRESHOLD) {
        return point_2d_new(0, 0);
    }
    // Cap at maximum bias
    if (normalized_magnitude > (double)max_bias) {
        normalized_magnitude = (double)max_bias;
    }
    // Convert direction to radians (meteorological convention)
    double radians = (270.0 - wind_direction) * M_PI / 180.0; // Convert to math convention
    // Calculate components
    double bias_x = normalized_magnitude * cos(radians);
    double bias_y = normalized_magnitude * sin(radians);

    // Round to nearest integers
    ssize_t x = (ssize_t)round(bias_x);
    ssize_t y = (ssize_t)round(bias_y);

    return point_2d_new(x, y);
}

void weather_entry_free(WeatherEntry* entry) {
    if (entry == NULL) return;
    free(entry);
}
