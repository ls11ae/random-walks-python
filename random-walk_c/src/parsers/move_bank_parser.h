#pragma once

#include "math/Point2D.h"
#include "parsers/terrain_parser.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif


KernelParametersTerrain* get_kernels_terrain(TerrainMap* terrain);

KernelParameters* kernel_parameters_biased(const int terrain_value, Point2D* biases);

KernelParametersTerrainWeather* get_kernels_terrain_weather(const TerrainMap* terrain, const WeatherGrid* weather);

KernelParametersTerrainWeather* get_kernels_terrain_biased(const TerrainMap* terrain, const Point2DArray* biases);

WeatherEntry* parse_csv(const char* csv_data, int* num_entries);

KernelParametersTerrainWeather*
get_kernels_terrain_biased_grid(const TerrainMap* terrain, Point2DArrayGrid* biases);

void kernel_parameters_terrain_free(KernelParametersTerrain* kernel_parameters_terrain);

void kernel_parameters_mixed_free(KernelParametersTerrainWeather* kernel_parameters_terrain);

KernelParameters* kernel_parameters_terrain(int terrain_value);

KernelParameters* kernel_parameters_new(int terrain_value, WeatherEntry* weather_entry);

Coordinate_array* extractLocationsFromCSV(const char* csv_file_path, const char* animal_id);

Coordinate_array* coordinate_array_new(Coordinate* coordinates, size_t length);

Point2DArray* getNormalizedLocations(const Coordinate_array* path, size_t W, size_t H);

Point2DArray* extractSteps(Point2DArray* path, size_t step_count);

void coordinate_array_free(Coordinate_array* coordinate_array);

Point2D* weather_entry_to_bias(WeatherEntry* entry, ssize_t max_bias);

void weather_entry_free(WeatherEntry* entry);

#ifdef __cplusplus
}
#endif
