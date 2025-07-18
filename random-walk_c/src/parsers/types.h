#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    double x; // longitude
    double y; // latitude
} Coordinate;

typedef struct {
    Coordinate* points;
    size_t length;
} Coordinate_array;


typedef struct {
    bool is_brownian;
    ssize_t S;
    ssize_t D;
    float diffusity;
    ssize_t bias_x;
    ssize_t bias_y;
} KernelParameters;

enum landmarkType {
    TREE_COVER = 10,
    SHRUBLAND = 20,
    GRASSLAND = 30,
    CROPLAND = 40,
    BUILT_UP = 50,
    SPARSE_VEGETATION = 60,
    SNOW_AND_ICE = 70,
    WATER = 80,
    HERBACEOUS_WETLAND = 90,
    MANGROVES = 95,
    MOSS_AND_LICHEN = 100
};


typedef struct {
    size_t width;
    size_t height;
    KernelParameters*** data;
} KernelParametersTerrain;

typedef struct {
    size_t width;
    size_t height;
    size_t time;
    KernelParameters**** data;
} KernelParametersTerrainWeather;

typedef struct {
    float temperature;
    int humidity;
    float precipitation;
    float wind_speed;
    float wind_direction;
    float snow_fall;
    int weather_code;
    int cloud_cover;
} WeatherEntry;

typedef struct {
    WeatherEntry** data;
    size_t length;
} WeatherTimeline;

typedef struct {
    size_t height;
    size_t width;
    WeatherTimeline*** entries; // Timeline at [y][x]
} WeatherGrid;

typedef struct {
    int** data;
    ssize_t width, height;
} TerrainMap;


#ifdef __cplusplus
}
#endif
