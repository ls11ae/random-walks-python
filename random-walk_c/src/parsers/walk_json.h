#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include "math/Point2D.h"
#include "terrain_parser.h"
#include "cJSON.h"

void save_walk_to_json(const Point2DArray *steps, const Point2DArray *walk, const TerrainMap *terrain,
                       const char *filename);

// Without steps
void save_walk_to_json_nosteps(const Point2DArray *walk, const TerrainMap *terrain,
                               const char *filename);

// Without terrain
void save_walk_to_json_noterrain(const Point2DArray *steps, const Point2DArray *walk, size_t W, size_t H,
                                 const char *filename);

// Without steps and terrain
void save_walk_to_json_onlywalk(const Point2DArray *walk, size_t W, size_t H, const char *filename);

void load_full_walk(const char *filename, Point2DArray *steps, Point2DArray *walk, TerrainMap *terrain);

void load_walk_with_terrain(const char *filename,
                            Point2DArray *walk,
                            TerrainMap *terrain);

void load_walk_with_steps(const char *filename,
                          Point2DArray *steps,
                          Point2DArray *walk);

void load_walk_only(const char *filename,
                    Point2DArray *walk);
#ifdef __cplusplus
    }
#endif
