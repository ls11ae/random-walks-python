#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <bits/atomic_base.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>

#include "walk/b_walk.h"
#include "walk/c_walk.h"
#include "walk/m_walk.h"
#include "matrix/matrix.h"
#include "parsers/terrain_parser.h"
#include "parsers/move_bank_parser.h"
#include "parsers/walk_json.h"
#include <sys/time.h>
#include "math/path_finding.h"
#include "math/kernel_slicing.h"
#include "parsers/weather_parser.h"
#include "parsers/serialization.h"

#include "config.h"
#include "memory_utils.h"

double chi_square_pdf(const double x, const int k) {
    return pow(x, (k / 2.0) - 1) * exp(-x / 2.0) / (pow(2, k / 2.0) * tgamma(k / 2.0));
}

double test_corr(ssize_t D) {
    double ram = get_mem_available_mib();
    ssize_t M = 21, W = 401, H = 401, T = 200;
    Tensor* c_ke_tensor;
    if (D == 1) {
        Matrix* b_kernel = matrix_generator_gaussian_pdf(M, M, 3.0, 5.5, 0, 0);
        // matrix_print(b_kernel);
        c_ke_tensor = tensor_new(M, M, 1);
        c_ke_tensor->data[0] = b_kernel;
    }
    if (D > 16) M = 21;
    c_ke_tensor = generate_kernels(D, M);
    TerrainMap* terrain = get_terrain_map("../../resources/landcover_142_brownian.txt", ' ');

    auto start = std::chrono::high_resolution_clock::now();
    auto t_map = tensor_map_new(terrain, c_ke_tensor);
    std::cout << "start_m\n";

    auto** DP = c_walk_init_terrain(W, H, c_ke_tensor, terrain, t_map, T, 200, 200);
    auto walk = backtrace(DP, T, c_ke_tensor, terrain, t_map, 380, 380, 0, D);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "dp_calculation took " << duration.count() << " seconds\n";
    tensor4D_free(DP, T);
    point2d_array_free(walk);
    terrain_map_free(terrain);
    return duration.count();
}

double test_brownian() {
    ssize_t M = 21, W = 401, H = 401, T = 200;
    Matrix* kernel = matrix_generator_gaussian_pdf(M, M, 3.0, 5.5, 0, 0);
    Matrix* start_m = matrix_new(W, H);
    matrix_set(start_m, 200, 200, 1.0);
    TerrainMap* terrain = get_terrain_map("../../resources/landcover_142.txt", ' ');

    auto start = std::chrono::high_resolution_clock::now();
    auto* kernel_map = kernels_map_new(terrain, kernel);
    auto* dp = b_walk_init_terrain(start_m, kernel, terrain, kernel_map, T);

    auto walk = b_walk_backtrace(dp, kernel, kernel_map, 380, 380);
    auto end = std::chrono::high_resolution_clock::now();

    tensor_free(dp);
    point2d_array_free(walk);
    matrix_free(start_m);
    terrain_map_free(terrain);
    // kernels_map_free(kernel_map);

    std::chrono::duration<double> duration = end - start;
    std::cout << "dp_calculation took " << duration.count() << " seconds\n";
    return duration.count();
}

Point2DArray* create_bias_array(const int T, const ssize_t bias_x, const ssize_t bias_y) {
    Point2D* bias_points = (Point2D*)malloc(sizeof(Point2D) * T);
    for (int t = 0; t < T; t++) {
        ssize_t current_bias_x;
        ssize_t current_bias_y;
        if (t < T / 3) {
            current_bias_x = bias_x;
            current_bias_y = bias_y;
        }
        else if (t < 2 * T / 3) {
            current_bias_x = 0; //ost
            current_bias_y = 0; //ost
        }
        else {
            current_bias_x = -bias_x;
            current_bias_y = -bias_y;
        }
        bias_points[t] = (Point2D){current_bias_x, current_bias_y};
    }
    Point2DArray* biases = point_2d_array_new(bias_points, T);
    return biases;
}

int test_biased_walk(Point2DArray* biases, const char* filename) {
    TerrainMap terrain;
    parse_terrain_map(filename, &terrain, ' ');
    std::cout << terrain.width << " H: " << terrain.height << "\n";

    const int T = 100;
    // Kernel Map generieren mit terrain und bias timeline
    KernelsMap4D* kmap = tensor_map_terrain_biased(&terrain, biases);

    Point2D start = {360, 227};
    Point2D goal = {290, 132};

    auto dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y);
    auto walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0);

    std::cout << matrix_get(dp[T - 1]->data[0], goal.x, goal.y) << "\n";


    save_walk_to_json(biases, walk, &terrain, "timewalk3.json");
    point2d_array_print(walk);

    tensor4D_free(dp, T);
    point2d_array_free(biases);
    point2d_array_free(walk);
    return 0;
}

int test_biased_walk_grid(Point2DArrayGrid* grid, const char* filename, ssize_t W, ssize_t H, size_t T, Point2D start,
                          Point2D goal) {
    TerrainMap terrain;
    parse_terrain_map(filename, &terrain, ' ');
    terrain.width = W;
    terrain.height = H;

    // Kernel Map generieren mit terrain und bias timeline
    KernelsMap4D* kmap = tensor_map_terrain_biased_grid(&terrain, grid);

    Tensor** dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y);
    std::cout << matrix_get(dp[T - 1]->data[0], goal.x, goal.y) << "\n";
    Point2DArray* walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0);


    Point2D* points = static_cast<Point2D*>(malloc(sizeof(Point2D) * 2));
    points[0] = start;
    points[1] = goal;

    Point2DArray* steps = point_2d_array_new(points, 2);
    save_walk_to_json(steps, walk, &terrain, "timewalk3.json");
    point2d_array_print(walk);

    return 0;
}


int test_serialization_terrain() {
    TerrainMap terrain;
    parse_terrain_map("../../resources/terrain_movebank.txt", &terrain, ' ');
    KernelsMap3D* tensor_map = tensor_map_terrain(&terrain);
    serialize_kernels_map_3d(tensor_map, "terrain_baboons");
    auto* loaded_map = deserialize_kernels_map_3d("terrain_baboons");
    std::cout << loaded_map->height << loaded_map->width << std::endl;
    return 0;
}

int test_geo_multi() {
    // Read file content
    Point2D start = {100, 120};
    Point2D st = {60, 20};
    Point2D goal = {150, 20};


    const char* csv = "../../resources/my_gridded_weather_grid_csvs";
    const char* terrain_path = "../../resources/terrain_movebank.txt";
    const char* walk_path = "../../resources/twalk3.json";
    ssize_t T = 100, W = 200, H = 200;

    Point2D* steps = static_cast<Point2D*>(malloc(sizeof(Point2D) * 3));
    steps[0] = start;
    steps[1] = st;
    steps[2] = goal;

    Point2DArray* steos = point_2d_array_new(steps, 3);

    auto walk = time_walk_geo_multi(T, csv, terrain_path, walk_path, 5, 5, steos);
    point2d_array_print(walk);
    point2d_array_free(walk);
    point2d_array_free(steos);
    return 0;
}

void test_brownian_terrain() {
    auto terrain = create_terrain_map("../../resources/landcover_142.txt", ' ');
    Matrix* kernel = matrix_generator_gaussian_pdf(15, 15, 1.0, 1.0, 0, 0);

    auto kernels_map = kernels_map_new(terrain, kernel);
    auto dp = brownian_walk_terrain_init(250, terrain->width, terrain->height, 200, 200, kernel, terrain, kernels_map);
    auto walk = b_walk_backtrace(dp, kernel, kernels_map, 200, 370);
    point2d_array_print(walk);
}

int main() {
    auto kernels = generate_kernels(8, 15);
    auto terrain = create_terrain_map("../../resources/landcover_142.txt", ' ');

    std::cout << terrain->data[0][0] ;

    auto tmap = tensor_map_new(terrain, kernels);
    Tensor** dp = c_walk_init_terrain(terrain->width, terrain->height, kernels, terrain, tmap, 80, 200, 200);
    auto walk = backtrace(dp, 80, kernels, terrain, tmap, 350, 390, 2, 8);
    point2d_array_print(walk);
    tensor4D_free(dp, 80);
}
