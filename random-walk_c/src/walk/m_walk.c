#include "m_walk.h"

#include <assert.h>
#include <math.h>
#include <time.h>

#include "math/math_utils.h"
#include "math/path_finding.h"
#include "parsers/walk_json.h"
#include "parsers/weather_parser.h"


Point2DArray* mixed_walk(ssize_t W, ssize_t H, TerrainMap* spatial_map,
                         KernelsMap3D* tensor_map, Tensor* c_kernel, ssize_t T, const Point2DArray* steps) {
	return c_walk_backtrace_multiple(T, W, H, c_kernel, spatial_map, tensor_map, steps);
}

Tensor** m_walk(ssize_t W, ssize_t H, const TerrainMap* terrain_map,
                const KernelsMap3D* kernels_map, const ssize_t T, const ssize_t start_x,
                const ssize_t start_y) {
	Tensor* start_kernel = kernels_map->kernels[start_y][start_x];
	size_t max_D = kernels_map->max_D;
	Matrix* map = matrix_new(W, H);
	const double init_value = 1.0 / (double)start_kernel->len;
	matrix_set(map, start_x, start_y, init_value);
	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);
	Tensor** DP_mat = malloc(T * sizeof(Tensor*));
	for (int i = 0; i < T; i++) {
		Tensor* current = tensor_new(W, H, max_D);
		DP_mat[i] = current;
	}
	for (int d = 0; d < max_D; d++) {
		matrix_set(DP_mat[0]->data[d], start_x, start_y, init_value);
	}


	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				const size_t D = kernels_map->kernels[y][x]->len;
				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;
					for (int di = 0; di < D; di++) {
						const Matrix* current_kernel = kernels_map->kernels[y][x]->data[di];
						const ssize_t kernel_width = current_kernel->width;
						Vector2D* dir_cell_set = kernels_map->kernels[y][x]->dir_kernel;
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + kernel_width / 2;
							const ssize_t kernel_y = prev_kernel_y + kernel_width / 2;
							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];
							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
			}
		}
		printf("(%zd/%zd)\n", t, T);
	}
	//printf("DP calculation finished\n");
	return DP_mat;
}

Point2DArray* m_walk_backtrace(Tensor** DP_Matrix, const ssize_t T,
                               KernelsMap3D* tensor_map, TerrainMap* terrain, const ssize_t end_x, const ssize_t end_y,
                               const ssize_t dir) {
	//printf("backtrace\n");
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	fflush(stdout);
	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		const Tensor* current_tensor = tensor_map->kernels[y][x];
		const ssize_t D = (ssize_t)current_tensor->len;
		const ssize_t kernel_width = (ssize_t)current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t* movements_x = (ssize_t*)malloc(max_neighbors * sizeof(ssize_t));
		ssize_t* movements_y = (ssize_t*)malloc(max_neighbors * sizeof(ssize_t));
		double* prev_probs = (double*)malloc(max_neighbors * sizeof(double));
		int* directions = (int*)malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		size_t count = 0;
		Vector2D* dir_kernel = current_tensor->dir_kernel;
		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				// Neighbor indices
				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;


				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}
				if (terrain_at(prev_x, prev_y, terrain) == WATER || d >= tensor_map->kernels[prev_y][prev_x]->len)
					continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				// Kernel indices
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;


				const Matrix* current_kernel = tensor_map->kernels[prev_y][prev_x]->data[d];

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width ||
					kernel_y >= current_kernel->height) {
					continue;
				}
				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
				directions[count] = d;
				count++;
			}
		}


		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free(path->points);
			free(path);
			return NULL;
		}

		const ssize_t selected = weighted_random_index(prev_probs, count);
		ssize_t pre_x = movements_x[selected];
		ssize_t pre_y = movements_y[selected];

		direction = directions[selected];

		x -= pre_x;
		y -= pre_y;

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}


Tensor** mixed_walk_time(ssize_t W, ssize_t H,
                         TerrainMap* terrain_map,
                         KernelsMap4D* kernels_map, const ssize_t T,
                         const ssize_t start_x,
                         const ssize_t start_y) {
	// Initial assertions for input validation
	assert(W > 0 && H > 0 && "Invalid matrix dimensions");
	assert(terrain_map != NULL && "Terrain map is NULL");
	assert(kernels_map != NULL && "Kernels map is NULL");
	assert(T > 0 && "Invalid time steps");
	assert(start_x >= 0 && start_x < W && "Start x out of bounds");
	assert(start_y >= 0 && start_y < H && "Start y out of bounds");

	const Tensor* start_kernel = kernels_map->kernels[start_y][start_x][0];
	assert(start_kernel != NULL && "Start kernel is NULL");
	size_t max_D = kernels_map->max_D;
	printf("max d: %zu", max_D);

	Matrix* map = matrix_new(W, H);
	assert(map != NULL && "Failed to create matrix");
	printf("START VAL: %f", 1.0 / (double)start_kernel->len);
	assert(start_kernel->len > 0 && "Kernel length must be > 0");
	matrix_set(map, start_x, start_y, 1.0 / (double)start_kernel->len);

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

	Tensor** DP_mat = malloc(T * sizeof(Tensor*));
	assert(DP_mat != NULL && "Failed to allocate DP_mat");

	for (int i = 0; i < T; i++) {
		Tensor* current = tensor_new(W, H, max_D);
		assert(current != NULL && "Failed to create tensor");
		DP_mat[i] = current;
	}

	for (int d = 0; d < max_D; d++) {
		assert(DP_mat[0]->data[d] != NULL && "Matrix in tensor is NULL");
		matrix_copy_to(DP_mat[0]->data[d], map);
	}

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				const Tensor* tensor_at_t = kernels_map->kernels[y][x][t];
				Vector2D* dir_cell_set = tensor_at_t->dir_kernel;
				assert(tensor_at_t != NULL && "Tensor at time step is NULL");
				const size_t D = tensor_at_t->len;
				assert(D <= max_D && "Direction count exceeds max_D");

				for (ssize_t d = 0; d < D; ++d) {
					assert(d < DP_mat[t]->len && "Direction index out of bounds");
					assert(DP_mat[t]->data[d] != NULL && "Matrix in tensor is NULL");
					double sum = 0.0;

					for (int di = 0; di < D; di++) {
						assert(di < tensor_at_t->len && "Direction index out of bounds");
						const Matrix* current_kernel = tensor_at_t->data[di];
						assert(current_kernel != NULL && "Kernel matrix is NULL");
						const ssize_t kernel_width = current_kernel->width;
						assert(dir_cell_set != NULL && "Direction cell set is NULL");

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							assert(i < dir_cell_set->sizes[d] && "Direction cell index out of bounds");
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + kernel_width / 2;
							const ssize_t kernel_y = prev_kernel_y + kernel_width / 2;
							assert(
								kernel_x >= 0 && kernel_x < current_kernel->width &&
								"Kernel x out of bounds");
							assert(
								kernel_y >= 0 && kernel_y < current_kernel->height &&
								"Kernel y out of bounds");

							assert(di < DP_mat[t-1]->len && "Previous direction index out of bounds");
							assert(DP_mat[t-1]->data[di] != NULL && "Previous matrix in tensor is NULL");
							assert(
								yy * W + xx < DP_mat[t-1]->data[di]->len && "Matrix index out of bounds");
							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * current_kernel->width +
								kernel_x];
							// if (isnan(a)) {
							// 	matrix_print(DP_mat[t - 1]->data[di]);
							// 	printf("t=%zu, d=%i, y=%zu, x=%zu\n", t, di, y, x);
							// 	matrix_print(current_kernel);
							// 	printf("tensor size: %zd", tensor_at_t->data[di]->len);
							// 	exit(1);
							// }

							sum += a * b;
						}
					}
					// if (isnan(sum)) {
					// 	sum = 0.0;
					// }
					assert(y * W + x < DP_mat[t]->data[d]->len && "Matrix index out of bounds");
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
			}
		}
		printf("(%zd/%zd)\n", t, T);
	}

	return DP_mat;
}

Point2DArray* backtrace_time_walk(Tensor** DP_Matrix, const ssize_t T, const TerrainMap* terrain,
                                  const KernelsMap4D* kernels_map, const ssize_t end_x, const ssize_t end_y,
                                  const ssize_t dir) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		const Tensor* current_tensor = kernels_map->kernels[y][x][t];
		const size_t D = current_tensor->len;
		const ssize_t kernel_width = current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const size_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		ssize_t* movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t* movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double* prev_probs = malloc(max_neighbors * sizeof(double));
		int* directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		size_t count = 0;
		Vector2D* dir_kernel = current_tensor->dir_kernel;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				if (d >= kernels_map->kernels[prev_y][prev_x][t - 1]->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix* current_kernel = kernels_map->kernels[prev_y][prev_x][t - 1]->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >=
					current_kernel->
					height)
					continue;

				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
				assert(!isnan(p_b_a));

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_b_a;
				directions[count] = d;
				count++;
			}
		}

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			return NULL;
		}

		const ssize_t selected = weighted_random_index(prev_probs, count);
		x -= movements_x[selected];
		y -= movements_y[selected];
		direction = directions[selected];

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray* time_walk_geo(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                            int grid_x, int grid_y,
                            Point2D start, Point2D goal) {
	Point2DArrayGrid* grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");
	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');

	KernelsMap4D* kmap = tensor_map_terrain_biased_grid(&terrain, grid);

	Tensor** dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y);
	Point2DArray* walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0);


	Point2D* points = (Point2D*)(malloc(sizeof(Point2D) * 2));
	points[0] = start;
	points[1] = goal;

	Point2DArray* steps = point_2d_array_new(points, 2);
	save_walk_to_json(steps, walk, &terrain, walk_path);

	point2d_array_print(steps);
	tensor4D_free(dp, T);
	//kernels_map4d_free(kmap);
	point_2d_array_grid_free(grid);

	return walk;
}

Point2DArray* time_walk_geo_multi(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                                  int grid_x, int grid_y,
                                  Point2DArray* steps) {
	Point2DArray* result = malloc(sizeof(Point2DArray));
	result->points = malloc(sizeof(Point2D) * (steps->length - 1) * T);
	result->length = (steps->length - 1) * T;

	int index = 0;

	Point2DArrayGrid* grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");
	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');

	KernelsMap4D* kmap = tensor_map_terrain_biased_grid(&terrain, grid);

	for (int i = 0; i < steps->length - 1; ++i) {
		Point2D start = steps->points[i];
		Point2D goal = steps->points[i + 1];
		Tensor** dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y);
		Point2DArray* walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0);

		for (int s = 0; s < walk->length; ++s) {
			result->points[index++] = walk->points[s];
		}
		point2d_array_print(steps);
		tensor4D_free(dp, T);
		point2d_array_free(walk);
	}
	point_2d_array_grid_free(grid);
	save_walk_to_json(steps, result, &terrain, walk_path);
	return result;
}
