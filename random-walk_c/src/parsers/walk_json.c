#include "walk_json.h"

#include <assert.h>

static void save_walk_to_json_general(
    const Point2DArray* steps,
    const Point2DArray* walk,
    const TerrainMap* terrain, size_t W, size_t H,
    const char* filename) {
    if (!walk || !filename) {
        assert(walk);
        assert(filename);
        fprintf(stderr, "Error: NULL input parameter\n");
        return;
    }

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    if (walk->length == 0) {
        fprintf(stderr, "Error: Walk array is empty\n");
        fclose(fp);
        return;
    }

    if (steps && steps->length == 0) {
        fprintf(stderr, "Error: Steps array is empty\n");
        fclose(fp);
        return;
    }

    fprintf(fp, "{\n");

    if (terrain) {
        fprintf(fp, "  \"Height\": %zu,\n", terrain->height);
        fprintf(fp, "  \"Width\": %zu,\n", terrain->width);
    }
    else {
        fprintf(fp, "  \"Height\": %zu,\n", H);
        fprintf(fp, "  \"Width\": %zu,\n", W);
    }

    if (steps && steps->length > 0) {
        fprintf(fp, "  \"Steps\": [\n");
        for (size_t i = 0; i < steps->length; ++i) {
            if (terrain && (steps->points[i].x >= terrain->width ||
                steps->points[i].y >= terrain->height)) {
                fprintf(stderr, "Coordinate out of bounds in Steps\n: %zu, %zu", steps->points[i].x,
                        steps->points[i].y);
                fclose(fp);
                return;
            }
            fprintf(fp, "    {\"x\": %zu, \"y\": %zu}",
                    steps->points[i].x, steps->points[i].y);
            fprintf(fp, "%s\n", (i < steps->length - 1) ? "," : "");
        }
        fprintf(fp, "  ]");
    }

    const Point2D* start = &walk->points[0];
    const Point2D* end = &walk->points[walk->length - 1];

    if (steps && steps->length > 0)
        fprintf(fp, ",\n");
    fprintf(fp, "  \"Start Point\": {\"x\": %zu, \"y\": %zu},\n", start->x, start->y);
    fprintf(fp, "  \"End Point\": {\"x\": %zu, \"y\": %zu},\n", end->x, end->y);

    fprintf(fp, "  \"Walk\": [\n");
    for (size_t i = 0; i < walk->length; ++i) {
        fprintf(fp, "    {\"x\": %zu, \"y\": %zu}",
                walk->points[i].x, walk->points[i].y);
        fprintf(fp, "%s\n", (i < walk->length - 1) ? "," : "");
    }
    fprintf(fp, "  ]");

    if (terrain) {
        fprintf(fp, ",\n  \"Terrain\": [\n");
        for (size_t row = 0; row < terrain->height; ++row) {
            fprintf(fp, "    [");
            for (size_t col = 0; col < terrain->width; ++col) {
                fprintf(fp, "%d", terrain->data[row][col]);
                if (col < terrain->width - 1) fprintf(fp, ", ");
            }
            fprintf(fp, "]%s\n", (row < terrain->height - 1) ? "," : "");
        }
        fprintf(fp, "  ]\n");
    }
    else {
        fprintf(fp, "\n");
    }

    fprintf(fp, "}\n");
    fclose(fp);
}

void save_walk_to_json(const Point2DArray* steps,
                       const Point2DArray* walk,
                       const TerrainMap* terrain,
                       const char* filename) {
    save_walk_to_json_general(steps, walk, terrain, terrain->width, terrain->height, filename);
    printf("Walk saved to: %s\n", filename);
}

void save_walk_to_json_nosteps(const Point2DArray* walk,
                               const TerrainMap* terrain,
                               const char* filename) {
    save_walk_to_json_general(NULL, walk, terrain, 0, 0, filename);
}

void save_walk_to_json_noterrain(const Point2DArray* steps,
                                 const Point2DArray* walk, size_t W, size_t H,
                                 const char* filename) {
    save_walk_to_json_general(steps, walk, NULL, W, H, filename);
}

void save_walk_to_json_onlywalk(const Point2DArray* walk, size_t W, size_t H,
                                const char* filename) {
    save_walk_to_json_general(NULL, walk, NULL, W, H, filename);
}

static void load_walk_from_json_general(const char* filename,
                                        Point2DArray* steps,
                                        Point2DArray* walk,
                                        TerrainMap* terrain) {
    if (!filename || !walk) {
        fprintf(stderr, "Error: NULL input parameter\n");
        return;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buffer = malloc(file_size + 1);
    if (!buffer) {
        fclose(fp);
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    fread(buffer, 1, file_size, fp);
    buffer[file_size] = '\0';
    fclose(fp);

    cJSON* root = cJSON_Parse(buffer);
    if (!root) {
        const char* error_ptr = cJSON_GetErrorPtr();
        fprintf(stderr, "JSON Error: %s\n", error_ptr ? error_ptr : "Unknown error");
        free(buffer);
        return;
    }

    // State tracking for cleanup
    int steps_allocated = 0;
    int walk_allocated = 0;
    int terrain_allocated = 0;

    // Load terrain dimensions if requested
    if (terrain) {
        cJSON* height = cJSON_GetObjectItemCaseSensitive(root, "Height");
        cJSON* width = cJSON_GetObjectItemCaseSensitive(root, "Width");
        if (!cJSON_IsNumber(height) || !cJSON_IsNumber(width)) {
            fprintf(stderr, "Invalid Height/Width format\n");
            goto cleanup;
        }
        terrain->height = height->valueint;
        terrain->width = width->valueint;
    }

    // Load steps if requested
    if (steps) {
        cJSON* steps_json = cJSON_GetObjectItemCaseSensitive(root, "Steps");
        if (!cJSON_IsArray(steps_json)) {
            fprintf(stderr, "Steps is not an array\n");
            goto cleanup;
        }

        steps->length = cJSON_GetArraySize(steps_json);
        steps->points = malloc(steps->length * sizeof(Point2D));
        if (!steps->points) {
            fprintf(stderr, "Steps memory allocation failed\n");
            goto cleanup;
        }
        steps_allocated = 1;

        size_t i = 0;
        cJSON* item;
        cJSON_ArrayForEach(item, steps_json) {
            cJSON* x = cJSON_GetObjectItem(item, "x");
            cJSON* y = cJSON_GetObjectItem(item, "y");
            if (!cJSON_IsNumber(x) || !cJSON_IsNumber(y)) {
                fprintf(stderr, "Invalid Step coordinates\n");
                goto cleanup;
            }
            steps->points[i++] = (Point2D){x->valueint, y->valueint};
        }
    }

    // Always load walk points
    cJSON* walk_json = cJSON_GetObjectItemCaseSensitive(root, "Walk");
    if (!cJSON_IsArray(walk_json)) {
        fprintf(stderr, "Walk is not an array\n");
        goto cleanup;
    }

    walk->length = cJSON_GetArraySize(walk_json);
    walk->points = malloc(walk->length * sizeof(Point2D));
    if (!walk->points) {
        fprintf(stderr, "Walk memory allocation failed\n");
        goto cleanup;
    }
    walk_allocated = 1;

    size_t i = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, walk_json) {
        cJSON* x = cJSON_GetObjectItem(item, "x");
        cJSON* y = cJSON_GetObjectItem(item, "y");
        if (!cJSON_IsNumber(x) || !cJSON_IsNumber(y)) {
            fprintf(stderr, "Invalid Walk coordinates\n");
            goto cleanup;
        }
        walk->points[i++] = (Point2D){x->valueint, y->valueint};
    }

    // Load terrain data if requested
    if (terrain) {
        cJSON* terrain_json = cJSON_GetObjectItemCaseSensitive(root, "Terrain");
        if (!cJSON_IsArray(terrain_json) ||
            cJSON_GetArraySize(terrain_json) != terrain->height) {
            fprintf(stderr, "Invalid Terrain format\n");
            goto cleanup;
        }

        terrain->data = malloc(terrain->height * sizeof(int*));
        if (!terrain->data) {
            fprintf(stderr, "Terrain memory allocation failed\n");
            goto cleanup;
        }
        terrain_allocated = 1;

        // Initialize pointers to NULL for safe cleanup
        for (size_t r = 0; r < terrain->height; r++) {
            terrain->data[r] = NULL;
        }

        for (size_t row = 0; row < terrain->height; row++) {
            cJSON* row_json = cJSON_GetArrayItem(terrain_json, row);
            if (!cJSON_IsArray(row_json) ||
                cJSON_GetArraySize(row_json) != terrain->width) {
                fprintf(stderr, "Invalid Terrain row\n");
                goto cleanup;
            }

            terrain->data[row] = malloc(terrain->width * sizeof(int));
            if (!terrain->data[row]) {
                fprintf(stderr, "Terrain row allocation failed\n");
                goto cleanup;
            }

            for (size_t col = 0; col < terrain->width; col++) {
                cJSON* val = cJSON_GetArrayItem(row_json, col);
                if (!cJSON_IsNumber(val)) {
                    fprintf(stderr, "Invalid Terrain value\n");
                    goto cleanup;
                }
                terrain->data[row][col] = val->valueint;
            }
        }
    }

    // Success path
    cJSON_Delete(root);
    free(buffer);
    return;

cleanup:
    // Cleanup allocated resources
    if (terrain_allocated) {
        for (size_t r = 0; r < terrain->height; r++) {
            free(terrain->data[r]);
        }
        free(terrain->data);
    }
    if (walk_allocated) {
        free(walk->points);
        walk->length = 0;
    }
    if (steps_allocated) {
        free(steps->points);
        steps->length = 0;
    }
    cJSON_Delete(root);
    free(buffer);
}

// Wrapper functions
void load_full_walk(const char* filename,
                    Point2DArray* steps,
                    Point2DArray* walk,
                    TerrainMap* terrain) {
    load_walk_from_json_general(filename, steps, walk, terrain);
}

void load_walk_with_terrain(const char* filename,
                            Point2DArray* walk,
                            TerrainMap* terrain) {
    load_walk_from_json_general(filename, NULL, walk, terrain);
}

void load_walk_with_steps(const char* filename,
                          Point2DArray* steps,
                          Point2DArray* walk) {
    load_walk_from_json_general(filename, steps, walk, NULL);
}

void load_walk_only(const char* filename,
                    Point2DArray* walk) {
    load_walk_from_json_general(filename, NULL, walk, NULL);
}
