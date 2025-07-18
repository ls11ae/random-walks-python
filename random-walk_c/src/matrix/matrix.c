#include <stdlib.h>  // Für malloc, free, NULL
#include <string.h>  // Für memset
#include <stdio.h>   // Für fprintf, fwrite
#include <stddef.h>  // Für size_t
#include <math.h>
#include <assert.h>

#include "matrix.h"
#include <float.h>

#include "math/math_utils.h"

Matrix* matrix_new(const ssize_t width, const ssize_t height) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL; // Fehlerbehandlung für Matrix Allokierung

    m->width = width;
    m->height = height;
    m->len = width * height;

    // Speicher für die Matrixdaten allokieren
    m->data = (double*)calloc(m->len, sizeof(double));
    if (!m->data) {
        free(m); // Speicher für Matrix freigeben
        return NULL; // Fehlerbehandlung
    }

    return m;
}

void matrix_free(Matrix* matrix) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    free(matrix->data);
    free(matrix);
}

void matrix_convolution(Matrix* input, Matrix* kernel, Matrix* output) {
    for (size_t i = 0; i < input->len; i++) {
        output->data[i] = input->data[i] * kernel->data[i];
    }
}

bool matrix_equals(const Matrix* matrix1, const Matrix* matrix2) {
    assert(matrix1 != NULL);
    assert(matrix2 != NULL);
    if (matrix1->len != matrix2->len) return false;
    for (size_t i = 0; i < matrix1->len; i++) {
        if (fabs(matrix1->data[i] - matrix2->data[i]) < 0.1) return false;
    }
    return true;
}

void matrix_pooling_avg(Matrix* dst, const Matrix* src) {
    if (!src || !dst || !src->data || !dst->data) {
        return; // Ungültige Eingabe
    }

    size_t pool_width = src->width / dst->width;
    size_t pool_height = src->height / dst->height;


    size_t dst_index = 0;
    for (size_t dst_y = 0; dst_y < dst->height; dst_y++) {
        for (size_t dst_x = 0; dst_x < dst->width; dst_x++) {
            double sum = 0.0;
            size_t count = 0;

            // Durchlaufe das Pooling-Fenster
            for (size_t src_y = 0; src_y < pool_height; src_y++) {
                for (size_t src_x = 0; src_x < pool_width; src_x++) {
                    size_t x = dst_x * pool_width + src_x;
                    size_t y = dst_y * pool_height + src_y;
                    sum += matrix_get(src, x, y);
                    count++;
                }
            }

            dst->data[dst_index++] = sum / count;
        }
    }
}

Matrix* matrix_copy(const Matrix* matrix) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist

    Matrix* copy = matrix_new(matrix->width, matrix->height);
    if (copy == NULL) {
        return NULL; // Fehler, wenn die Kopie nicht erfolgreich erstellt werden konnte
    }

    memcpy(copy->data, matrix->data, sizeof(double) * matrix->len);
    return copy;
}

void matrix_copy_to(Matrix* dest, const Matrix* src) {
    assert(dest->width == src->width && dest->height == src->height);
    memcpy(dest->data, src->data, dest->width * dest->height * sizeof(double));
}

int matrix_in_bounds(const Matrix* matrix, size_t x, size_t y) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    return x < matrix->width && y < matrix->height;
}

double matrix_get(const Matrix* matrix, const size_t x, const size_t y) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(x >= 0 && y >= 0 && x < matrix->width && y < matrix->height);
    return matrix->data[y * matrix->width + x];
}

void matrix_set(const Matrix* matrix, const size_t x, const size_t y, double val) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(x >= 0 && y >= 0 && x < matrix->width && y < matrix->height);
    matrix->data[y * matrix->width + x] = val;
}


void matrix_fill(Matrix* matrix, const double value) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    if (value == 0.0) {
        memset(matrix->data, 0, matrix->len * sizeof(double));
        return;
    }
    // Direktes Setzen von Werten mit einer optimierten Schleife
    double* data_index = matrix->data;
    const double* data_end = matrix->data + matrix->len;
    while (data_index < data_end) {
        *(data_index++) = value;
    }
}

Matrix* matrix_add(const Matrix* a, const Matrix* b) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(b != NULL); // Überprüft, ob matrix nicht NULL ist
    if (a->len != b->len) return NULL; // Grundprüfung

    Matrix* result = matrix_new(a->width, a->height);
    if (result == NULL) return NULL;

    const size_t len = a->len;
    const double* data_a = a->data;
    const double* data_b = b->data;
    double* data_result = result->data;
    for (size_t i = 0; i < len; i++) {
        data_result[i] = data_a[i] + data_b[i];
    }
    return result;
}

Matrix* matrix_sub(const Matrix* a, const Matrix* b) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(b != NULL); // Überprüft, ob matrix nicht NULL ist
    if (a->len != b->len) return NULL; // Grundprüfung

    Matrix* result = matrix_new(a->width, a->height);
    if (result == NULL) return NULL;

    const size_t len = a->len;
    const double* data_a = a->data;
    const double* data_b = b->data;
    double* data_result = result->data;
    for (size_t i = 0; i < len; ++i) {
        data_result[i] = data_a[i] - data_b[i];
    }
    return result;
}

Matrix* matrix_mul(const Matrix* a, const Matrix* b) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(b != NULL); // Überprüft, ob matrix nicht NULL ist
    if (a->len != b->len) return NULL; // Grundprüfung

    Matrix* result = matrix_new(b->width, a->height);
    if (result == NULL) return NULL;

    const double* data_a = a->data;
    const double* data_b = b->data;
    double* data_result = result->data;
    for (size_t i = 0; i < a->height; ++i) {
        for (size_t j = 0; j < b->width; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a->width; ++k) {
                sum += data_a[i * a->width + k] * data_b[k * b->width + j];
            }
            data_result[i * b->width + j] = sum;
        }
    }

    return result;
}

Matrix* matrix_elementwise_mul(const Matrix* a, const Matrix* b) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(b != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(a->len == b->len);
    if (a->len != b->len) return NULL; // Grundprüfung

    Matrix* result = matrix_new(a->width, a->height);
    if (result == NULL) return NULL;

    const size_t len = a->len;
    const double* data_a = a->data;
    const double* data_b = b->data;
    double* data_result = result->data;
    for (size_t i = 0; i < len; ++i) {
        data_result[i] = data_a[i] * data_b[i];
    }
    return result;
}


double matrix_sum(const Matrix* matrix) {
    if (matrix == NULL) return 0.0;
    double sum = 0.0;
    for (size_t index = 0; index < matrix->len; index++) {
        sum += matrix->data[index];
    }
    return sum;
}

void matrix_transpose(Matrix* m) {
    assert(m != NULL);
    Matrix* temp = matrix_copy(m);
    for (size_t y = 0; y < m->height; y++) {
        for (size_t x = 0; x < m->width; x++) {
            matrix_set(m, x, y, matrix_get(temp, y, x));
        }
    }
    matrix_free(temp);
}

Matrix* matrix_invert(const Matrix* input) {
    if (input->width != input->height) {
        fprintf(stderr, "Fehler: Nur quadratische Matrizen können invertiert werden.\n");
        exit(EXIT_FAILURE);
    }

    if (input->width == 2) {
        double det = matrix_determinant(input);
        if (det == 0) {
            fprintf(stderr, "Fehler: Matrix ist singulär und kann nicht invertiert werden.\n");
            exit(EXIT_FAILURE);
        }

        // Inverse berechnen für 2x2 Matrix
        Matrix* inv = matrix_new(input->width, input->height);
        if (inv->data == NULL) {
            fprintf(stderr, "Fehler bei der Speicherzuweisung für die Inverse Matrix!\n");
            exit(EXIT_FAILURE);
        }

        // Berechnung der Inversen einer 2x2 Matrix
        inv->data[0] = input->data[3] / det;
        inv->data[1] = -input->data[1] / det;
        inv->data[2] = -input->data[2] / det;
        inv->data[3] = input->data[0] / det;

        return inv;
    }

    // TODO 2x2 < NxN
    // Für größere Matrizen müsste eine andere Methode verwendet werden (z.B. Gaussian Elimination)
    fprintf(stderr, "Fehler: Diese Funktion unterstützt nur 2x2 Matrizen.\n");
    exit(EXIT_FAILURE);
}

double matrix_determinant(const Matrix* mat) {
    if (mat->width != mat->height) {
        fprintf(stderr, "Fehler: Nur quadratische Matrizen haben eine Determinante.\n");
        exit(EXIT_FAILURE);
    }

    // Basisfall: 2x2-Matrix
    if (mat->width == 2) {
        return mat->data[0] * mat->data[3] - mat->data[1] * mat->data[2];
    }

    // TODO 2x2 < NxN
    fprintf(stderr, "Fehler: Diese Funktion unterstützt nur 2x2 Matrizen.\n");
    exit(EXIT_FAILURE);
    return 0;
}

void matrix_normalize(const Matrix* mat, double sum) {
    for (int i = 0; i < mat->len; ++i) {
        if (mat->data[i] == 0.0)
            mat->data[i] /= sum;
    }
}

void matrix_normalize_L1(Matrix* m) {
    if (!m || !m->data || m->len == 0) return;

    double sum = 0.0;

    // Gesamtsumme berechnen
    for (size_t i = 0; i < m->len; i++) {
        sum += m->data[i];
    }

    if (sum == 0.0) return; // Verhindert Division durch 0

    // Werte normalisieren
    for (size_t i = 0; i < m->len; i++) {
        m->data[i] /= sum;
    }
}

void matrix_normalize_01(Matrix* m) {
    if (!m || !m->data || m->len == 0) return;

    double sum = 0;

    // Minimum und Maximum finden
    for (size_t i = 0; i < m->len; i++) {
        sum += m->data[i];
    }
    // Werte normalisieren
    for (size_t i = 0; i < m->len; i++) {
        m->data[i] /= sum;
    }
}

char* matrix_to_string(const Matrix* mat) {
    const char presition = 4;
    // Berechnen der benötigten Größe für den String
    size_t buffer_size = (mat->len << 1) * presition; // Platz für Zeilenumbrüche und Nullterminator
    char* result = (char*)malloc(buffer_size);
    if (!result) {
        fprintf(stderr, "Fehler: Speicher konnte nicht zugewiesen werden.\n");
        exit(EXIT_FAILURE);
    }

    size_t str_index = 0; // Aktuelle Position im String
    size_t w_index = 0;
    for (size_t index = 0; index < mat->len; ++index) {
        str_index += sprintf(&result[str_index], "%0.2f", mat->data[index]); // Format: %0.2f für 2 Dezimalstellen
        char c = ' ';
        w_index++;
        if (w_index == mat->width) {
            c = '\n';
            w_index = 0;
        }
        result[str_index++] = c;
    }
    result[str_index] = '\0'; // Nullterminator für den String

    return result;
}

size_t matrix_save(const Matrix* mat, const char* filename) {
    if (mat == NULL) return 0;

    FILE* file = fopen(filename, "wb"); // Open the file in binary write mode
    if (file == NULL) {
        perror("Error opening file");
        return 0;
    }

    size_t len = 0;
    len += fwrite(&mat->width, sizeof(size_t), 1, file);
    len += fwrite(&mat->height, sizeof(size_t), 1, file);
    len += fwrite(mat->data, sizeof(double), mat->len, file);
    if (len != mat->len + 2) {
        perror("Error writing data to file");
    }

    fclose(file);
    return len * sizeof(double);
}

Matrix* matrix_load(const char* filename) {
    FILE* file = fopen(filename, "rb"); // Open the file in binary read mode
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    size_t width, height;
    fread(&width, sizeof(size_t), 1, file);
    fread(&height, sizeof(size_t), 1, file);
    Matrix* mat = matrix_new(width, height);
    if (mat == NULL) {
        perror("Error allocating memory for matrix");
        fclose(file);
        return NULL;
    }

    size_t len = fread(mat->data, sizeof(double), mat->len, file);
    if (len != mat->len) {
        perror("Error reading data from file");
    }
    fclose(file);
    return mat;
}

void matrix_print(const Matrix* m) {
    for (size_t i = 0; i < m->height; i++) {
        for (size_t j = 0; j < m->width; j++) {
            printf("%0.5f ", matrix_get(m, j, i)); // Werte auf 3 Dezimalstellen
        }
        printf("\n");
    }
    printf("\n");
}

Matrix* matrix_combind(const Matrix* matrix1, const Matrix* matrix2) {
    if (matrix1 == NULL || matrix2 == NULL) {
        printf("matrix_combind: error dst == NULL || src == NULL");
        return NULL; // Fehlerbehandlung
    }

    if (matrix1->len != matrix2->len) {
        printf("matrix_combind: error matrix1->len != matrix2->len");
        return NULL; // Fehlerbehandlung
    }

    Matrix* result = matrix_new(matrix1->width, matrix1->height);
    if (result == NULL) return NULL;

    for (int i = 0; i < matrix1->len; i++) {
        result->data[i] = matrix1->data[i] * matrix2->data[i];
    }
    return result;
}

int matrix_combind_inplace(Matrix* dst, const Matrix* src) {
    if (dst == NULL || src == NULL) {
        printf("matrix_combind: error dst == NULL || src == NULL");
        return 0; // Fehlerbehandlung
    }

    if (dst->len != src->len) {
        printf("matrix_combind: error dst->len != src->len");
        return 0; // Fehlerbehandlung
    }

    for (int i = 0; i < src->len; i++) {
        dst->data[i] *= src->data[i];
    }

    return 1;
}

int matrix_add_inplace(Matrix* dst, const Matrix* src) {
    if (dst == NULL || src == NULL) {
        printf("matrix_add_inplace: error dst == NULL || src == NULL");
        return 0; // Fehlerbehandlung
    }

    if (dst->len != src->len) {
        printf("matrix_add_inplace: error dst->len != src->len");
        return 0; // Fehlerbehandlung
    }

    for (int i = 0; i < src->len; i++) {
        dst->data[i] += src->data[i];
    }

    return 1;
}


Matrix* matrix_upsample_bilinear(const Matrix* input, size_t new_w, size_t new_h) {
    Matrix* output = matrix_new(new_w, new_h);
    if (!output) return NULL;

    double x_ratio = (double)(input->width - 1) / (new_w - 1);
    double y_ratio = (double)(input->height - 1) / (new_h - 1);

    for (size_t ny = 0; ny < new_h; ny++) {
        for (size_t nx = 0; nx < new_w; nx++) {
            double gx = nx * x_ratio;
            double gy = ny * y_ratio;

            int x = (int)gx;
            int y = (int)gy;
            double x_diff = gx - x;
            double y_diff = gy - y;

            // Randbehandlung
            int x1 = (x + 1 < (int)input->width) ? x + 1 : x;
            int y1 = (y + 1 < (int)input->height) ? y + 1 : y;

            double A = input->data[y * input->width + x];
            double B = input->data[y * input->width + x1];
            double C = input->data[y1 * input->width + x];
            double D = input->data[y1 * input->width + x1];

            output->data[ny * new_w + nx] =
                A * (1 - x_diff) * (1 - y_diff) +
                B * x_diff * (1 - y_diff) +
                C * (1 - x_diff) * y_diff +
                D * x_diff * y_diff;
        }
    }

    return output;
}

Matrix* matrix_rotate(Matrix* original, double angle) {
    // Berechne den Rotationswinkel in Bogenmaß
    double radians = angle * M_PI / 180.0;

    // Neue Dimensionen der rotierten Matrix
    size_t new_width = original->height;
    size_t new_height = original->width;

    // Erstelle eine neue Matrix für das Ergebnis
    Matrix* rotated = matrix_new(new_width, new_height);
    rotated->len = new_width * new_height;
    rotated->data = (double*)malloc(rotated->len * sizeof(double));

    // Rotationsmatrix anwenden
    for (size_t i = 0; i < original->height; i++) {
        for (size_t j = 0; j < original->width; j++) {
            // Berechne die neuen Positionen
            size_t new_i = (size_t)(round(i * cos(radians) + j * sin(radians)));
            size_t new_j = (size_t)(round(-i * sin(radians) + j * cos(radians)));

            // Falls die Position innerhalb der Grenzen liegt
            if (new_i < new_height && new_j < new_width) {
                rotated->data[new_i * new_width + new_j] = original->data[i * original->width + j];
            }
        }
    }

    return rotated;
}

Matrix* matrix_rotate_center(Matrix* original, double angle) {
    // Berechne den Rotationswinkel in Bogenmaß
    double radians = angle * M_PI / 180.0;

    // Neue Dimensionen der rotierten Matrix
    size_t new_width = original->width;
    size_t new_height = original->height;

    // Erstelle eine neue Matrix für das Ergebnis
    Matrix* rotated = matrix_new(new_width, new_height);
    rotated->len = new_width * new_height;
    rotated->data = (double*)malloc(rotated->len * sizeof(double));

    // Berechne den Mittelpunkt der Matrix
    double center_x = (new_width - 1) / 2.0;
    double center_y = (new_height - 1) / 2.0;

    // Rotationslogik für jeden Punkt der Matrix
    for (size_t y = 0; y < new_height; y++) {
        for (size_t x = 0; x < new_width; x++) {
            // Berechne die Koordinaten relativ zum Mittelpunkt
            double rel_x = x - center_x;
            double rel_y = y - center_y;

            // Rotiere die Koordinaten
            double rotated_x = rel_x * cos(radians) - rel_y * sin(radians);
            double rotated_y = rel_x * sin(radians) + rel_y * cos(radians);

            // Setze die rotierten Koordinaten in die Zielmatrix
            size_t new_x = (size_t)(rotated_x + center_x);
            size_t new_y = (size_t)(rotated_y + center_y);

            // Falls die Position innerhalb der Matrix liegt, setze den Wert
            if (new_x < original->width && new_y < original->height) {
                rotated->data[y * new_width + x] = original->data[new_y * original->width + new_x];
            }
        }
    }

    return rotated;
}

// Convert diffusity to Gaussian parameters with terrain modulation
void get_gaussian_parameters(float diffusity, int terrain_value, float* out_sigma, float* out_scale) {
    // Base sigma-scaling factors per terrain type
    const float terrain_modifiers[] = {
        0.8f, // 10: Tree cover (reduced spread)
        1.1f, // 20: Shrubland
        1.3f, // 30: Grassland
        1.0f, // 40: Cropland
        0.6f, // 50: Built-up (constrained)
        1.5f, // 60: Desert (wide spread)
        0.5f, // 70: Snow/ice (concentrated)
        0.4f, // 80: Water
        0.9f, // 90: Wetland
        0.7f, // 95: Mangroves
        1.2f // 100: Moss/lichens
    };
    // Normalize terrain value to array index (assuming class values 10,20,...100)
    int terrain_index = (terrain_value / 10) - 1;
    terrain_index = fmax(0, fmin(terrain_index, 10));

    float effective_diffusity = diffusity * terrain_modifiers[terrain_index];

    // Sigma-Mindestwert einführen (z.B. 1.5)
    *out_sigma = fmax(1.5f, 0.5f + effective_diffusity * 1.5f); // Min. 1.5 statt 0.5
    *out_scale = 1.0f; // Scaling deaktiviert
}

Matrix* matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                      ssize_t y_offset) {
    scale = 1.0; // TODO: remove scaling
    assert(sigma > 0 && "Sigma must be positive");
    sigma = (sigma < 2.0) ? 2.0 : sigma;
    Matrix* matrix = matrix_new(width, height);
    if (matrix == NULL) return NULL;

    const ssize_t width_half = width >> 1;
    const ssize_t height_half = height >> 1;

    x_offset += width_half;
    y_offset += height_half;

    //prozess distribution subsample_matrix
    size_t index = 0;
    for (ssize_t y = 0; y < matrix->height; y++) {
        for (ssize_t x = 0; x < matrix->width; x++) {
            const double distance_squared = euclid_sqr(x_offset, y_offset, x, y);
            const double gaussian_value = exp(-distance_squared / (2 * pow(sigma, 2)));
            matrix->data[index++] = gaussian_value;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < matrix->len; ++i) {
        sum += matrix->data[i];
    }

    //printf("%f\n", sum);

    for (int i = 0; i < matrix->len; ++i) {
        matrix->data[i] /= sum;
    }

    return matrix;
}

Matrix* matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                  ssize_t y_offset) {
    scale = 1.0; // TODO: remove scaling
    Matrix* matrix = matrix_generator_gaussian_pdf(width, height, sigma, scale, x_offset, y_offset);
    if (x_offset != 0 || y_offset != 0) {
        // Mische Gaußverteilung mit Gleichverteilung
        const double alpha = 0.001;
        const double uniform_value = 1.0 / (double)(width * height);

        for (int i = 0; i < matrix->len; ++i) {
            matrix->data[i] = (1.0 - alpha) * matrix->data[i] + alpha * uniform_value;
        }

        //  normalisieren
        double sum = 0.0;
        for (int i = 0; i < matrix->len; ++i) sum += matrix->data[i];
        for (int i = 0; i < matrix->len; ++i) matrix->data[i] /= sum;
    }

    return matrix;
}

