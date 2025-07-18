#include "ScalarMapping.h"

ScalarMapping *Point2D_new(const double val, const size_t ind) {
    ScalarMapping *point = malloc(sizeof(ScalarMapping));
    if (point != NULL) {
        point->value = val;
        point->index = ind;
    }
    return point;
}

void set_values(ScalarMapping *point, double val, size_t ind) {
    if (point != NULL) {
        point->value = val;
        point->index = ind;
    }
}

void scalar_mapping_delete(ScalarMapping *self) {
    free(self);
}


