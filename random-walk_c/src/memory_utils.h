#pragma once
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void memory_size_print(double size_in_bytes) {
    const char* size_units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int unit_index = 0;
    while (size_in_bytes >= 1024 && unit_index < 4) {
        size_in_bytes /= 1024;
        unit_index++;
    }
    printf("%.2f%s", size_in_bytes, size_units[unit_index]);
}

