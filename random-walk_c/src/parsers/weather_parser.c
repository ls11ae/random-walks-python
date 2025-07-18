#include "weather_parser.h"

#include <stdio.h>

WeatherEntry* weather_entry_new(float temperature,
                                int humidity,
                                float precipitation,
                                float wind_speed,
                                float wind_direction,
                                float snow_fall,
                                int weather_code,
                                int cloud_cover) {
    WeatherEntry* weather_entry = malloc(sizeof(WeatherEntry));
    weather_entry->temperature = temperature;
    weather_entry->humidity = humidity;
    weather_entry->precipitation = precipitation;
    weather_entry->wind_speed = wind_speed;
    weather_entry->wind_direction = wind_direction;
    weather_entry->snow_fall = snow_fall;
    weather_entry->weather_code = weather_code;
    weather_entry->cloud_cover = cloud_cover;
    return weather_entry;
}

WeatherTimeline* weather_timeline_new(size_t time) {
    WeatherTimeline* weather_entry = malloc(sizeof(WeatherTimeline));
    weather_entry->data = malloc(sizeof(WeatherEntry*) * time);
    weather_entry->length = time;
    return weather_entry;
}

WeatherGrid* weather_grid_new(const size_t height, const size_t width) {
    WeatherGrid* timeline = malloc(sizeof(WeatherGrid));
    timeline->height = height;
    timeline->width = width;
    WeatherTimeline*** weather_entries = malloc(sizeof(WeatherTimeline**) * height);
    for (int i = 0; i < height; i++) {
        weather_entries[i] = malloc(sizeof(WeatherTimeline*) * width);
    }
    timeline->entries = weather_entries;
    return timeline;
}

void weather_entry_print(const WeatherEntry* entry) {
    printf("Temperature: %.2f\n", entry->temperature);
    printf("Humidity: %i\n", entry->humidity);
    printf("Precipitation: %.2f\n", entry->precipitation);
    printf("Wind speed: %.2f\n", entry->wind_speed);
    printf("Wind direction: %.2f\n", entry->wind_direction);
    printf("Snow fall: %.2f\n", entry->snow_fall);
    printf("Weather code: %d\n", entry->weather_code);
    printf("Cloud cover: %d\n", entry->cloud_cover);
}

void weather_timeline_print(const WeatherTimeline* timeline) {
    for (int i = 0; i < timeline->length; i++) {
        weather_entry_print(timeline->data[i]);
    }
}

void weather_grid_print(const WeatherGrid* weather_grid) {
    for (int y = 0; y < weather_grid->height; y++) {
        for (int x = 0; x < weather_grid->width; x++) {
            weather_timeline_print(weather_grid->entries[y][x]);
        }
    }
}
