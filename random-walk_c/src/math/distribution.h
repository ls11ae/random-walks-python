#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const double mean;
    const double stddev;
    const double _a;
    const double _b;
} NormalDistribution;

NormalDistribution *normal_distribution_new(double mean, double stddev);

double normal_distribution_generate(NormalDistribution *dist, double x);

double normal_pdf(double mean, double stddev, double x);

typedef struct {
    const int k; // Freiheitsgrade
    const double _a;
} ChiDistribution;

ChiDistribution *chi_distribution_new(int k);

double chi_distribution_generate(ChiDistribution *dist, double x);

double chi_pdf(int k, double x);

typedef struct {
    double period;
} WrappedDistribution;

double wrapped_generate(WrappedDistribution *dist, double x);

double wrapped_normal_pdf(double mu, double rho, double x);

double wrapped_normal_approx_pdf(double mu, double rho, double x);

int discrete_pdf(const double *probabilities, size_t size);

int discrete_distribution(double *probs, size_t size);

#ifdef __cplusplus
}
#endif
