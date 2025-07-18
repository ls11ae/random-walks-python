#include <math.h>
#include <stdlib.h>  // Für malloc, free, NULL

#include "distribution.h"

#include <stdio.h>

NormalDistribution *normal_distribution_new(double mean, double stddev) {
    NormalDistribution *dist = (NormalDistribution *) malloc(sizeof(NormalDistribution));
    if (!dist) return NULL;

    *(double *) &dist->mean = mean;
    *(double *) &dist->stddev = stddev;
    *(double *) &dist->_a = 1 / stddev * sqrt(2 * M_PI);
    *(double *) &dist->_b = -1 / (2 * stddev * stddev);
    return dist;
}

double normal_distribution_generate(NormalDistribution *dist, double x) {
    const double c = (x - dist->mean);
    return dist->_a * exp(c * c * dist->_b);
}

double normal_pdf(double mu, double sigma, double x) {
    // Calculate the PDF value for the normal distribution at x
    double factor = 1.0 / (sigma * sqrt(2 * M_PI));
    double exponent = -0.5 * pow((x - mu) / sigma, 2);
    return factor * exp(exponent);
};

ChiDistribution *chi_distribution_new(int k) {
    ChiDistribution *dist = (ChiDistribution *) malloc(sizeof(ChiDistribution));
    if (!dist) return NULL;
    *(int *) &dist->k = k;
    *(double *) &dist->_a = 1 / (pow(2.0, k * 0.5 - 1.0) * tgamma(k * 0.5));
    return dist;
}

double chi_distribution_generate(ChiDistribution *dist, double x) {
    if (x <= 0) return 0.0;
    const double b = pow(x, dist->k - 1) * exp(-x * x * 0.5);
    return b * dist->_a;
}

double chi_pdf(const int k, const double x) {
    if (x <= 0) return 0.0; // PDF ist nur für x ≥ 0 definiert
    const double numerator = pow(x, k - 1) * exp(-x * x / 2);
    const double denominator = pow(2, k / 2.0 - 1) * tgamma(k / 2.0);
    return numerator / denominator;
}

double wrapped_generate(WrappedDistribution *dist, double x) {
    return 0.0;
}

#define MAX_N 10  // Number of terms to sum (can be adjusted for better precision)

double wrapped_normal_pdf(const double mu, double rho, double x) {
    double pdf_value = 0.0;

    // Summing over n from -MAX_N to MAX_N
    for (int n = -MAX_N; n <= MAX_N; ++n) {
        double diff = x - 2 * M_PI * n - mu; // Compute the difference (x - mu)
        double exp_term = exp(-0.5 * diff * diff / (rho * rho)); // Gaussian exponent
        pdf_value += exp_term; // Sum the exponential terms
    }

    // Normalize the result by dividing by 2 * PI * rho (standard normal distribution factor)
    pdf_value /= (2 * M_PI * rho);

    return pdf_value;
}

/*
double warped_normal(const double mu, const double rho, const double x) {
    double sigma = std::sqrt(-2 * std::log(rho));
    boost::math::normal_distribution<> dist(mu, sigma);
    return boost::math::pdf(dist, x); // mod 2 pi not needed bcs of atan2 (alpha)
}
*/


double wrapped_normal_approx_pdf(double mu, double rho, double x) {
    double sigma = sqrt(-2.0 * log(rho));
    double exponent = -0.5 * pow((x - mu) / sigma, 2);
    double coeff = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    return coeff * exp(exponent);
}

double randfrom(double min, double max) {
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int discrete_distribution(double *probs, size_t size) {
    double total_sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        total_sum += probs[i]; // Berechne die Summe der Wahrscheinlichkeiten
    }

    double random_value = (rand() / (double) RAND_MAX) * total_sum;
    double cumulative_sum = 0.0;

    for (size_t i = 0; i < size; i++) {
        cumulative_sum += probs[i];
        if (random_value <= cumulative_sum) {
            return i; // Rückgabe des Index basierend auf der gewichteten Verteilung
        }
    }

    return -1; // Sollte nie erreicht werden, falls probs korrekt ist
}

int discrete_pdf(const double *probabilities, size_t size) {
    // Generate a random number between 0 and 1
    double rand_val = (double) rand() / RAND_MAX;

    // Traverse the probabilities and pick the corresponding index
    double cumulative_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        cumulative_sum += probabilities[i];
        if (rand_val < cumulative_sum) {
            return i; // Return the index corresponding to the random value
        }
    }

    return -1; // Error case (should never reach here if probabilities sum to 1)
}
