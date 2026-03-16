#include <stddef.h>

double dot_c(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}
