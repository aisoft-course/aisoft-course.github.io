#include <stdio.h>

void vec_elem_mul(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void vec_elem_add(double* a, double* b, double* c, int n) {
    for (int i=0; i<n; i++) { 
        c[i] = a[i] + b[i];
    }
}
