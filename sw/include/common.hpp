#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <iomanip>

#include "types.h"

template <typename T>
void print_2d(const T *p, size_t nx, size_t ny, int width=5)
{
  printf("Dimensions: %ld x %ld\n", nx, ny);
  size_t x, y;
  for (y = 0; y < ny; y++) {
    for (x = 0; x < nx; x++)
      std::cout << std::fixed << std::setprecision(1) << \
      std::setw(width) << std::setfill('0') << p[nx * y + x] << " ";
    printf("\n");
  }
}

#endif // COMMON_H