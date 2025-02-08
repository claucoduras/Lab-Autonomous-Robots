/**
 * @file intersect.h
 * 
 * @author Jaime Boal Martín-Larrauri (jboal@comillas.edu)
 * 
 * @version 1.0.0
 * 
 * @date 2020-12-30
 * 
 * @brief Library to compute intersections.
 * 
 * COMPILATION INSTRUCTIONS
 * 
 * Windows (Install MinGW-W64-builds from http://mingw-w64.org/doku.php)
 * gcc -c intersect.c -O3
 * gcc -shared -o libintersect.dll intersect.o
 * 
 * macOS (2 options)
 * a) clang -dynamiclib intersect.c -o libintersect.dylib -O3
 * b) gcc -dynamiclib intersect.c -o libintersect.dylib -O3
 * 
 * Linux
 * gcc -shared -o libintersect.so -fPIC intersect.c -Wall -g -O3
 */

#ifndef _INTERSECT_H
#define _INTERSECT_H

#include <stdbool.h>

/**
 * @brief Computes the intersection point between two segments.
 * 
 * @param[out] xi x coordinate of the intersection point. NAN if it does not exist.
 * @param[out] yi y coordinate of the intersection point. NAN if it does not exist.
 * @param[in]  x0 Initial x coordinate of the first segment.
 * @param[in]  y0 Initial y coordinate of the first segment.
 * @param[in]  x1 Final x coordinate of the first segment.
 * @param[in]  y1 Final y coordinate of the first segment.
 * @param[in]  x2 Initial x coordinate of the second segment.
 * @param[in]  y2 Initial y coordinate of the second segment.
 * @param[in]  x3 Final x coordinate of the second segment.
 * @param[in]  y3 Final x coordinate of the second segment.
 * 
 * @return true if the intersection exists and is within both segments; false otherwise.
 */
bool segment_intersect(
    double* xi, double* yi, 
    double x0, double y0, double x1, double y1, 
    double x2, double y2, double x3, double y3
);

#endif
