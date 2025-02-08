/**
 * @file intersect.c
 * 
 * @author Jaime Boal Martín-Larrauri (jboal@comillas.edu)
 * 
 * @version 1.0.0
 * 
 * @date 2020-12-30
 * 
 * @brief Library to compute intersections.
 */

// ----------------------------------------------------------------------------

#include "intersect.h"
#include <math.h>

// ----------------------------------------------------------------------------
// ----------------------- PRIVATE FUNCTION PROTOTYPES ------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Computes the intersection point of a line with the y axis.
 * 
 * @param[in] slope Slope of the line.
 * @param[in] x x coordinate of any point of the line.
 * @param[in] y y coordinate of any point of the line.
 * 
 * @return Intercept.
 */
double compute_intercept(double slope, double x, double y);

// ----------------------------------------------------------------------------

/**
 * @brief Computes the slope of a line defined by two points.
 * 
 * @param[in] x0 x coordinate of the first point.
 * @param[in] y0 y coordinate of the first point.
 * @param[in] x1 x coordinate of the second point.
 * @param[in] y1 y coordinate of the second point.
 * 
 * @return Slope; INFINITY if the line is vertical.
 */
double compute_slope(double x0, double y0, double x1, double y1);

// ----------------------------------------------------------------------------

/**
 * @brief Computes the intersection point between two lines.
 * 
 * @param[out] xi x coordinate of the intersection point. NAN if it does not exist.
 * @param[out] yi y coordinate of the intersection point. NAN if it does not exist.
 * @param[in]  x0 Initial x coordinate of the first line.
 * @param[in]  y0 Initial y coordinate of the first line.
 * @param[in]  x1 Final x coordinate of the first line.
 * @param[in]  y1 Final y coordinate of the first line.
 * @param[in]  x2 Initial x coordinate of the second line.
 * @param[in]  y2 Initial y coordinate of the second line.
 * @param[in]  x3 Final x coordinate of the second line.
 * @param[in]  y3 Final x coordinate of the second line.
 */
void intersect(
    double* xi, double* yi, 
    double x0, double y0, double x1, double y1, 
    double x2, double y2, double x3, double y3
);

// ----------------------------------------------------------------------------

/**
 * @brief Rounds a double to 6 decimal places
 * 
 * @param[in] x Number.
 * 
 * @return Rounded number.
 */
double round6(double x);

// ----------------------------------------------------------------------------
// ----------------------------- PUBLIC FUNCTIONS -----------------------------
// ----------------------------------------------------------------------------

bool segment_intersect(
    double* xi, double* yi, 
    double x0, double y0, double x1, double y1, 
    double x2, double y2, double x3, double y3
) {
    intersect(xi, yi, x0, y0, x1, y1, x2, y2, x3, y3);

    if (isnan(*xi) || isnan(*yi))
        return false;

    double xr0 = round6(x0);  
    double xr1 = round6(x1);
    double xr2 = round6(x2);
    double xr3 = round6(x3);
    double xri = round6(*xi);
    double yr0 = round6(y0);  
    double yr1 = round6(y1);
    double yr2 = round6(y2);
    double yr3 = round6(y3);
    double yri = round6(*yi);

    if (xr0 < xr1) {
        if (xri < xr0 || xri > xr1)
            return false;
    }
    else {
        if (xri > xr0 || xri < xr1)
            return false;
    }

    if (yr0 < yr1) {
        if (yri < yr0 || yri > yr1)
            return false;
    }
    else {
        if (yri > yr0 || yri < yr1)
            return false;
    }

    if (xr2 < xr3) {
        if (xri < xr2 || xri > xr3)
            return false;
    }
    else {
        if (xri > xr2 || xri < xr3)
            return false;
    }

    if (yr2 < yr3) {
        if (yri < yr2 || yri > yr3)
            return false;
    }
    else {
        if (yri > yr2 || yri < yr3)
            return false;
    }

    return true;
}

// ----------------------------------------------------------------------------
// ---------------------------- PRIVATE FUNCTIONS -----------------------------
// ----------------------------------------------------------------------------

double compute_intercept(double slope, double x, double y) {
    return y - slope * x;
}

// ----------------------------------------------------------------------------

double compute_slope(double x0, double y0, double x1, double y1) {
    double slope = INFINITY;

    if (fabs(x1 - x0) >= 1e-6)
        slope = (y1 - y0) / (x1 - x0);

    return slope;
}

// ----------------------------------------------------------------------------

void intersect(
    double* xi, double* yi, 
    double x0, double y0, double x1, double y1, 
    double x2, double y2, double x3, double y3
) {
    double m1 = compute_slope(x0, y0, x1, y1);
    double m2 = compute_slope(x2, y2, x3, y3);
    double b1, b2;
    *xi = NAN;
    *yi = NAN;

    if (!isinf(m1) && !isinf(m2)) {
        b1 = compute_intercept(m1, x0, y0);
        b2 = compute_intercept(m2, x2, y2);

        if (fabs(m1 - m2) >= 1e-6) {
            *xi = (b2 - b1) / (m1 - m2);
        }

        *yi = m1 * *xi + b1;
    }
    else if (isinf(m1) && !isinf(m2)) {
        b2 = compute_intercept(m2, x2, y2);
        *xi = x0;
        *yi = m2 * *xi + b2;
    }
    else if (isinf(m2) && !isinf(m1)) {
        b1 = compute_intercept(m1, x0, y0);
        *xi = x2;
        *yi = m1 * *xi + b1;
    }
}

// ----------------------------------------------------------------------------

double round6(double x) {
    return (int)(x * 1e6 + 0.5) / 1e-6; 
}
