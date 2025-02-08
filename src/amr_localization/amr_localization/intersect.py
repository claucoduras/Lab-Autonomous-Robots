import math

from typing import Optional


class Intersect:
    """Class to find the intersection point between lines and segments.

    This implementation is faster than built in alternatives such as those of the shapely package.

    See also: https://www.codeproject.com/Tips/864704/Python-Line-Intersection-for-Pygame

    """

    def __init__(self):
        """Intersect class initializer."""
        pass

    def intersect(
        self, line1: list[tuple[float, float]], line2: list[tuple[float, float]]
    ) -> tuple[float, float]:
        """Computes the intersection point between two lines.

        Args:
            line1: First line. Defined by two points [(x0, y0), (x1, y1)].
            line2: Second line. Defined by two points [(x2, y2), (x3, y3)].

        Returns: (xi, yi) intersection point.

        """
        m1 = self._slope(line1[0], line1[1])
        m2 = self._slope(line2[0], line2[1])

        if not math.isinf(m1) and not math.isinf(m2):
            b1 = self._intercept(m1, line1[0])
            b2 = self._intercept(m2, line2[0])

            try:
                x = (b2 - b1) / (m1 - m2)
            except ZeroDivisionError:
                x = math.nan

            y = m1 * x + b1
        elif math.isinf(m1) and not math.isinf(m2):
            b2 = self._intercept(m2, line2[0])
            x = line1[0][0]
            y = m2 * x + b2
        elif math.isinf(m2) and not math.isinf(m1):
            b1 = self._intercept(m1, line1[0])
            x = line2[0][0]
            y = m1 * x + b1
        else:
            x = math.nan
            y = math.nan

        return x, y

    def segment_intersect(
        self, segment1: list[tuple[float, float]], segment2: list[tuple[float, float]]
    ) -> Optional[tuple[float, float]]:
        """Computes the intersection point between two segments.

        Args:
            segment1: First segment. Defined by two points [(x0, y0), (x1, y1)].
            segment2: Second segment. Defined by two points [(x2, y2), (x3, y3)].

        Returns:
            (xi, yi) intersection point;
            None if the segments are parallel or the intersection is not contained in both segments.

        """
        point = self.intersect(segment1, segment2)

        if math.isnan(point[0]) or math.isnan(point[1]):
            return None

        for i in range(2):
            # Round variables to avoid issues with float precision
            pt = round(point[i], 6)
            l11 = round(segment1[0][i], 6)
            l12 = round(segment1[1][i], 6)

            if l11 < l12:
                if pt < l11 or pt > l12:
                    return None
            else:
                if pt > l11 or pt < l12:
                    return None

            l21 = round(segment2[0][i], 6)
            l22 = round(segment2[1][i], 6)

            if l21 < l22:
                if pt < l21 or pt > l22:
                    return None
            else:
                if pt > l21 or pt < l22:
                    return None

        return point

    @staticmethod
    def _intercept(slope: float, p: tuple[float, float]) -> float:
        """Computes the intersection point of a line with the y axis.

        Args:
            slope: Slope of the line.
            p: Any point of the line.

        Returns:
            float: Intercept.

        """
        return p[1] - slope * p[0]

    @staticmethod
    def _slope(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Computes the slope of a line defined by two points.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            float: Slope; inf if the line is vertical.

        """
        try:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        except ZeroDivisionError:
            slope = float("inf")

        return slope
