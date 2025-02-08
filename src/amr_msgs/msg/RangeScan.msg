# Single scan from an array of range sensors that measure distance within an arc.
# This message is not appropriate for laser scanners. Use LaserScan instead.

# To represent fixed-distance (binary) rangers set min_range == max_range == distance of detection.
# These sensors should follow REP 117 and output -Inf if the object is detected and +Inf otherwise.

# Radiation type enum
uint8 ULTRASOUND = 0
uint8 INFRARED = 1

std_msgs/Header header  # Timestamp in the header is the acquisition time of the first ranger.

uint8 radiation_type    # Type of radiation used by the sensor (ultrasound or infrared) [enum].

float32 field_of_view   # The width of the measurement arc [rad]. The object causing the reading
                        # may be anywhere within [-field_of_view / 2, field_of_view / 2]. 
                        # 0 rad corresponds to the x-axis of the sensor.

float32 min_range       # Minimum range value [m].
float32 max_range       # Maximum range value [m].

float32[] ranges        # Range data [m]. Out-of-range values should be discarded.