# A tracked object's direction in the sky, from the star-tracker calibration.
int32 id
int32 state               # same as TrackDetection.state
float64 x                 # bbox centre in image pixels
float64 y
float64 azimuth_deg       # 0 = true north, 90 = east
float64 elevation_deg     # apparent elevation above the horizon
float64 angular_width_deg # approx. angular size of the bbox
