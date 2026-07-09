# A single reassigned patrol sector for one sentinel, produced by the
# Base Station RL policy during elastic geo-fence healing.
string robot_id

# New (stretched) patrol polygon for this sentinel (map/odom frame, meters).
geometry_msgs/Point32[] polygon

# Optional SBLP parameter overrides to tune exploration in the enlarged sector.
# A non-positive value means "leave unchanged".
float32 levy_beta
float32 l_max
