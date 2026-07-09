# Emitted by a sentinel (via a single encrypted Zenoh micro-burst) when it
# leaves its assigned sector, so the Base Station learns of an unguarded breach.
std_msgs/Header header

# Identity of the reporting sentinel (e.g. "alpha", "beta", "gamma").
string robot_id

# Where the sentinel was when it broke the patrol net.
geometry_msgs/Point last_position

# The polygon of the sector now left unguarded (map/odom frame, meters).
geometry_msgs/Point32[] vacated_sector

# Reason for the breach: "pursuit", "disabled", "comms_loss", ...
string reason
