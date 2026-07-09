# The single broadcast the Base Station emits to heal a breach. Carries the new
# geo-fence for every remaining active sentinel so the fleet reallocates as a unit.
std_msgs/Header header

# The breach this reallocation responds to (for traceability/logging).
string triggered_by

# Expected reward of the chosen reallocation under the RL value estimate.
float32 expected_reward

# One directive per remaining active sentinel.
GeoFenceDirective[] directives
