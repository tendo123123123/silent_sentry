#!/usr/bin/env python3

import os
import sys


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <bot_name>", file=sys.stderr)
        sys.exit(1)

    bot_name = sys.argv[1]

    # Locate the template relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, os.pardir, "config")
    template_path = os.path.join(config_dir, "gazebo_bridge.yaml")
    output_path = os.path.join(config_dir, f"gz_bridge_{bot_name}.yaml")

    with open(template_path, "r") as f:
        template = f.read()

    resolved = template.replace("{bot_name}", bot_name)

    with open(output_path, "w") as f:
        f.write(resolved)

    # Return absolute path for the launch system
    print(os.path.abspath(output_path), end="")


if __name__ == "__main__":
    main()
