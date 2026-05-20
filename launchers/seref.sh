#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package seref.py _config_path:="$(rospack find my_package)/config/config.yaml"
dt-launchfile-join