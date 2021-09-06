#!/bin/bash

echo "Run multi-ZMQ Server in multi-processing"
read -e -p "input config file path:" cfg
read -p "number of zmq servers:" num

script_name=$(printf 'python zmq/server.py %s --num %d' "$cfg" "$num")
echo "run script: $script_name"
$script_name