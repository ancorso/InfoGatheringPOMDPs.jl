#!/bin/bash

echo "Determining the operating system..."
os_name=$(uname)

echo "Operating System detected: $os_name"

# Determine the number of CPU threads based on the operating system
if [ "$os_name" = "Linux" ]; then
    num_threads=$(nproc)
elif [ "$os_name" = "Darwin" ]; then
    # Darwin is the operating system name for macOS
    num_threads=$(sysctl -n hw.ncpu)
else
    echo "Unsupported Operating System. Defaulting to 1 thread."
    num_threads=1
fi

echo "Number of CPU threads available: $num_threads"

export JULIA_NUM_THREADS=$num_threads
echo "Setting Julia to use $JULIA_NUM_THREADS threads"

echo "Running Geothermal Example Experiments..."

# Run with first argument "geo"
julia1.9 examples/geothermal_example.jl both

# # Run with second argument "econ"
# julia1.9 examples/geothermal_example.jl geo

# # Run with third argument "both"
# julia1.9 examples/geothermal_example.jl econ

echo "Script execution completed."