#!/bin/bash
# Start Xvfb in the background on display :99
Xvfb :99 -screen 0 1920x1440x24 +extension GLX +render -noreset &
XVFB_PID=$! # Get its PID
export DISPLAY=:99
echo "Waiting for Xvfb to start..."
sleep 5 # Give Xvfb a moment to initialize

# Check if glxinfo works (optional debug)
glxinfo -B

# Execute the main command passed to the script
echo "Running command: $@"
exec "$@"

# Cleanup (might not be reached if exec replaces the process)
kill $XVFB_PID