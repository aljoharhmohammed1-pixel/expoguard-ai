"""ExpoGuard AI HTTP service."""
import os

# Inference must never install optional packages in response to untrusted input.
os.environ['YOLO_AUTOINSTALL'] = 'false'
