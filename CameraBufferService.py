# This is working, but the camera setup is still not behaving as expected. The standard config gives me only a very small frame
# I got Chat GPT to force the settings to match the libcamera-hello setup for now.  Will need to improve this later on 
# Tested CPU, RAM and GPU usage and it's fine.  1 camera at this level uses ~15% cpu and ~ 400MB RAM.
# Will need to replace the user input() with a trigger from another thread

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
import time

# Initialize Picamera2
picam2 = Picamera2()

# Match `libcamera-hello` resolution & format
video_config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},  # Matches libcamera-hello default
    lores={"size": (640, 480), "format": "YUV420"},  # Low-res preview (optional)
    controls={"FrameRate": 30}  # Match FPS
)
picam2.configure(video_config)
picam2.start()  # Start camera

# Set up encoder and circular buffer output
encoder = H264Encoder()
circular_buffer = CircularOutput(buffersize=300)  # Buffer for ~10 sec at 30 FPS
picam2.start_recording(encoder, circular_buffer)

print("Recording to circular buffer... Press Enter to save and exit.")

input()  # Wait for user input to save the buffer contents

# Save the buffered video to a file
output_file = "event_capture.h264"
circular_buffer.fileoutput = output_file
circular_buffer.start()  # Start writing buffered frames to file

time.sleep(1)  # Ensure buffer is fully flushed before stopping
# Stop recording and cleanup
picam2.stop_recording()
picam2.close()

