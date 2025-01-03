# This is partially working, the data flow seems to be working correctly when showing the raw numbers, but the spectrograph isn't working as expected.
# Next step is probably to use the HW to make some recordings of shooting and then analyse these files and set up a scheme to identify when a shot is made
# then use this as a trigger to another thread
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Audio parameters
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono recording
RATE = 44100  # Sample rate in Hz

# Initialize PyAudio
p = pyaudio.PyAudio()

# Check available devices (for debugging)
print("Available devices:")
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

# Specify the input device index (replace with your actual device index)
device_index = 1  # Replace with the correct index from the list
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=device_index, frames_per_buffer=CHUNK)

# Set up the plot
fig, ax = plt.subplots()
spectrogram = np.zeros((100, CHUNK // 2 + 1))  # History buffer for the spectrogram, size should be 513 for 1024 samples
img = ax.imshow(spectrogram, aspect='auto', cmap='inferno', origin='lower', extent=[0, RATE // 2, 0, 100], vmin=0, vmax=100)
fig.colorbar(img, ax=ax)

ax.set_title("Live Spectrogram")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Time")

# Function to update the spectrogram
def update(frame):
    global spectrogram
    audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    fft_data = np.abs(np.fft.rfft(audio_data))  # Compute FFT
 #   print(fft_data[:10])  # Print the first few values to see if FFT is computed correctly
    fft_data = 20 * np.log10(fft_data)  # Convert to dB scale
    
    spectrogram = np.roll(spectrogram, -1, axis=0)  # Shift spectrogram up
    spectrogram[-1, :] = fft_data  # Add new FFT data

    img.set_array(spectrogram)
    return img,

# Animate the spectrogram
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

# Show the spectrogram
plt.show()

# Close the stream when done
stream.stop_stream()
stream.close()
p.terminate()
