# Archery
#ARCHITECTURE
Hub Device
   receiving data from a variable number of cameras (USB, CSI, remote devices over BT or WiFi)
       may want cameras to collect a video clip of consistent duration (but might be different 
       between cameras) around a shot or a still after a shot
       If local, run as a separate service so that it's equivalent and modular with a remote camera
   Run UI to present data to the user (as a web page?)
   Receive data from other sensors or BT devices
       Wii balance board
      MantisX
   Maintain a master clock for everything
   Save everything to a structured database

Module for managing a camera
   
Target Scanner Device:
# initialise hardware - camera, microphone, IMU
# take an image (does this need several to average out lighting, etc)
# detect the target(s)
#   use the algorithm I've done in findcolor and on the other PC to identify the centre of the target
# detect orientation of the target(s) (which way is vertical up?)
# show these to the user and confirm to start shooting
# request user input target size and distance
# map target space onto a nominal target and show this on the screen
# wait for an arrow loop
    # detect an arrow has arrived (sound in microphone?  Vibration in IMU? Just use the images? command from base?)
    # find the arrow
        #edge detect?  average some frames? frame difference to isolate the new arrow?
        #identify whose arrow it is
    # calculate the arrow's position on the nominal target
    # score the arrow and mark it on the on-screen nominal target
# break out if user presses the 'end' button
    # show scores and request judging of linebreakers
    # record original guess and actual result for later analysis
    # send result to the control station


High Speed Camera Device:


