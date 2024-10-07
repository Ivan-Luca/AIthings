import tensorflow as tf
import cv2
import socket
import struct
import pickle
import numpy as np

# do NOT forget to run webcam_server from windows side (or camcapture)
#
# small program to test mobinetv3 on the webcam

IMG_HEIGHT = 256
IMG_WIDTH = 256 
CHANNELS = 3

input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

print('Loading model with ImageNet weights...')
vgg16_conv_base = tf.keras.applications.MobileNetV3Small(input_shape=input_shape,
                                                    include_top=True, # We will supply our own top.
                                                    weights='imagenet',
                                                   )
# Preprocessing function
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
# Decode predictions function
decode_predictions = tf.keras.applications.mobilenet_v3.decode_predictions



# Setup socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.25.144.1', 8485))  # Replace with your Windows machine's IP
conn = client_socket.makefile('rb')

while True:

    
    # Read the fixed-size header (4 bytes)
    packed_msg_size = conn.read(struct.calcsize("I"))
    if not packed_msg_size:
        print("No message size received. Breaking out.")
        break

    # Unpack the message size
    msg_size = struct.unpack("I", packed_msg_size)[0]


    # Read the frame data based on the size received
    data = b""
    while len(data) < msg_size-4096:
        packet = conn.read(4096)  # Use recv instead of read for better performance
        if not packet:
            print("No packet received. Breaking out.")
            break
        data += packet

    packet = conn.read(msg_size-len(data)) 
    data += packet
    if len(data) != msg_size:
        print(f"Received incomplete frame data. Expected {msg_size}, but got {len(data)}. Breaking out.")
        break

    frame = pickle.loads(data)  # Deserialize the frame

    # Preprocess the frame for MobileNetV3
    cropped_frame = frame[:, 80:560]
    input_frame = cv2.resize(cropped_frame, (256, 256))
    #input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = preprocess_input(input_frame)

    # Make predictions
    predictions = vgg16_conv_base.predict(input_frame)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the predictions on the frame
    label = f"{decoded_predictions[0][1]}: {decoded_predictions[0][2]*100:.2f}%"
    #print(decoded_predictions[0])
    cv2.putText(cropped_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Client Webcam', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()
