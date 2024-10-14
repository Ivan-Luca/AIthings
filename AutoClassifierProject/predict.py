import cv2
import numpy as np
import tensorflow as tf

def decode(prediction):
    pred = np.argmax(prediction[0])
    if prediction[0][pred] < 0.5:
        return "nuknow"
    match pred:
        case 0:
            return "Nothing"
        case 1:
            return "Wario"
        case 2:
            return "Briquet"
        case 3:
            return "Yoshi"
        case _:
            return "Unknown"


interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()  # Needed before execution!
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

print()
print("Input details:")
print(input)
print()
print("Output details:")
print(output)
print()


input_data = tf.constant(1., shape=[1, 1])

cam = cv2.VideoCapture(2)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"frame width = {frame_width} frame height = {frame_height}")

while True:

    ret, frame = cam.read()
    # Preprocess the frame for MobileNetV3
    cropped_frame = frame[:, 80:560]
    print(cropped_frame.shape)
    input_frame = cv2.resize(cropped_frame, (224, 224))
    #input_frame = np.expand_dims(input_frame, axis=0)
    input_data = tf.constant(input_frame)
    # Make predictions
    interpreter.set_tensor(input['index'], [np.float32(input_frame)])
    interpreter.invoke()
    #interpreter.get_tensor(output).shape
    result = interpreter.get_tensor(output['index'])
    print("Inference output:", result)
    # Display the predictions on the frame
    label = decode(result)
    #print(decoded_predictions[0])
    cv2.putText(cropped_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Client Webcam', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
