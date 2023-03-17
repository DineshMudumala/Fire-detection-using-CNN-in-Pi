import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="fire_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read an image from the webcam
    ret, frame = cap.read()
    
    # Preprocess the image
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data - 127.5) / 127.5
    
    # Run inference with the TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Check if fire is detected
    if output_data > 0.5:
        
        # Apply Gaussian filter to enhance the image
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
       
        # Display the image
        cv2.imshow("Fire Detection", frame)
    else:
           
        # Display the image
        cv2.imshow("Fire Detection", frame)
    
    # Wait for a key press and check if it is the 'q' key to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
