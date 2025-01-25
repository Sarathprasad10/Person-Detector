# Person-Detector



### Purpose
The goal of the code is to load an image, use a pre-trained MobileNet SSD model to detect people (or other objects), and display the image with bounding boxes around the detected persons. The MobileNet SSD model was trained to recognize multiple object classes (e.g., cars, bicycles, persons) from an image.

### Libraries and Files
- **OpenCV (`cv2`)**: OpenCV is used for image processing and computer vision tasks such as reading, resizing images, and drawing bounding boxes.
- **NumPy**: Used for array manipulation and mathematical operations (e.g., scaling the bounding box coordinates).
- **imutils**: A helper library used to resize the image.

### Key Components
1. **Prototxt and Caffe Model**:
   - `protopath`: Path to the `.prototxt` file, which defines the model's architecture.
   - `modelpath`: Path to the pre-trained `.caffemodel` file that contains the weights for the model.
   - The `detector` object is created by loading the MobileNet SSD model using `cv2.dnn.readNetFromCaffe()`, which takes both the prototxt and model files.

2. **Class Labels (`CLASSES`)**:
   - A list containing object categories that the MobileNet SSD model can recognize. Each index corresponds to a specific class (e.g., index 15 corresponds to "person").

3. **Main Function** (`main()`):
   - **Image Loading**: The input image (`people.jpg`) is read using `cv2.imread()`. If the image is not found or loaded, an error message is displayed.
   - **Image Resizing**: The image is resized using `imutils.resize()` to ensure that it fits within the desired size for processing (600 pixels wide in this case). The resized image dimensions (`H` and `W`) are printed.
   - **Blob Creation**: The image is converted into a blob using `cv2.dnn.blobFromImage()`. This blob is used as input to the model for detection. The blob scales and normalizes the image, making it compatible with the model.
   - **Detection**: The blob is passed to the model using `detector.setInput(blob)` and detections are made with `detector.forward()`. The output is stored in `person_detections`, which contains the detected objects and their properties.
   - **Looping Through Detections**: The loop iterates over all detected objects and checks the confidence for each detection. If the confidence is greater than 0.2 (indicating a reliable detection), the code checks if the object is a "person" using the `CLASSES` list.
     - The bounding box coordinates for the person are extracted and scaled according to the image dimensions (`W`, `H`).
     - The bounding box is drawn on the image using `cv2.rectangle()`.
   - **Displaying the Image**: The result image, with bounding boxes around detected persons, is displayed using `cv2.imshow()`. The window waits for any key press before closing.

4. **Error Handling**:
   - If no person is detected (i.e., the confidence for all detections is below 0.2), the program prints "No persons detected!".
   - If the `cv2.imshow()` function fails to open a window (due to missing dependencies), an error related to window display might be raised, as seen in your error logs.

### Expected Output
- The output image will show bounding boxes around any persons detected in the input image (`people.jpg`).
- If the model detects multiple people, each person will be highlighted with a red rectangle.
- The confidence for each detection is printed in the console, indicating how confident the model is in each detection.
- If no person is detected, the message "No persons detected!" will be printed.

