import cv2
import numpy as np
import imutils

protopath = r'C:\Users\LENOVO\Desktop\personDetection\MobileNetSSD_deploy.prototxt'
modelpath = r'C:\Users\LENOVO\Desktop\personDetection\MobileNetSSD_deploy.caffemodel'
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def main():
    image_path = r'C:\Users\LENOVO\Desktop\personDetection\people.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    else:
        print(f"Image loaded successfully: {image.shape}")

    image = imutils.resize(image, width=600)
    (H, W) = image.shape[:2]
    print(f"Resized Image dimensions: {H}x{W}")

    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)
    print(f"Blob created: {blob.shape}")

    detector.setInput(blob)
    person_detections = detector.forward()
    print(f"Number of detections: {person_detections.shape[2]}")

    if person_detections.shape[2] == 0:
        print("No persons detected!")

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        print(f"Confidence for detection {i}: {confidence}")

        if confidence > 0.2:
            print(f"Detection {i} with confidence: {confidence}")
            idx = int(person_detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")
            print(f"Person detected at [{startX}, {startY}, {endX}, {endY}]")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
