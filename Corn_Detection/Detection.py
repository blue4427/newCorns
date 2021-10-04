import numpy as np
import cv2
import time
from tracker import EuclideanDistTracker

tracker_corn = EuclideanDistTracker()
tracker_hole = EuclideanDistTracker()

#Load Yolo
net = cv2.dnn.readNet("weights2.weights", "custom-yolov4-tiny-detector.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))


#Loading image
cap = cv2.VideoCapture("vid1_Trim.mp4")
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
next_corn_id = 0
next_hole_id = 0
while True:
    _, roi = cap.read()
    frame_id += 1
    height, width, channel = roi.shape

    #Detecting Objects
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (320, 320), (0,0,0),True, crop = False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Showing info on screen
    class_ids = []
    confidences = []
    boxes= []


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id=np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.5)
    detections = []
    for i in range (len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(roi, (x, y), (x + w, y + h), color,2)

            detections.append([x,y,w,h])
    if label == 'Corn':
        corn_boxes_id = tracker_corn.update(detections)
        for corn_box_id in corn_boxes_id:
            x,y,w,h,corn_id = corn_box_id
            if corn_id > next_corn_id:
                next_corn_id = corn_id
            cv2.putText(roi,"Corn " + str(corn_id), (x, y), font, 2, color, 2)

    elif label == 'Hole':
        hole_boxes_id = tracker_hole.update(detections)
        for hole_box_id in hole_boxes_id:

            x, y, w, h, hole_id = hole_box_id
            if hole_id > next_hole_id:
                next_hole_id = hole_id

            cv2.putText(roi, "Hole " + str(hole_id), (x, y), font, 2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = (frame_id/elapsed_time)
    cv2.putText(roi, "FPS " + str(round(fps,2)), (10, 30), font, 2, (0,0,0), 2)
    cv2.putText(roi, "CORN COUNT = " + str(next_corn_id), (10, 60), font, 2, (0,0,0), 2)
    cv2.putText(roi, "HOLE COUNT = " + str(next_hole_id), (10, 90), font, 2, (0,0,0), 2)
    cv2.imshow("image", roi)
    #cv2.imshow("roi",roi)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Counted Corns:", next_corn_id)
print("Counted Holes:", next_hole_id)

