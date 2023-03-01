import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1cT5yLKXRQOh7lAuq5oNeDVgav5NBAxE2',
                                    dest_path='./FaceDetection/ft.h5',
                                    unzip=False)

face_tracker = load_model('./FaceDetection/ft.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[315:765, 735:1185,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resized = tf.image.resize(rgb, (120,120))
    
    
    y_pred = face_tracker.predict(np.expand_dims(resized/255,0))
    sample_coords = y_pred[1][0]
    
    if y_pred[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('Face Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()