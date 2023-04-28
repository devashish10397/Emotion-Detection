import cv2
import numpy as np
import tensorflow as tf

# Load a saved model
model = tf.keras.models.load_model('cnn_raf_fer2013_model_stopped.h5')

# Define the face cascade and emotions
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

no_face_detection_alert = "Cannot Detect Face"
low_confidence_alert = "Cannot Detect Emotion"

# Define the predict_emotion function
def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_AREA)
            # if a face is detected make a prediction
            if np.sum([face])!=0:
                face = face.astype('float')/255.0
                face = tf.keras.utils.img_to_array(face)
                face = np.expand_dims(face, axis=0)
                prediction = model.predict(face)
                # if the confidence level of the model is greater than 50% display emotion to user
                if any(prob >.5 for prob in prediction[0]):
                    emotion = emotions[np.argmax(prediction)]
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                # else notify user that the emotion can not detected
                else:
                    cv2.putText(frame, low_confidence_alert, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255),
                                2)
            # if no face detected then notify the user that a face is not detected
            else:
                cv2.putText(frame, no_face_detection_alert, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2)


    return frame


# Start the video capture and emotion detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        frame = predict_emotion(frame)
        cv2.imshow('Live Facial Emotion Detection', frame)
        
    # end program when press letter q on keyboard
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

