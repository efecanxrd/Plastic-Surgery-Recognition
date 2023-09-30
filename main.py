import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Yüz tanıma modeli
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Burun estetiklik modeli
model = load_model("model.h5")

# Kamera ayarları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Yüzü bulma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        # Yüz noktalarını bulma
        landmarks = predictor(gray, face)
        
        # Burun noktalarını alma
        nose_points = np.array([(landmarks.part(31).x, landmarks.part(31).y),
                                (landmarks.part(32).x, landmarks.part(32).y),
                                (landmarks.part(33).x, landmarks.part(33).y),
                                (landmarks.part(34).x, landmarks.part(34).y),
                                (landmarks.part(35).x, landmarks.part(35).y),
                                (landmarks.part(36).x, landmarks.part(36).y)])
        
        # Burun bölgesini kırpma
        x1, y1 = np.min(nose_points, axis=0)
        x2, y2 = np.max(nose_points, axis=0)
        
        # Burun bölgesinin sınırlarını genişletme
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1-w//3)
        y1 = max(0, y1-h//2)
        x2 = min(frame.shape[1]-1, x2+w//2)
        y2 = min(frame.shape[0]-1, y2+h//2)
        
        nose_img = frame[y1:y2, x1:x2]
        
        # Burun estetiklik tahminlemesi
        nose_img = cv2.resize(nose_img, (224, 224))
        nose_img = cv2.cvtColor(nose_img, cv2.COLOR_BGR2RGB)
        nose_img = nose_img.astype(np.float32) / 255.
        nose_img = np.expand_dims(nose_img, axis=0)
        pred = model.predict(nose_img)[0][0]
        
        # Sonuçları ekrana yazdırma
        if pred > 0.5:
            text = "Estetik"
            color = (0, 255, 0)
        else:
            text = "Estetiksiz"
            color = (0, 0, 255)
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
