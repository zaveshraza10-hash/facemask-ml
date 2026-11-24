import cv2
import joblib
from ml_features import extract_features
MODEL_PATH   = "mask_detector_ml.joblib"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
LABELS = {0: ("Mask", (0, 200, 0)), 1: ("No Mask", (0, 0, 220))}
print("[INFO] Loading model...")
model    = joblib.load(MODEL_PATH)
detector = cv2.CascadeClassifier(FACE_CASCADE)
has_proba = hasattr(model, "predict_proba")
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press Q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        feat = extract_features(face).reshape(1, -1)
        pred = int(model.predict(feat)[0])
        name, color = LABELS[pred]
        text = name
        if has_proba:
            text += f": {model.predict_proba(feat)[0][pred]*100:.0f}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, max(0, y-30)), (x+w, y), color, -1)
        cv2.putText(frame, text, (x+5, max(15, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow("Face Mask Detector (ML) [press Q to quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()