import cv2

# Load cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

cap = cv2.VideoCapture(0)

blink_count = 0
eye_closed_frames = 0
EYE_CLOSED_THRESHOLD = 3  # frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h//2, x:x+w]   # upper face only
        roi_color = frame[y:y+h//2, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                          (255, 0, 0), 2)

        # ---- BLINK LOGIC ----
        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= EYE_CLOSED_THRESHOLD:
                blink_count += 1
            eye_closed_frames = 0

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
