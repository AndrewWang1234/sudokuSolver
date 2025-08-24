import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully!")

# Capture a few frames and display them
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Webcam", frame)
    else:
        print("Error capturing frame.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()