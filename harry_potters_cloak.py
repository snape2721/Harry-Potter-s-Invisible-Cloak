import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture static background (user not in frame!)
for i in range(60):
    ret, background = cap.read()
    if not ret:
        continue
    background = cv2.flip(background, 1)

background = cv2.flip(background, 1)

# Define HSV range for PINK cloak
lower_pink = np.array([160, 100, 100])
upper_pink = np.array([179, 255, 255])

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for pink cloak
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Refine mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Invert mask
    mask_inv = cv2.bitwise_not(mask)

    # Extract cloak from background and non-cloak from current frame
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    normal_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Final result
    final = cv2.addWeighted(cloak_area, 1, normal_area, 1, 0)

    cv2.imshow("Invisibility Cloak - Pink Edition", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
