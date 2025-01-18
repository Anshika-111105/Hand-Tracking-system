import time                        #used to calculate frames per second (FPS).
import cv2
import HandTrackingModule as htm

# Initialize variables
pTime = 0
cap = cv2.VideoCapture(0)  # Open webcam
detector = htm.handDetector()  #hand detection object

while True:
    success, img = cap.read()       #Reads video frames
    if not success:                        
        print("Failed to grab frame.")
        break

    # Detect hands and get positions
    img = detector.findHands(img)
    lmList = detector.findPos(img)
    if len(lmList) != 0:
        print(lmList[0])  # Print the first landmark position
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # Show the image
    cv2.imshow("Image", img)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
