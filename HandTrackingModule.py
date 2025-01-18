import cv2                                             #video capture and image processing
import mediapipe as mp                                 
import time                                            #used to calculate frames per second (FPS).


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode                       #Specifies whether the hand detection model runs in static or dynamic mode
        self.maxHands = maxHands               #Sets the maximum number of hands that the model will detect in a single frame.
        self.detectionCon = detectionCon       #Specifies the minimum confidence level required for the model to consider a hand detection as valid
        self.trackCon = trackCon               #Sets the minimum confidence level required for the model to track a detected hand across frames.

        self.mpHands = mp.solutions.hands              
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):                     #Converts the input image to RGB for processing.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #converts the image from BGR to RGB
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, handNo=0, draw=True):             #Extracts the positions of landmarks for the first detected hand 
        lmList = []   #stores a list of landmarks
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList                       #Returns a list of [id, x, y] for each landmark.


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)         #video capture

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    detector = handDetector()            #Initializes the handDetector object

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to read frame from webcam.")
            break

        img = detector.findHands(img)
        lmList = detector.findPos(img)

        if lmList:
            print("Landmark 0 Position:", lmList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)    #displays the FPS on the output video
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()
#Continuously reads frames, processes them to detect hands, and prints the position of the first landmark if available.

if __name__ == "__main__":          
    main()
