import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode = False, maxNoHands = 2, decCon = 0.5, trackCon = 0.5): #Here we are giving the parameters that are needed by the hands object
        self.mode = mode
        self.maxNoHands = maxNoHands
        self.decCon = decCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxNoHands, min_detection_confidence = self.decCon, min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #This line draws the mine on the palm
        return img
                

    def findPos(self, img, handNo = 0, draw= True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx,cy])
                    if draw and id==4:
                        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        return lmList
def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        sucess, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPos(img)
        if len(lmList) != 0:
            print(lmList[0], lmList[4], lmList[20])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()