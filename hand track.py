import cv2
import time
import mediapipe as mp 
#hand track, track hand
cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
draw = mp.solutions.drawing_utils
oldtime = 0
currenttime = 0
while True: 
    success, img = cam.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands.process(imgrgb)
    print(output.multi_hand_landmarks) #print the landmarks i.e the circles on the final image
    if output.multi_hand_landmarks:
        for handLms in output.multi_hand_landmarks: #for the hand in camera get the id and draw the landmarks
            for id, lm in enumerate(handLms.landmark):#for the id and landmarks in hand
                print(id,lm)
                h, w, c = img.shape #height and width of camera gui 
                centerx, centery = int(lm.x*w),int(lm.y*h) #landmark pixel, multiple the landmarksx value by the width, if the screen was 1 pixel the x value would be the x value(ik this sounds stupid), but we need to multiple by the width to get it up to scale same thing for height
                print(id,centerx,centery)
                if id == 0: #change id to get the point you want
                    cv2.circle(img,(centerx,centery),25, (0,255,255),cv2.FILLED ) #yellow circle on landmark

            draw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)
    currenttime = time.time()#get time of refresh in secound
    fps = 1/(currenttime - oldtime)
    oldtime = currenttime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) #display the fps
    cv2.imshow('img',img)#show img
    if cv2.waitKey(1) & 0xFF == ord('z'): # if the z key is clicked we break the loop and run the 2 lines cam.relase() and cv2.destroyallwindows()
        break
cam.release()
cv2.destroyAllWindows()
