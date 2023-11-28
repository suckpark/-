import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pygame
import time
import RPi.GPIO as GPIO

pygame.mixer.init() # mp3 or wav or ogg
pygame.mixer.music.load('/home/pi/mocar/CAMO-Six Weeks.mp3') # wav mp3 ogg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


FAN = 18   #연결된 번호 GPIO

GPIO.setmode(GPIO.BCM)   #GPIO 설정
GPIO.setwarnings(False)   #경고 메시지 나타내지 않음
GPIO.setup(FAN, GPIO.OUT)   #FAN 연결된 GPIO 출력 핀으로 설정

#채널 : FAN연결된 핀으로, 주파수 = 1000Hz인 pwm 발생
control = 100
pi_pwm = GPIO.PWM(FAN, 10000)
pi_pwm.start(0)   #duty 사이클 0%에서 pwm 시작(풍속 0), 최대 풍속 duty 100%

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
            
        if not success:
            print("empty camera frame.")
            break

        # Flip the frame horizontally for a later selfie-view display, and convert
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                thumb_finger_state = 0 # 손가락 설정
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_MCP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                            thumb_finger_state = 1

                index_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                            index_finger_state = 1

                middle_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                            middle_finger_state = 1

                ring_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                            ring_finger_state = 1

                pinky_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
                            pinky_finger_state = 1
                
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50) #라즈베리파이 용
                #font = ImageFont.truetype("/fonts/gulim.ttc", 30) #윈도우
                #font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Andale Mono.ttf", 30) #맥 용
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
    
                text = ""    
    
                if thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "on / 1"
                    control = 0 # on = fan 1
   
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "1"
                    control = 1
 
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "2"
                    control = 2

                elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "Off"
                    control = 3
                
                # midea control
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                    text = "player mode"
                    control = 4

                elif thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                    text = "suond unpause"
                    control = 5
 
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1: #!!
                    text = "sound pause"
                    control = 6     

                w, h = font.getsize(text)

                x = 50
                y = 50

                draw.rectangle((x, y, x + w, y + h), fill='black')
                draw.text((x, y), text, font=font, fill=(255, 255, 255))
                image = np.array(image)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                 #fan            
                if control == 0:   
                        pi_pwm.ChangeDutyCycle(30)               
                elif control == 1:
                        pi_pwm.ChangeDutyCycle(50)                          
                elif control == 2:                   
                        pi_pwm.ChangeDutyCycle(100)                           
                elif control == 3:
                        pi_pwm.ChangeDutyCycle(0)

                #media
                elif control == 4: 
                        pygame.mixer.music.play(1)        
                        #pygame.mixer.music.set_volume(0.7)
                elif control == 5:                  
                        pygame.mixer.music.unpause()     
                elif control == 6:
                        pygame.mixer.music.pause()

                                         
        cv2.imshow('fan+mirror+media', image)                        
                        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()   