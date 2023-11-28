#import os
#os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" #윈도우 테스틍용
import cv2
import mediapipe as mp
import pygame
from PIL import ImageFont, ImageDraw, Image
import numpy as np


pygame.mixer.init()
pygame.mixer.music.load('/Users/owon/midpipe_infor-main/CAMO-Six Weeks.wav') # wav mp3 ogg
test_sound = pygame.mixer.Sound('/Users/owon/midpipe_infor-main/CAMO-Six Weeks.wav') # wav ogg


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1) #0~1
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
            
        if not success:
            print("empty camera frame.")

            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
      
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # 엄지를 제외한 나머지 4개 손가락의 마디 위치 관계를 확인하여 플래그 변수를 설정합니다. 손가락을 일자로 편 상태인지 확인합니다.
                thumb_finger_state = 0
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


                #font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50) #라즈베리파이 용
                #font = ImageFont.truetype("/fonts/gulim.ttc", 30) #윈도우
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Andale Mono.ttf", 30) #맥 용
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
    
                text = ""
                #if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                    #text = "On"
                    #control = 0
    
                #elif thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    #text = "1"
                    #control = 1
   
                #elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    #text = "2"
                    #control = 2
 
                #elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                    #text = "3"
                    #control = 3

                #elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    #text = "Off"
                    #control = 4
                
                if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1: #!!
                    text = "player mode"
                    control = 3 # 보

                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0: #!!
                    text = "sound 30%"
                    control = 4 # 엄지 검지 중지
                
                elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0: #!!
                    text = "sound pause"
                    control = 5 # 주먹

                elif thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1: #!!
                    text = "sound unpause"
                    control = 6 # 넷 
                
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

                cv2.imshow('test Hands', image)

                #while True:                
                if control == 3:
                    pygame.mixer.music.play(1)
                        #pygame.mixer.music.set_volume(0.7)
                        
                
                #elif control == 4:
                        #pygame.mixer.music.set_volume(0.3)
                #       test_sound.stop()
                 
                elif control == 5:   
                    pygame.mixer.music.pause()
                    

                elif control == 4:
                    #for dc in range(0):
                        #pi_pwm.ChangeDutyCycle(0)   
                    pygame.mixer.music.unpause()

            
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()   