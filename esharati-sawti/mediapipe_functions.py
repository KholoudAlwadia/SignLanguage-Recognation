import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def add_image(image,results, action):

    #height,width = image.shape
    #print(image.shape)
    width = image.shape[1]#480
    height= image.shape[0]#640


    def overlay_transparent(background, overlay, x, y):
        # height and width of background image
        background_width = background.shape[1]
        background_height = background.shape[0]

        # if coordinate x and y is larger than background width and height, stop code
        if x >= background_width or y >= background_height:
            return background

        # height and width of overlay image
        h, w = overlay.shape[0], overlay.shape[1]

        # print('x:',x)
        # print('overlay_width:',w)
        # print('background_width:',background_width)

        # print('y:',y)
        # print('overlay_height:',h)
        # print('background_height:',background_width)

        if w >= background_width:
            return background
        if h >= background_height:
            return background

        # if coordinate x + width of overlay is larger than background width and height, stop code
        if x + w > background_width:
            # w = background_width - x
            # overlay = overlay[:, :w]
            return background
        if x - w < 2:
            # w = background_width - x
            # overlay = overlay[:, :w]
            return background
        if y + h > background_height:
            # h = background_height - y
            # overlay = overlay[:h]
            return background

        if y - h < 2:
            # h = background_height - y
            # overlay = overlay[:h]
            return background

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
                ],
                axis=2,
            )
        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

        return background

    index = 10
    pose = []

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face_keypoint = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

# print(len(face_keypoint))
# print(action)
    if face_keypoint.size != 0 and np.any(face_keypoint[index]) == True:
        if action == 'thanks':
            print("شكرا")
        elif action == 'IM - sorry':
            print('انا اسف ')

        elif action == 'sabah-alnoor':
            print('صباح النور')

        elif action == 'ok':
            file_name = print('حسنا')
        elif action == 'yes':
            file_name = print('نعم')
        elif action == 'no':
            file_name = print('لا')
        elif action == 'alaslam-alaykum':
            file_name = print('السلام عليكم')
        elif action == 'alaykum-alaslam':
            file_name = print('عليكم السلام ')
        elif action == 'goodbye':
            file_name = print('الى القاء')
        else:
            file_name = print('حاول مرة اخرى ')



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()

    multiple = 47
    for num, prob in enumerate(res):
        if np.argmax(res) == num and res[np.argmax(res)] >= threshold:
            (text_width, text_height), baseline = cv2.getTextSize(
                actions[num] + ' ' + str(round(prob * 100, 2)) + '% ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.rectangle(output_frame, (0, 60 + num * multiple), (int(prob * text_width), 95 + num * multiple),
                          colors[num], -1)  # change length of bar depending on probability

            cv2.putText(output_frame, actions[num] + ' ' + str(round(prob * 100, 2)) + '%',
                        (5, 90 + num * multiple), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        else:
            (text_width, text_height), baseline = cv2.getTextSize(
                actions[num] + ' ' + str(round(prob * 100, 2)) + '% ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.rectangle(output_frame, (0, 60 + num * multiple), (int(prob * text_width), 95 + num * multiple),
                          colors[num], -1)  # change length of bar depending on probability

            cv2.putText(output_frame, actions[num] + ' ' + str(round(prob * 100, 2)) + '%',
                        (5, 90 + num * multiple), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


'''def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
'''


