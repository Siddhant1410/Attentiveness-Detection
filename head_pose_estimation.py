import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import csv
import sys

flag = 0
conf = 0.00
fields = ["STUDENT1", "STUDENT2", "STUDENT3", "STUDENT4", "STUDENT5"]
list = ["", "", "", "", ""]

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):

    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)
    
face_model = get_face_detector()
landmark_model = get_landmark_model()
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
file_ = sys.argv[1]
filearg = str(file_)
img = cv2.imread(filearg)
print(filearg)
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
#while True:
#ret, img = cap.read()

img = cv2.imread(filearg)
# if ret == True:
faces = find_faces(img, face_model)
for face in faces:
    marks = detect_marks(img, landmark_model, face)
    # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
    image_points = np.array([
                            marks[30],     # Nose tip
                            marks[8],     # Chin
                            marks[36],     # Left eye left corner
                            marks[45],     # Right eye right corne
                            marks[48],     # Left Mouth corner
                            marks[54]      # Right mouth corner
                        ], dtype="double")
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    cv2.line(img, p1, p2, (0, 255, 255), 2)
    cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
    # for (x, y) in marks:
    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
    try:
        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        ang1 = int(math.degrees(math.atan(m)))
    except:
        ang1 = 90

    try:
        m = (x2[1] - x1[1])/(x2[0] - x1[0])
        ang2 = int(math.degrees(math.atan(-1/m)))
    except:
        ang2 = 90
        # print('div by zero error')
    if ang1 >= 30:
        conf = (ang1 / 70) * 100
        confstr = str(conf)
        #cv2.putText(img, 'Head down, Confidence: ', (30, 30), font, 2, (255, 255, 128), 3)
        if conf > 30:
            cv2.putText(img, 'DOWN '+ confstr, (30, 40), font, 1, (255, 255, 128), 3)
            if flag == 0:
                list[0] = 'Down'
            elif flag == 1:
                list[1] = 'Down'
            elif flag == 2:
                list[2] = 'Down'
            elif flag == 3:
                list[3] = 'Down'
            elif flag == 4:
                list[4] = 'Down'
            flag += 1
        conf = 0.00

    elif ang1 <= -40:
        ang11 = abs(ang1)
        conf = ( ang11 / 70) * 100
        confstr = str(conf)
        #cv2.putText(img, 'Head up, Confidence: ', (30, 30), font, 2, (255, 255, 128), 3)
        if conf > 30:
            cv2.putText(img, 'UP ' + confstr, (320, 30), font, 1, (255, 255, 128), 3)
            if flag == 0:
                list[0] = 'Up'
            if flag == 1:
                list[1] = 'Up'
            if flag == 2:
                list[2] = 'Up'
            if flag == 3:
                list[3] = 'Up'
            if flag == 4:
                list[4] = 'Up'
            flag += 1
        conf = 0.00
    if ang2 >= 30:
        conf = (ang2 / 60) * 100
        confstr = str(conf)
        #cv2.putText(img, 'Head right, Confidence: ', (90, 30), font, 2, (255, 255, 128), 3)
        if conf > 30:
            cv2.putText(img, 'RIGHT ' + confstr, (320, 200), font, 1, (255, 255, 128), 3)
            if flag == 0:
                list[0] = 'Right'
            if flag == 1:
                list[1] = 'Right'
            if flag == 2:
                list[2] = 'Right'
            if flag == 3:
                list[3] = 'Right'
            if flag == 4:
                list[4] = 'Right'
            flag += 1
        conf = 0.00

    elif ang2 <= -38:
        ang22 = abs(ang2)
        conf = (ang22 / 65) * 100
        confstr = str(conf)
        #cv2.putText(img, 'Head left, Confidence', (90, 30), font, 2, (255, 255, 128), 3)
        if conf > 30:
            cv2.putText(img, 'LEFT ' + confstr, (80, 30), font, 1, (255, 255, 128), 3)
            if flag == 0:
                list[0] = 'Left'
            if flag == 1:
                list[1] = 'Left'
            if flag == 2:
                list[2] = 'Left'
            if flag == 3:
                list[3] = 'Left'
            if flag == 4:
                list[4] = 'Left'
            flag += 1
        conf = 0.00


    else:
        cv2.putText(img, 'FRONT ', (100, 80), font, 1, (255, 255, 128), 3)
        if flag == 0:
            list[0] = 'Front'
        if flag == 1:
            list[1] = 'Front'
        if flag == 2:
            list[2] = 'Front'
        if flag == 3:
            list[3] = 'Front'
        if flag == 4:
            list[4] = 'Front'
        flag += 1


            # if conf > 50:
            #     cv2.putText(img, 'FRONT ' + confstr, (100, 80), font, 1, (255, 255, 128), 3)
        # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
        # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
cv2.imshow('img', img)
cv2.waitKey(3000)
cv2.destroyAllWindows()

if list[0] == '':
    list[0] = 'N.A'
if list[1] == '':
    list[1] = 'N.A'
if list[2] == '':
    list[2] = 'N.A'
if list[3] == '':
    list[3] = 'N.A'
if list[4] == '':
    list[4] = 'N.A'

with open('data.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerow(list)

with open('graph_data/graph_data_1.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow([list[0]])
with open('graph_data/graph_data_2.csv', 'a', newline='', encoding='UTF8') as f:
 writer = csv.writer(f)
 writer.writerow([list[1]])
with open('graph_data/graph_data_3.csv', 'a', newline='', encoding='UTF8') as f:
 writer = csv.writer(f)
 writer.writerow([list[2]])
with open('graph_data/graph_data_4.csv', 'a', newline='', encoding='UTF8') as f:
 writer = csv.writer(f)
 writer.writerow([list[3]])
with open('graph_data/graph_data_5.csv', 'a', newline='', encoding='UTF8') as f:
 writer = csv.writer(f)
 writer.writerow([list[4]])


f.close()



