import numpy as np
import cv2
import time
import os
import sys
import matplotlib.pyplot as plt
import csv
import pandas as pd

results1 = []
results2 = []
results3 = []
results4 = []
results5 = []

with open('graph_data/graph_data_1.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Head_POS"])
with open('graph_data/graph_data_2.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Head_POS"])
with open('graph_data/graph_data_3.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Head_POS"])
with open('graph_data/graph_data_4.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Head_POS"])
with open('graph_data/graph_data_5.csv', 'a', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Head_POS"])
x=1
itr = 0


while True:
    try:
        cap = cv2.VideoCapture("video_1.mp4")
        total_frames = cap.get(13)
        filename = "frames/image_rend" + str(int(x)) + ".jpg"
        x=x+1
        itr = itr+150
        cap.set(1, itr)
        ret, frame = cap.read()
        cv2.imwrite(filename, frame)
        print("Frame Written!")
        if x == 13:
            print("All frames written Successfully!")
            break
    except:
        print("Failed to write all fames. Video length is short.")
        exit()

os.system('python head_pose_estimation.py frames/image_rend1.jpg')
os.system('python head_pose_estimation.py frames/image_rend2.jpg')
os.system('python head_pose_estimation.py frames/image_rend3.jpg')
os.system('python head_pose_estimation.py frames/image_rend4.jpg')
os.system('python head_pose_estimation.py frames/image_rend5.jpg')
os.system('python head_pose_estimation.py frames/image_rend6.jpg')
os.system('python head_pose_estimation.py frames/image_rend7.jpg')
os.system('python head_pose_estimation.py frames/image_rend8.jpg')
os.system('python head_pose_estimation.py frames/image_rend9.jpg')
os.system('python head_pose_estimation.py frames/image_rend10.jpg')
os.system('python head_pose_estimation.py frames/image_rend11.jpg')
os.system('python head_pose_estimation.py frames/image_rend12.jpg')


# os.remove("frames/image_rend1.jpg")
# os.remove("frames/image_rend2.jpg")
# os.remove("frames/image_rend3.jpg")
# os.remove("frames/image_rend4.jpg")
# os.remove("frames/image_rend5.jpg")

# Python program to read CSV file line by line

from csv import reader
with open('graph_data/graph_data_1.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # print(row)
            results1 += (row)
            
with open('graph_data/graph_data_2.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # print(row)
            results2 += (row)
            
with open('graph_data/graph_data_3.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # print(row)
            results3 += (row)
            
with open('graph_data/graph_data_4.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # print(row)
            results4 += (row)
            
with open('graph_data/graph_data_5.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # print(row)
            results5 += (row)



atnt1 = results1.count("Front")
flag1 = results1.count("N.A")
atnt_score1 = (atnt1 / len(results1)) * 100
atnt_score_str1 = str(abs(atnt_score1))

atnt2 = results2.count("Front")
flag2 = results2.count("N.A")
atnt_score2 = (atnt2 / len(results2)) * 100
atnt_score_str2 = str(abs(atnt_score2))

atnt3 = results3.count("Front")
flag3 = results3.count("N.A")
atnt_score3 = (atnt3 / len(results3)) * 100
atnt_score_str3 = str(abs(atnt_score3))

atnt4 = results4.count("Front")
flag4 = results4.count("N.A")
atnt_score4 = (atnt4 / len(results4)) * 100
atnt_score_str4 = str(abs(atnt_score4))

atnt5 = results5.count("Front")
flag5 = results5.count("N.A")
atnt_score5 = (atnt5 / len(results5)) * 100
atnt_score_str5 = str(abs(atnt_score5))




x = ["Sec_0", "Sec_15", "Sec_30", "Sec_45", "Sec_60"]
columns = ["Head_POS"]
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        
if flag1 == 5:
    print("No Valid Faces Were Found for First Student")
else:
    df = pd.read_csv('graph_data/graph_data_1.csv', usecols=columns)
    y = df.Head_POS
    plt.plot(x, y, 'k')
    plt.title('Overall Attention: ' + atnt_score_str1 +'%', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('head_position', fontdict=font)
    plt.savefig('graphs/Student_1.png')
    plt.show()

if flag2 == 5:
    print("No Valid Faces Were Found for Second Student")
else:
    df = pd.read_csv('graph_data/graph_data_2.csv', usecols=columns)
    y = df.Head_POS
    plt.plot(x, y, 'k')
    plt.title('Overall Attention: ' + atnt_score_str2 +'%', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('head_position', fontdict=font)
    plt.savefig('graphs/Student_2.png')
    plt.show()

if flag3 == 5:
    print("No Valid Faces Were Found Third student")
else:
    df = pd.read_csv('graph_data/graph_data_3.csv', usecols=columns)
    y = df.Head_POS
    plt.plot(x, y, 'k')
    plt.title('Overall Attention: ' + atnt_score_str3 +'%', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('head_position', fontdict=font)
    plt.savefig('graphs/Student_3.png')
    plt.show()

if flag4 == 5:
    print("No Valid Faces Were Found for Fourth Student")
else:
    df = pd.read_csv('graph_data/graph_data_4.csv', usecols=columns)
    y = df.Head_POS
    plt.plot(x, y, 'k')
    plt.title('Overall Attention: ' + atnt_score_str4 +'%', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('head_position', fontdict=font)
    plt.savefig('graphs/Student_4.png')
    plt.show()

if flag5 == 5:
    print("No Valid Faces Were Found for Fifth Student")
else:
    df = pd.read_csv('graph_data/graph_data_5.csv', usecols=columns)
    y = df.Head_POS
    plt.plot(x, y, 'k')
    plt.title('Overall Attention: ' + atnt_score_str5 +'%', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('head_position', fontdict=font)
    plt.savefig('graphs/Student_5.png')
    plt.show()

os.remove("graph_data/graph_data_1.csv")
os.remove("graph_data/graph_data_2.csv")
os.remove("graph_data/graph_data_3.csv")
os.remove("graph_data/graph_data_4.csv")
os.remove("graph_data/graph_data_5.csv")
