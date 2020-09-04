import numpy as np
import cv2 as cv
import tkinter as tk
import PIL.Image, PIL.ImageTk
import time
import os.path
import pytesseract
import argparse
import sys

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("HLX-ALPR v1.0")
window.maxsize(700,600)
window.minsize(700,600)
window.config(background="#34495e")

#Graphics window
imageFrame = tk.Frame(window, width=640, height=530, bg="#34495e")
imageFrame.grid(row=0, column=0, padx=10, pady=2)

tk.Label(imageFrame, text="Welcome to HLX Building", font = ("Arial", 16, "bold"), bg="#34495e",fg='white').grid(row=0, column=0, padx=5, pady=5)
#tk.Label(imageFrame, text='', bg="#34495e").grid(row=1, column=0, padx=5, pady=5)

display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, padx=18, pady=2)  #Display 1

btn_enter = tk.Button(imageFrame, text='Enter', width = 20, height = 3)
btn_enter.grid(row=1, column=0,sticky='nsew')

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')

class abc():
    def __init__(self, name):
        self.name = name

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "darknet-yolov3.cfg"
modelWeights = "lapi.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#Capture video frames
cap = cv.VideoCapture(0)

def show_frame():

    def printero(txt):
        root = tk.Toplevel()
        root.title('NOTICE')
        root.minsize(300,200)
        infoFrame = tk.Frame(root, width=400, height=250, )#bg="#34495e")
        infoFrame.grid(row=0, column=0, padx=10, pady=2)

        #print info

        tk.Label(infoFrame, text='Detail', font=('Arial', 16, 'bold')).grid(row=1, column=1,padx=5, pady=5)
        tk.Label(infoFrame, text='Vehicle No.\t: ').grid(row=2, column=1,padx=5, pady=5)
        tk.Label(infoFrame, text='Time Enter\t: ').grid(row=3, column=1,padx=5, pady=5)
        tk.Label(infoFrame, text=txt).grid(row=2, column=2,padx=5, pady=5)
        tim = time.strftime("%d-%m-%Y  %H:%M:%S")
        tk.Label(infoFrame, text = tim, fg='black').grid(row=3,column=2,padx=60, pady=5)

        tk.Label(infoFrame, text='Have a good day!', font=('Arial', 16, 'bold')).grid(row=4, column=1,padx=5, pady=5)
    
        root.after(5000, root.destroy)

    # Get the names of the output layers
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        crop = cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 3)

        cv.imshow("Detection",crop)
        #cv.imshow("Car plate",crop[top:bottom,left:right])
        cv.imwrite('test100.jpg', crop[top-1:bottom+1,left+1:right-1])
    
        image = cv.imread('test100.jpg')
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 12')

        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        new_text = ''

        for element in text:
            if element not in chars:
                element = ''
    
            new_text = new_text + element

        print('Car plate: ' + new_text)

        printero(new_text)


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            #print("out.shape : ", out.shape)
            for detection in out:
                #if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                #if scores[classId]>confThreshold:
                confidence = scores[classId]
                if detection[4]>confThreshold:
                    print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                    print(detection)
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    
    hasFrame, frame = cap.read()

    if hasFrame == True:
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        cv2image = cv.resize(cv2image, (640,480), interpolation=cv.INTER_NEAREST)
        img = PIL.Image.fromarray(cv2image)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        display1.imgtk = imgtk #Shows frame for display 1
        display1.configure(image=imgtk)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
    
    window.after(1, show_frame)

display1 = tk.Label(imageFrame)
display1.grid(row=2, column=0, padx=18, pady=2)  #Display 1

show_frame()
window.mainloop()  #Starts GUI