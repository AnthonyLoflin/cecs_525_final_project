import cv2
import os
import sys
from mail import sendEmail
from flask import Flask, render_template, Response, request, send_from_directory, jsonify, redirect, url_for
from camera import VideoCamera
import time
import threading
import numpy as np
import traceback

email_update_interval = 10 # sends an email only once in this time interval
video_camera = VideoCamera(flip=False) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["","Anthony Loflin", "Manorath Dhakal", "Ryan", "Alec"]
directory_counter = 5

# App Globals (do not edit)
app = Flask(__name__, static_url_path='')
last_epoch = 0

def check_for_objects():
	global last_epoch
	
	
	while True:
		try:
                        
                    upload_done_event.wait()
                    frame, found_obj = video_camera.get_object(object_classifier)
                    if found_obj and (time.time() - last_epoch) > email_update_interval:
                            last_epoch = time.time()
                            cv2.imwrite("output.jpg", frame)
                            test_img = cv2.imread("output.jpg")
                            predicted_img = predict(test_img)
                            ret, jpeg = cv2.imencode('.jpg', predicted_img)
                            
                            print "Sending email..."
                            sendEmail(jpeg.tobytes())
                            print "done!"

		except:
			print "Error sending email: ", sys.exc_info()[0]
			traceback.print_exc(file=sys.stdout)

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/app')
def send_html():
    return send_from_directory(os.path.join('.', 'static', 'html'), 'image_upload_test.html')

@app.route('/upload', methods=["POST"])
def get_data():
    print "[ enter upload ]"
    upload_done_event.clear()
    global directory_counter
    global subjects
    global face_recognizer
    if request.method == "POST":
        if request.files and request.form['subject_name']:
            subjects.append(str(request.form['subject_name']))
            print subjects
            dirpath = os.path.relpath('training-data/s'+ str(directory_counter))
            os.makedirs(dirpath)
            directory_counter += 1
            for i, file_name in enumerate(request.files):
                if file_name == '':
                    log.error('[upload] Upload attempt with no filename')
                    return Response('No filename uploaded', status=500)
                file = request.files[file_name]
                file.save(os.path.join(dirpath, file.filename))
            
            #create our LBPH face recognizer 
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("Preparing data...")
            faces, labels = prepare_training_data("training-data")
            print("Data prepared")

            #print total faces and labels
            print("Total faces: ", len(faces))
            print("Total labels: ", len(labels))
            
            
            
            #train our face recognizer of our training faces
            face_recognizer.train(faces, np.array(labels))
            
            upload_done_event.set()
            return Response('uploaded successfully', status=200)
        else:
            log.error('[upload] Request.files is empty')
            upload_done_event.set()
            return Response('No files uploaded', status=500)
            #return request.data

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
            print image_path

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img



if __name__ == '__main__':
    
    #create our LBPH face recognizer 
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    
    
    
    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
     
    t = threading.Thread(target=check_for_objects, args=())
    upload_done_event = threading.Event()
    upload_done_event.set()
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=False)
