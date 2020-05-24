# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:59:18 2019

@author: IevaBrantevica & YoupSuurmeijer
"""
import cv2
from imageai.Detection import ObjectDetection
import os
from operator import itemgetter

class BodyExtractor:
    """
    Create Body Extractor Object
    models: dict containing the locations of the CNN models to be used in the body extractor
    'MIN_PROB': Minimum detection probability to detect a body (range 1-100)
    'IMG_SIZE': Image size for the CNN input (default 128)
    'UPPER_TO_LOWER_RATIO': Ratio between lower and upper body height, used in distinguising upper from lower body
    'FACE_SCALE_FACTOR': Scaling factor for face detection
    'FACE_MIN_NEIGHBOURS': Detection sensitivity parameter for the face detection
    'FACE_MIN_SIZE': Minimum size of face in pixels (wxh)
    'EYES_SCALE_FACTOR': Detection sensitivity parameter for the eye detection
    'EYES_MIN_NEIGHBOURS': Detection sensitivity parameter for the eye detection
    """

    def __init__(self, settings, models_path):

        self.settings = settings

        self.face_detector = cv2.CascadeClassifier(os.path.join(models_path, self.settings.face.model))
        self.eye_detector = cv2.CascadeClassifier(os.path.join(models_path, self.settings.eyes.model))

        self.body_detector = ObjectDetection()
        self.body_detector.setModelTypeAsYOLOv3()
        self.body_detector.setModelPath(os.path.join(models_path, self.settings.body.model))
        self.body_detector.loadModel(detection_speed="flash")
        self.custom_objects = self.body_detector.CustomObjects(person=True)

    def select_face_from_faces(self, faces, body):
        # Initialize list of scores, in the end the body with the highest score gets returned
        scores = []

        for face in faces:
            score = 0

            # We assume the face we want to classify is in the middle of the body, so a score of 0-10 is given for
            # how near a face is to the center of the body (0: completely on the edge, 10: dead center)
            body_center = body[2] / 2
            face_center = face[0] + face[2] / 2
            center_bonus_score = 20 - 20 * abs(body_center - face_center) / body_center
            score += center_bonus_score

            # We also want a large face area as usually misdetections are small patches
            size_bonus_score = 10 - 10 * abs(body[2] - face[2]) / body[2]
            score += size_bonus_score

            scores.append((face, score))

        face_with_max_score = max(scores, key=itemgetter(1))[0]
        return face_with_max_score

    def detect_faces(self, image):
        """
        Load all the separate image files and extract coordinates of faces that have eyes, save it in the list

        Parameters
        ----------
        image : output of the model predict function, an image with colors for each separate class

        Returns
        -------
        list of detected faces with the face location and eyes locations (x, y, w, h)
        """
        # Initialize output variable
        output = []

        # convert the input image to grayscale (opencv face detector expects gray images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying the haar classifier to detect faces
        temp_faces = self.face_detector.detectMultiScale(gray_image,
                                                         scaleFactor=self.settings.face.min_scale_factor,
                                                         minNeighbors=self.settings.face.min_neighboors,
                                                         minSize=tuple(self.settings.face.min_size)
                                                         )
        # Extract the detected face area
        for f in temp_faces:
            face_image = image[f[1]:f[1] + f[3], f[0]:f[0] + f[2]]

            # Applying the haar classifier to detect eyes within face area
            eyes = self.eye_detector.detectMultiScale(face_image,
                                                      scaleFactor=self.settings.eyes.min_scale_factor,
                                                      minNeighbors=self.settings.eyes.min_neighboors
                                                      )

            # Extract only faces with at least 1 eye and save it in the defined path
            output.append({'face': f.tolist(), 'eyes': len(eyes), 'eyes_list': eyes})

        return output

    def detect_body(self, image_url):
        """
        Load all the separate image files and extract coordinates of faces that have eyes, save it in the list

        Parameters
        ----------
        image_url (string): Location of image on local drive
        min_prob (int): Minimum detection probability for detecting bodies in the input image

        Returns
        -------
        list of detected bodies with the body location (x, y, w, h)
        """
        # Initialize output list
        output = []

        # Applying the haar classifier to detect entire body
        detections = self.body_detector.detectCustomObjectsFromImage(
            custom_objects=self.custom_objects,
            input_image=image_url,
            output_image_path="temp.jpg",
            minimum_percentage_probability=self.settings.body.min_prob)

        # For each detection append an entry to the output with x, y, w, h location of body
        for eachObject in detections:
            temp = eachObject["box_points"]

            body_whole = [temp[0], temp[1],
                          temp[2] - temp[0],
                          temp[3] - temp[1]]

            output.append(body_whole)

        return output

    def extract_body_parts(self, image, image_url):
        """
        Load all the separate image files and extract coordinates of faces that have eyes, save it in the list

        Parameters
        ----------
        image (open-cv image): input image to detect the bodies from
        image_url (string): Location of image on local drive

        Returns
        -------
        list of detected bodies with the body location (x, y, w, h)
        """
        # Initialize output list
        output = []

        # Detect bodies from image
        bodies = self.detect_body(image_url)

        # For each body detect the faces and the eyes and append to output
        for i in range(len(bodies)):
            # Segement the image to only the current body
            x, y, w, h = bodies[i]
            body_image = image[y:y + h, x:x + w]

            # Detect the number of faces within the current body
            temp_faces = self.detect_faces(body_image)

            # If there is one or more faces in the body take the first face that is detected (suboptimal!)
            # NOTE: write a face selector function instead of just picking the first face!
            if len(temp_faces) > 0:
                # Take first face from face detector and extract the face image
                face = self.select_face_from_faces([x['face'] for x in temp_faces], body=bodies[i])
                face = [x + face[0], y + face[1], face[2], face[3]]
                eyes = temp_faces[0]["eyes"]
                eyes_list = []

                # For each eye that is detected append it to the eyes list
                if eyes > 0:
                    for eye in temp_faces[0]["eyes_list"]:
                        eyes_list.append([face[0] + eye[0], face[1] + eye[1], eye[2], eye[3]])

                # Append to output list
                output.append({'image': image, 'body': bodies[i], 'face': face, 'eyes': eyes, 'eyes_list': eyes_list})
            else:
                # If no faces are detected just append the body output
                output.append({'image': image, 'body': bodies[i]})

        return output


if __name__ == '__main__':
    import glob
    import cv2
    from vipo.utils import *
    from line_profiler import LineProfiler
    from datetime import datetime

    SETTINGS = DotDict({
        "face": {
            "model": "./models/haarcascade_frontalface_default.xml",
            "scale_factor": 1.03,
            "min_neighboors": 20,
            "min_size": [128, 128]
        },
        "eyes": {
            "model": "./models/haarcascade_eye.xml",
            "scale_factor": 1.05,
            "min_neighboors": 6
        },
        "body": {
            "model": "./models/yolo.h5",
            "min_prob": 50
        }
    })

    PLOT_COLORS = {
        'body': [(0, 0, 255), 8, 0.5],
        'face': [(255, 0, 0), 5, 0.1],
        'eyes': [(0, 255, 255), 5, 1],
        'upper_body': [(0, 255, 0), 8, 1],
        'lower_body': [(0, 255, 0), 8, 1],
        'default': [(0, 0, 0), 0, 0]
    }

    # set_learning_phase(0)
    body_extractor = BodyExtractor(settings=SETTINGS)

    # Set images directory
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    search_dir = os.path.join(root_dir, r'images\test_set_21_01\TEST1VIPO')
    os.chdir(search_dir)

    now = datetime.now()
    output_path = os.path.join(search_dir, 'bodyextractor_' + now.strftime("%Y%m%d%H%M"))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize the line profiler!
    lp = LineProfiler()
    lp_wrapper = lp(body_extractor.extract_body_parts)

    errors = 0
    for file in glob.glob("*.jpg")[:10]:
        image = cv2.imread(file)
        try:
            bodyparts = lp_wrapper(image, file)
            selected_bodyparts = select_body_from_body_parts(bodyparts)
            body_image = selected_bodyparts['image']

            text = {'gender': '', 'age_label': '',
                    'upper': '', 'lower': ''}
            plot_image_with_body_parts(image, body_image, bodyparts,
                                       selected_bodyparts['body'], PLOT_COLORS,
                                       text, save=os.path.join(output_path, 'body_' + file))
        except:
            print('ERROR')
            errors += 1
            pass

    lp.print_stats(output_unit=1e-03)
    print('Number of errors:', errors)
