import cv2
import dlib
import glob
import numpy as np
import os
import openface
import pandas as pd
import pickle
import argparse
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

imgDim = 96

dlib_shape_predictor_path = '../models/dlib/shape_predictor_68_face_landmarks.dat'

# aligning face using openface which in turn uses dlib
fa_openface = openface.AlignDlib(dlib_shape_predictor_path)

# Torch Neural Net to get the face embeddings
face_encoder = openface.TorchNeuralNet('../models/openface/nn4.small2.v1.t7', imgDim)

# svm classifier model file
svm_classifier_filename = "./classifier.pkl"


def training(training_data):
    """
    Training our classifier (Linear SVC). Saving model using pickle.
    We need to have only one person/face per picture.
    :param people: people to classify and recognize
    """

    # get the embeddings and labels to process further for SVM model training
    df = get_embeddings_label_dataframe(training_data)

    # converting labels into int
    le = LabelEncoder()
    y = le.fit_transform(df[128])
    # print(y)
    print("Training for {} classes.".format(len(le.classes_)))
    X = df.drop(128, axis=1)
    print("Training with {} pictures.".format(len(X)))

    # training
    clf = SVC(C=2, kernel='linear', probability=True)
    clf.fit(X, y)

    # dumping model
    print("Saving classifier to '{}'".format(svm_classifier_filename))
    with open(svm_classifier_filename, 'wb') as f:
        pickle.dump((le, clf), f)


def get_embeddings_label_dataframe(known_faces_folder):
    """
    :param known_faces_folder:
    :return:
    """

    # generating labels and encodings
    people = os.listdir(known_faces_folder)
    print(people)
    df = pd.DataFrame()
    for p in people:
        image_data = []
        print(p)
        # for each face in the current class folder, find the encodings and stack them together in 'l'
        # 'l' is a ncx129 dimensional matrix where the 129th column is the class id
        for filename in glob.glob(known_faces_folder + '/' + p + '/*.jpg'):
            image = cv2.imread(filename)
            detected_faces = detect_face_dlib(image)
            aligned_faces = alignFace_openface(image, detected_faces)
            print('{} faces detected in {}'.format(len(aligned_faces), filename))
            if len(aligned_faces) > 0:
                face_encoding = get_face_encodings_openface(aligned_faces[0])
                image_data.append(np.append(face_encoding, [p]))
        # converting the 'l' into dataframe and appending it to the main dataframe that holds
        # the encodings for all classes and all images in each class
        df = pd.concat([df, pd.DataFrame(np.array(image_data))])
    df.reset_index(drop=True, inplace=True)
    return df


def get_face_encodings_openface(alignedFace):
    temp = cv2.resize(alignedFace, (imgDim, imgDim))
    embeddings = face_encoder.forward(temp)
    return np.array(embeddings)


def alignFace_openface(image, detected_faces):
    alignedFaces = []
    for (x1, y1, x2, y2) in detected_faces:
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        alignedFaces.append(
            fa_openface.align(
                imgDim,
                image,
                dlib_rect,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
    return alignedFaces


def detect_face_dlib(image):
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 0)

    # if faces were not found, try finding faces in upsampled the image
    if detected_faces is None:
        # Run the HOG face detector on the image data.
        # The result will be the bounding boxes of the faces in our image.
        detected_faces = face_detector(image, 1)

    return [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]


def predict(image, le, clf):
    detected_faces = detect_face_dlib(image)
    aligned_faces = alignFace_openface(image, detected_faces)

    prediction = []
    img = np.copy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, aligned_face in enumerate(aligned_faces):

        # predict each face
        p = clf.predict_proba(get_face_encodings_openface(aligned_face).reshape(1, 128))
        maxValue = max(p[0])
        maxIndex = np.argmax(p)
        y_pred = le.classes_[maxIndex]
        (x1, y1, x2, y2) = detected_faces[idx]
        prediction.append([y_pred, maxValue, (x1, y1), (x2, y2)])

        print('predicted {} with {} prob'.format(y_pred, maxValue))

        # draw bounding boxes
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if maxValue > 0.5:
            cv2.putText(img, y_pred, (x1, y1 - 5), font, np.max(img.shape[:2]) / 1500, (0, 255, 0), 2)

    return img, prediction


def test(le, clf):
    """
    Detect and recognize a face using a trained classifier.
    """

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()

        # call the prediction function for the current frame
        img, _ = predict(image, le, clf)

        # Display the resulting frame
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face Recognition Engine")
    parser.add_argument('mode',
                        type=str,
                        help='train or test',
                        default='test')

    args = parser.parse_args()
    if args.mode == 'train':
        training('../TrainingData')
    elif args.mode == 'test':
        with open(svm_classifier_filename, 'rb') as f:
            (le, clf) = pickle.load(f)
        test(le, clf)

    cv2.destroAllWindows()
