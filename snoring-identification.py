# -*- coding: utf-8 -*-
########## Extra Credit ##########################
import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import FeatureExtractor
import os

# Load the classifier:
output_dir = 'training_output'
classifier_filename = 'classifier.pickle'

with open(os.path.join(output_dir, classifier_filename), 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
feature_extractor = FeatureExtractor(debug=False)
    


## Write the code for test code"
data = np.zeros((0, 8002))  # 8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

data_dir = './test_data/data_odd'
class_names = ["NoSnore", "Snore"]

nosnore = 0
snore = 0

snoreTP = 0
snoreFP = 0
snoreFN = 0
snoreTN = 0


nosnoreTP = 0
nosnoreFP = 0
nosnoreFN = 0
nosnoreTN = 0

j = 0

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("snore-data"):
        filename_components = filename.split("-")  # split by the '-' character
        speaker = filename_components[2]
        # print("Loading data for {}.".format(speaker))
        if speaker not in class_names:
            class_names.append(speaker)
        speaker_label = class_names.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
        # print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
        sys.stdout.flush()
        data_for_current_speaker[:, -1] = speaker_label
        # print("speaker_label =", speaker_label)

        n_features = 4
        X = np.zeros((0, n_features))
        y = np.zeros(0, )
        pred = []

        for i, window_with_timestamp_and_label in enumerate(data_for_current_speaker):
            window = window_with_timestamp_and_label[1:-1]
            # label = data[i][-1]
            label = window_with_timestamp_and_label[-1]
            # if label > 1:
            #     print("break here")
            # print(label)
            if len([x for x in window if x != 0]) != 0:
                x = feature_extractor.extract_features(window)
                if len(x) != X.shape[1]:
                    print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
                pred.extend(classifier.predict(np.reshape(x, (1, -1))))
                X = np.append(X, np.reshape(x, (1, -1)), axis=0)
                y = np.append(y, label)

        if len(pred) > 0:
            pred_label = round(sum(pred) / len(pred))
            if speaker_label == 1:
                if pred_label == 1:
                    snoreTP += 1
                    nosnoreTN += 1
                else:
                    nosnoreFP += 1
                    snoreFN += 1
            elif speaker_label == 0:
                if pred_label == 0:
                    nosnoreTP += 1
                    snoreTN += 1
                else:
                    snoreFP += 1
                    nosnoreFN += 1

        if speaker_label == 1:
            snore += 1
        elif speaker_label == 0:
            nosnore += 1

        data = np.append(data, data_for_current_speaker, axis=0)
    j += 1
    if j % 50 == 0:
        print("Predicted",j,"datasets")
        # data[-1] = speaker_label

print()
print("===============================================")
print("Prediction for [\"Snore\", \"NoSnore\"]")
print("Number of datasets:",[snore, nosnore])
print("True Positives:",[snoreTP, nosnoreTP])
print("True Negatives:",[snoreTN, nosnoreTN])
print("False Positives:",[snoreFP, nosnoreFP])
print("False Negatives:",[snoreFN, nosnoreFN])
print()
print("Accuracy:", (snoreTP + nosnoreTP)/(snore + nosnore))
print("Precision:", [snoreTP/(snoreTP + snoreFP), nosnoreTP/(nosnoreTP + nosnoreFP)])
print("Recall:", [snoreTP/(snoreTP + snoreFN), nosnoreTP/(nosnoreTP + nosnoreFN)])