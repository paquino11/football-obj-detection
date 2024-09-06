

create a folder on the root dir called
input_videos
and add this video to it
https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view

run the yolo_infernece.py using the yolov8x.pt model, which means it will detect objects on it but not players, refs, goalkeepers, boal

them go to a google colab file, change to a GPU processor and train a new model with a dataset with the labels we want like, players, refs, goalkeepers

download the best.pt and add it to the models folder on the root dir

and change the yolo_inference.py to run with the new model

