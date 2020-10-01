from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN


filename = 'test1.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)

#for result in faces:
    # get coordinates
    #x, y, width, height = result['box']
    #for _, value in result['keypoints'].items():
        #print (value)
    #print (len(result['keypoints'].items()))

#{'left_eye': (207, 110), 'right_eye': (252, 119), 'nose': (220, 143), 'mouth_left': (200, 148), 'mouth_right': (244, 159)}
single_face = faces[0]
print (f"Left Eye: {single_face['keypoints']['left_eye']}")