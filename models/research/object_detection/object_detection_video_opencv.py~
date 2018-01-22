import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
i=0
j=0
cap = cv2.VideoCapture(sys.argv[1])
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
success,image=cap.read()
success=True
if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/fatoks/CodigosDeep/Skycatch/fine_tuned_model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/fatoks/CodigosDeep/Skycatch/data/', 'object-detection.pbtxt')

NUM_CLASSES = 3
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)    
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	while(cap.isOpened()):
	    ret,frame=cap.read()
	    i+=1
	    if ret and i>=4:##only read 1 of 4 frames
		i=0 
		frame=cv2.resize(frame,None,fx=0.17,fy=0.17,interpolation =cv2.INTER_CUBIC)##resize
	    	#frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	    	image=Image.fromarray(frame)		
		#image.thumbnail((250,250),Image.ANTIALIAS)
            	# the array based representation of the image will be used later in order to prepare the
            	# result image with boxes and labels on it.
            	image_np = load_image_into_numpy_array(image)
            	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            	image_np_expanded = np.expand_dims(image_np, axis=0)
            	# Actual detection.
            	(boxes, scores, classes, num) = sess.run(
              	[detection_boxes, detection_scores, detection_classes, num_detections],
              	feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            	vis_util.visualize_boxes_and_labels_on_image_array(
              	image_np,
              	np.squeeze(boxes),
              	np.squeeze(classes).astype(np.int32),
              	np.squeeze(scores),
              	category_index,
              	use_normalized_coordinates=True,
              	line_thickness=8)
		
		cv2.imshow('imagen',image_np)   
		k=cv2.waitKey(5)&0xFF
		if k==27:
			break         	
		#plt.figure(figsize=IMAGE_SIZE)
            	#plt.imshow(image_np)
cv2.destroyAllWindows()
