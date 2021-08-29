import argparse
import cv2
from datetime import datetime
from screeninfo import get_monitors

# --- XML IMPORTS ---
import xml.etree.ElementTree as ET
# -------------------

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def change_frame(frame, width, height=0):
    if height == 0:
        ratio = frame.shape[1] / frame.shape[0]
        height = int(width/ratio)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def xmlStructure(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xmlStructure(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ----- SCREEN SIZE -----
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
# -----------------------

# ----- DATA FOR XML -----
timestamp_counter = 0
timestamp_interval = 100
# ------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='C:\Repos\Samples\VID_001.MP4',
    help='Path to video file')
parser.add_argument('-a', '--accuracy', type=int, default=10,
    help='Video is split into [accuracy] parts and described every [length]/[accuracy] seconds.')
args = vars(parser.parse_args())

cap = cv2.VideoCapture(args['file'])
file_name = args['file'].split(sep="/")[-1].split(".")[0]
directory_name = args['file'].split(sep="/")[-2].split(sep="/")[0]

# XML INIT
root = ET.Element(directory_name)
xml_video_ready = True

while(cap.isOpened()):
    (ret, frame) = cap.read()

    video_width = str(frame.shape[1])
    video_height = str(frame.shape[0])
    video_fps = str(25) #TODO: automatize it

    if xml_video_ready:
        xml_video = ET.Element(file_name, attrib={'width': video_width, 'height': video_height, 'fps': video_fps})
        root.append(xml_video)
        xml_video_ready = False

    frame = change_frame(frame, 1280)
    cv2.imshow(file_name, frame)

    timestamp_counter = timestamp_counter+1
    if timestamp_counter % timestamp_interval == 0:
        timestamp = datetime.now().strftime("%H_%M_%S")
        xml_timestamp = ET.SubElement(xml_video, "TIME_" + timestamp)

        xml_object_A = ET.SubElement(xml_timestamp, "Object_A", attrib={'name': 'Cat', 'size': '500'})
        xml_object_B = ET.SubElement(xml_timestamp, "Object_B", attrib={'name': 'Person', 'size': '1900'})

        xml_general = ET.SubElement(xml_timestamp, "General")
        xml_general_light = ET.SubElement(xml_general, "Light")
        xml_general_light.text = "80%"
        xml_general_sharpness = ET.SubElement(xml_general, "Sharpness")
        xml_general_sharpness.text = "20%"

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# XML FINALIZE
tree = ET.ElementTree(root)
xmlStructure(root)

with open("output.xml", "wb") as file:
    tree.write(file,
               xml_declaration=True,
               encoding="utf-8")

