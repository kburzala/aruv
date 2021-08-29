import argparse
import time
from pathlib import Path

import os
import numpy as np
from matplotlib import pyplot as plt

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

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
            
def print_arr(arr, name='arr'):
    print('{}({}):\n{}'.format(name, arr.dtype, arr))

@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=True,  # show results
           save_txt=True,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           borders=True,
           frame_interval = 1000,
           ):

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    # ----- SCREEN SIZE -----
    screen_width = get_monitors()[0].width
    screen_height = get_monitors()[0].height

    timestamp_counter = 0
    
    sourceType = ""

    if ".MP4" in source or ".mp4" in source or ".braw" in source:
        print("Run ARUV for MP4 file")
        sourceType = "file"
        directory_name = source.split(sep="/")[-2]
        file_path_without_file = source.replace(source.split(sep="/")[-1],"")
    else:
        print("Run ARUV for directory")
        sourceType = "directory"
        directory_name = source.split(sep="/")[-1]
        file_path_without_file = source
        
    xmlFileCounter = 0
    for root, dirs, files in os.walk(file_path_without_file):
        for file in files:    
            if file.endswith('.xml'):
                xmlFileCounter += 1
                
    file_counter = 0
    files_number = str(len(os.listdir(file_path_without_file)) - xmlFileCounter)
    #directory_name = "DIR_" + directory_name
        
    root = ET.Element("DIRECTORY", attrib={'name': directory_name, 'files': files_number})
    #xml_video_ready = True
    
    skip_to_next_file = ""
    new_path = ""
    #video_fps = -1

    # ===== DIRECTORY LOOP ===== #
    for path, img, im0s, vid_cap in dataset:

        if skip_to_next_file == path:
            continue
            
        if new_path != path: #or current_path != path:
            print("\nNext Video added!", path.split("/")[-1])
            video_type = str(path.split(".")[-1])
            video_fps = str(int(vid_cap.get(5)))
            video_total_frames = str(int(vid_cap.get(7)))
            video_length = str(time.strftime('%H:%M:%S', time.gmtime(int(int(video_total_frames) / int(video_fps)))))
            video_width = str(int(vid_cap.get(3)))
            video_height = str(int(vid_cap.get(4)))
            video_size = str(int(os.path.getsize(path)/1000000))
            timestamp = 0
            timestamp_counter = 0
            xml_video = ET.Element("VIDEO", attrib={'name':path.split("/")[-1].split(".")[0],
                                                                          'width': video_width,
                                                                          'height': video_height,
                                                                          'type': video_type,
                                                                          'fps': video_fps,
                                                                          'length': video_length,
                                                                          'size': video_size + "MB"})
            root.append(xml_video)
            if video_length < "00:00:05":
                print("Video shorter than 5 seconds. Skipping...")
                ET.SubElement(xml_video, "TIME", attrib={'timestamp': "--:--:--"})
                timestamp_counter = 0
                skip_to_next_file = path
                continue
            #else:
                #xml_video_ready = False
                
        new_path = path
        
        if timestamp_counter % int(video_fps) != 0:
            timestamp_counter = timestamp_counter + 1
            continue
        #else:
            #print("Next Timestamp!")
            #print("VIDEO DEBUG:", path.split("/")[-1].split(".")[0])
            
            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):

            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            imc = im0.copy() if save_crop else im0 
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                if timestamp_counter % int(video_fps) == 0:
                    #print("Next Timestamp!")
                    timestamp = str(time.strftime('%H:%M:%S', time.gmtime(int(timestamp_counter / int(video_fps)))))
                    xml_timestamp = ET.SubElement(xml_video, "TIME", attrib={'timestamp': timestamp, 'detected_objects': "0"})

                    xyxyCounter = 0
                    
                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        c = int(cls) 
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        
                        # object + str(xyxyCounter)
                        
                        xyxyCounter = xyxyCounter + 1
                        xml_object = ET.SubElement(xml_timestamp,
                                                                    "OBJECT",
                                                                     attrib={'name': label.replace(label.split(" ")[-1],"").strip(),
                                                                                'quality': label.split(" ")[-1].strip(),
                                                                                'x_center': str("{:.2f}".format(xywh[0])),
                                                                                'y_center': str("{:.2f}".format(xywh[1])), 
                                                                                'width': str("{:.2f}".format(xywh[2])), 
                                                                                'height': str("{:.2f}".format(xywh[3]))
                                                                                })
                        if view_img:  # Add bbox to image
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    
                    #for element in root.findall("DIRECTORY"):                  
                    # for video in root:
                        # for times in video:
                            # print("time =", times)
                            # if timestamp in times.attrib['timestamp']:
                                # times.set('detected_objects', str(xyxyCounter))
                                
                    xml_timestamp.set('detected_objects', str(xyxyCounter))
                    
                    if view_img:       
                        resized_display = cv2.resize(im0, (1280, 720))
                        cv2.imshow(path, resized_display)
                        cv2.waitKey(1000)

                    #xml_screen = ET.SubElement(xml_timestamp, "SCREEN")
                    
                    _, frameToAnalyse = vid_cap.read()
                    med_val = np.median(frameToAnalyse) 
                    lowerEdge = int(max(0, 0.7 * med_val))
                    upperEdge = int(min(255, 1.3 * med_val))

                    edges_var = cv2.Canny(frameToAnalyse, lowerEdge, upperEdge).var()
                    laplacian_var = cv2.Laplacian(frameToAnalyse, cv2.CV_64F).var()
                    sobelx_var = cv2.Sobel(frameToAnalyse, cv2.CV_64F, 1, 0, ksize=5).var()
                    sobely_var = cv2.Sobel(frameToAnalyse, cv2.CV_64F, 0, 1, ksize=5).var()

                    xml_screen_sharpness = ET.SubElement(xml_timestamp, "SHARPNESS", attrib={'edges': str("{:.2f}".format(edges_var)),
                                                                                                                                   'laplacian': str("{:.2f}".format(laplacian_var)),
                                                                                                                                   'sobel_x': str(int(sobelx_var)),
                                                                                                                                   'sobel_y': str(int(sobely_var))
                                                                                                                                   })
                    
                    avgR = np.mean(frameToAnalyse[:,:,2])
                    avgG = np.mean(frameToAnalyse[:,:,1])
                    avgB = np.mean(frameToAnalyse[:,:,0])
                    
                    xml_screen_histograms = ET.SubElement(xml_timestamp, "SATURATION", attrib={'red': str(int(avgR)),
                                                                                                                                   'green': str(int(avgG)),
                                                                                                                                   'blue': str(int(avgB))
                                                                                                                                   })
                                                                                                                                   
            
           
           
            # (h, m, s) = timestamp.split(':')
            # timestamp_result = int(h) * 3600 + int(m) * 60 + int(s)
            
            # (h, m, s) = video_length.split(':')
            # video_length_result = int(h) * 3600 + int(m) * 60 + int(s)
            
            # print("Next Timestamp!", timestamp_result, "/", video_length_result)
            print("Next Timestamp!", timestamp_counter)
            timestamp_counter = timestamp_counter + 1

            # if video_length == timestamp:
                # #timestamp_counter = 0
                # skip_to_next_file = path
                # xml_video_ready = True
                # break
                
        #else:
        #    continue
        #continue
            
    if update:
        strip_optimizer(weights) 

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    tree = ET.ElementTree(root)
    xmlStructure(root)
    
    with open(source.split(".")[0] + "_output.xml", "wb") as file:
        tree.write(file,
            xml_declaration=True,
            encoding="utf-8")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--frame_interval', default=1000, type=int, help='hide confidences')
    opt = parser.parse_args()
    return opt


def main(opt):
    #print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    #check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
