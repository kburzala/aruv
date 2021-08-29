import argparse
import xml.etree.ElementTree as ET

from pathlib import Path
import os
import shutil

from math import dist
from math import sqrt
#from math import abs

import numpy

from statistics import stdev

import matplotlib.pyplot as plt

# <?xml version='1.0' encoding='utf-8'?>
# <DIRECTORY name="girl_in_park" files="15">
  # <VIDEO name="20210629_183144" width="1920" height="1080" fps="30" length="00:00:10" size="20MB">
    # <TIME timestamp="00:00:00" detected_objects="1">
      # <OBJECT name="person" quality="0.75" x_center="0.43" y_center="0.53" width="0.10" height="0.33" />
      # <SHARPNESS edges="2116.99" laplacian="42.55" sobel_x="139924" sobel_y="284465" />
      # <SATURATION red="147" green="151" blue="146" />
    # </TIME>
    # <TIME timestamp="00:00:01" detected_objects="1">
      # <OBJECT name="person" quality="0.83" x_center="0.43" y_center="0.51" width="0.10" height="0.34" />
      # <SHARPNESS edges="2551.14" laplacian="61.30" sobel_x="146123" sobel_y="280821" />
      # <SATURATION red="141" green="145" blue="140" />
    # </TIME>
  # </VIDEO>
# </DIRECTORY>

# def weighted_avg_and_std(values, weights):
    # """
    # Return the weighted average and standard deviation.

    # values, weights -- Numpy ndarrays with the same shape.
    # """
    # average = numpy.average(values, weights=weights)
    # # Fast and numerically precise:
    # variance = numpy.average((values-average)**2, weights=weights)
    # return (average, sqrt(variance))

class Video:
    """ Object of this class represents each one video from directory """
    
    def __init__(self, name, width, height, type, fps, length, size):
        self.name = name
        self.width = width
        self.height = height
        self.type = type
        self.fps = fps
        self.length = length
        self.size = size
        
        self.frames = []
        
        # Some condtitions can make it False, e.g. too short video
        self.is_valid = True
        
        # Mean Values
        self.mean_detected_objects = -1
        self.mean_object_name = -1
        self.mean_object_quality = -1
        self.mean_object_x_center = -1
        self.mean_object_y_center = -1
        self.mean_object_area = -1
        self.mean_objects_area = -1
        self.mean_sharpness_edges = -1
        self.mean_sharpness_laplacian = -1
        self.mean_sharpness_sobel_x = -1
        self.mean_sharpness_sobel_y = -1
        
        self.mean_composity_central = -1
        self.stdev_composity_central = -1
        self.mean_composity_thirds = -1
        self.stdev_composity_thirds = -1
        self.mean_composity_symmetrical = -1
        self.stdev_composity_symmetrical = -1
        
        self.mean_saturation_red = -1
        self.ratio_saturation_red = -1
        self.mean_saturation_green = -1
        self.ratio_saturation_green = -1
        self.mean_saturation_blue = -1
        self.ratio_saturation_blue = -1
        
        # For ARUV
        self.scene = -1
        self.shot = -1
        
        
    def count_average():
        return True
    
    def count_ranking():
        return True



class Frame:
    """ Object of this class represents single frame from videos 
        (they have timestamp, video and directory fields to allow to read them seperately) """
    
    def __init__(self, timestamp, video, directory):
        self.timestamp = timestamp
        self.video = video
        self.directory = directory
        
        self.objects = []
        
        self.sharpness_edges = -1
        self.sharpness_laplacian = -1
        self.sharpness_sobel_x = -1
        self.sharpness_sobel_y = -1
        
        self.saturation_red = -1
        self.saturation_green = -1
        self.saturation_blue = -1
        
        self.composity_central = -1         # <0, 1.0>
        self.composity_thirds = -1           # <0, 1.0> = <0, 0.25> x 4
        self.composity_symmetrical = -1 # <0, 1.0> = <0, 0.10> x 10
        



class Object:
    """ Object of this class represents detected object from single frame """
    
    def __init__(self, name, quality, x_center, y_center, width, height):
        self.name = name
        self.quality = quality
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    


def run_aruv(source='output.xml'):

    """
    H I E R A R C H I A
    ( I ) ostrosc
    ( II ) kompozycja
        (i) centralna
        (ii) krawedzie
    ( III ) ciemnosc
    ( IV ) jasnosc
    """

    # XML file load to root
    tree = ET.parse(source)
    #print("Tree loaded...")
    root = tree.getroot()
    #print("Root loaded...")
    
    dir_name = root.attrib['name']
    files_number = root.attrib['files']
    
    #print("Directory =", dir_name)
    #print("Number of files =", files_number)

    videos = []
    
    video_counter = 0
    frame_counter = 0
    object_counter = 0
    global_object_counter = 0
    
    # === VIDEO === #
    for level_1 in root:
        videos.append (
            Video (
                level_1.attrib['name'],
                int(level_1.attrib['width']),
                int(level_1.attrib['height']),
                level_1.attrib['type'],
                int(level_1.attrib['fps']),
                level_1.attrib['length'],
                level_1.attrib['size']
            )
        )
        
        sum_detected_objects = 0
        sum_object_name = ""
        sum_object_quality = 0
        sum_object_x_center = 0
        sum_object_y_center = 0
        sum_object_area = 0
        sum_objects_area = 0
        sum_sharpness_edges = 0
        sum_sharpness_laplacian = 0
        sum_sharpness_sobel_x = 0
        sum_sharpness_sobel_y = 0
        sum_saturation_red = 0
        sum_saturation_green = 0
        sum_saturation_blue = 0
        
        # === TIMESTAMP === #
        for level_2 in level_1:
            video = videos[video_counter]
            video.frames.append (
                Frame (
                    level_2.attrib['timestamp'],
                    level_1.attrib['name'],
                    dir_name
                )
            )
            
            if level_2.attrib['timestamp'] == "--:--:--":
                video.is_valid = False
                break
            
            # === FEATURES === #
            for level_3 in level_2:
                frame = video.frames[frame_counter]
                if object_counter < int(level_2.attrib['detected_objects']):
                    # <OBJECT>
                    frame.objects.append (
                        Object (
                            level_3.attrib['name'],
                            float(level_3.attrib['quality']),
                            float(level_3.attrib['x_center']),
                            float(level_3.attrib['y_center']),
                            float(level_3.attrib['width']),
                            float(level_3.attrib['height'])
                        )
                    )
                    object_counter = object_counter + 1
                    global_object_counter = global_object_counter + 1
                else:
                    if frame.sharpness_edges == -1:
                        # <SHARPNESS>
                        frame.sharpness_edges = float(level_3.attrib['edges'])
                        frame.sharpness_laplacian = float(level_3.attrib['laplacian'])
                        frame.sharpness_sobel_x = int(level_3.attrib['sobel_x'])
                        frame.sharpness_sobel_y = int(level_3.attrib['sobel_y'])
                        
                        sum_sharpness_edges = sum_sharpness_edges + float(level_3.attrib['edges'])
                        sum_sharpness_laplacian = sum_sharpness_laplacian + float(level_3.attrib['laplacian'])
                        sum_sharpness_sobel_x = sum_sharpness_sobel_x + int(level_3.attrib['sobel_x'])
                        sum_sharpness_sobel_y = sum_sharpness_sobel_y + int(level_3.attrib['sobel_y'])
                    
                    elif frame.saturation_red == -1:
                        # <SATURATION>
                        frame.saturation_red = int(level_3.attrib['red'])
                        frame.saturation_green = int(level_3.attrib['green'])
                        frame.saturation_blue = int(level_3.attrib['blue'])
                        
                        sum_saturation_red = sum_saturation_red + int(level_3.attrib['red'])
                        sum_saturation_green = sum_saturation_green + int(level_3.attrib['green'])
                        sum_saturation_blue = sum_saturation_blue + int(level_3.attrib['blue'])
                        
            # END level_3 LOOP
            object_counter = 0    
            frame_counter = frame_counter + 1
        
        # END level_2 LOOP
        if video.is_valid:
            video.mean_sharpness_edges = "{:.2f}".format(sum_sharpness_edges / frame_counter)
            video.mean_sharpness_laplacian = "{:.2f}".format(sum_sharpness_laplacian / frame_counter)
            video.mean_sharpness_sobel_x = int(sum_sharpness_sobel_x / frame_counter)
            video.mean_sharpness_sobel_y = int(sum_sharpness_sobel_y / frame_counter)
            
            m_red = int(sum_saturation_red / frame_counter)
            m_green = int(sum_saturation_green / frame_counter)
            m_blue = int(sum_saturation_blue / frame_counter)
            m_all = m_red + m_green + m_blue
            
            video.mean_saturation_red = m_red
            video.ratio_saturation_red = "{:.3f}".format(m_red / m_all)
            video.mean_saturation_green = m_green
            video.ratio_saturation_green = "{:.3f}".format(m_green / m_all)
            video.mean_saturation_blue = m_blue
            video.ratio_saturation_blue = "{:.3f}".format(m_blue / m_all)
        
        frame_counter = 0
        video_counter = video_counter + 1
    
    # END level_1 LOOP
    video_counter = 0
    
    # TRIPLE C METRIC: Composity
    for video in videos:
        if not video.is_valid:
            continue
    
        # [video] COMPOSITY: Central
        mean_central_sum = 0
        composity_central_list = []
        
        # [video] COMPOSITY: Thirds
        mean_thirds_sum = 0
        composity_thirds_list = []
        
        # [video] COMPOSITY: Symmetrical
        mean_symmetrical_sum = 0
        composity_symmetrical_list = []
        
        for frame in video.frames:
            
            # [frame] COMPOSITY: Central
            central_sum = 0
            central_weights = 0
            
            # [frame] COMPOSITY: Thirds
            thirds_pkt1_sum = 0
            thirds_pkt2_sum = 0
            thirds_pkt3_sum = 0
            thirds_pkt4_sum = 0
            
            thirds_min_first_sum = 0
            thirds_min_first_corner = 0
            thirds_min_first_point = [0.0, 0.0]
            
            thirds_min_second_sum = 0
            thirds_min_second_corner = 0
            thirds_min_second_point = [0.0, 0.0]
            
            thirds_min_x = 0.0
            thirds_max_x = 1.0
            thirds_min_y = 0.0
            thirds_max_y = 1.0
            
            thirds_sector_value = 0.0
            
            thirds_sum = 0
            thirds_weights = 0
            
            # [frame] COMPOSITY: Symmetrical
            symmetrical_cells = [0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0]
            # left_external_cells = [0,0,0,0,0]
            # left_internal_cells = [0,0,0,0,0]
            # right_internal_cells = [0,0,0,0,0]
            # right_external_cells = [0,0,0,0,0]
            
            symmetrical_rows = [0.2, 0.4, 0.6, 0.8, 1.0]       #  +0.2
            symmetrical_columns = [0.25, 0.5, 0.75, 1.0]     #  +0.25
            
            symmetrical_up_left_sum = 0.0001
            symmetrical_up_right_sum = 0.0001
            symmetrical_down_left_sum = 0.0001
            symmetrical_down_right_sum = 0.0001

            
            for object in frame.objects:
            
                object_area = object.width * object.height
                
                # [object] COMPOSITY: Central
                central_sum = central_sum + ( dist([object.x_center, object.y_center], [0.5, 0.5]) * object_area )
                central_weights = central_weights + object_area
                
                # [object] COMPOSITY: Pre-Thirds
                # 1  |  2                                                        A  x = [0, 1.0], y = [0, 0.5]    
                # --------                  D  x = [0, 0.5], y = [0, 1.0]                                     B  x = [0.5, 1.0], y = [0, 1.0]    
                # 4  |  3                                                        C  x = [0, 1.0], y = [0.5, 1.0]    
                #
                thirds_pkt1_sum = thirds_pkt1_sum + ( dist([object.x_center, object.y_center], [0.33, 0.33]) * object_area ) # 1, value <0.00, 0.25> so    * 0.25    + 0
                thirds_pkt2_sum = thirds_pkt2_sum + ( dist([object.x_center, object.y_center], [0.66, 0.33]) * object_area ) # 2, value <0.25, 0.50> so    * 0.25    + 0.25
                thirds_pkt3_sum = thirds_pkt3_sum + ( dist([object.x_center, object.y_center], [0.66, 0.66]) * object_area ) # 3, value <0.50, 0.75> so    * 0.25    + 0.50
                thirds_pkt4_sum = thirds_pkt4_sum + ( dist([object.x_center, object.y_center], [0.33, 0.66]) * object_area ) # 4, value <0.75, 1.00> so    * 0.25    + 0.75
           
                # [object] COMPOSITY: Symmetrical
                # object_p1 = [object.x_center - object.width/2, object.y_center - object.height/2] 
                # object_p2 = [object.x_center + object.width/2, object.y_center + object.height/2]
               
                # column_counter = 0
                # column_start = 0
                # column_end = 0
                
                # row_counter = 0
                # row_start = 0
                # row_end = 0
                
                if object.x_center > 0.5:
                    if object.y_center > 0.5:
                        symmetrical_down_right_sum = symmetrical_down_right_sum + object_area
                    else:
                        symmetrical_up_right_sum = symmetrical_up_right_sum + object_area
                else:
                    if object.y_center > 0.5:
                        symmetrical_down_left_sum = symmetrical_down_left_sum + object_area
                    else:
                        symmetrical_up_left_sum = symmetrical_up_left_sum + object_area


                # for column in symmetrical_columns:
                    # for row in symmetrical_rows:
                        # if object_p1[0] < column and object_p1[1] < row:
                            # column_start = column_counter
                            # row_start = row_counter
                        # if object_p2[0] > (column - 0.25) and object_p2[1] > (row - 0.2):
                            # column_end = column_counter
                            # row_end = row_counter
                        # row_counter = row_counter + 1
                    # column_counter = column_counter + 1
                    # row_counter = 0
                    
                
                # for column in range(column_counter - 1):
                    # for row in range(row_counter - 1):
                        # if row >= row_start and column >= column_start and row <= row_end and column <= column_end:
                            # symmetrical_cells[column * len(symmetrical_rows) + row] = 1.0
                
            # [frame] COMPOSITY: Central
            if central_weights == 0:
                frame.composity_central = 0
            else:
                frame.composity_central = 1 - ( (  central_sum / central_weights ) / dist([0.0, 0.0], [0.5, 0.5]) )
            mean_central_sum = mean_central_sum + frame.composity_central
            
            composity_central_list.append(frame.composity_central)
            
            
            # [frame] COMPOSITY: Thirds... can't be done now
            
            
            # [frame] COMPOSITY: Symmetrical
            # cells_per_side = int((len(symmetrical_rows) * len(symmetrical_columns)) / 2)
            
            # frame.composity_symmetrical = 0
            
            # for cell in range(cells_per_side):
                # if symmetrical_cells[cell] == symmetrical_cells[cell + cells_per_side]:
                    # frame.composity_symmetrical = frame.composity_symmetrical + (1.0 / cells_per_side) 
                    
            composity_symmetrical_up = 0
            composity_symmetrical_down = 0
                    
            # print("up_left =", symmetrical_up_left_sum)
            # print("up_right =", symmetrical_up_right_sum)
            # print("down_right =", symmetrical_down_right_sum)
            # print("down_left =", symmetrical_down_left_sum)
            
            if symmetrical_up_left_sum < symmetrical_up_right_sum:
                composity_symmetrical_up = symmetrical_up_left_sum / symmetrical_up_right_sum
            else:
                composity_symmetrical_up = symmetrical_up_right_sum / symmetrical_up_left_sum
            
            if symmetrical_down_left_sum < symmetrical_down_right_sum:
                composity_symmetrical_down = symmetrical_down_left_sum / symmetrical_down_right_sum
            else:
                composity_symmetrical_down = symmetrical_down_right_sum / symmetrical_down_left_sum
            
            frame.composity_symmetrical = (composity_symmetrical_up + composity_symmetrical_down) / 2
            mean_symmetrical_sum = mean_symmetrical_sum + frame.composity_symmetrical
            composity_symmetrical_list.append(frame.composity_symmetrical)
            
            
            # [frame] COMPOSITY: Pre-Thirds
            # choose FIRST corner
            thirds_min_first_sum = thirds_pkt1_sum
            thirds_min_first_corner = 1
            
            if thirds_pkt2_sum < thirds_min_first_sum:
                thirds_min_first_sum = thirds_pkt2_sum
                thirds_min_first_corner = 2
                
            if thirds_pkt3_sum < thirds_min_first_sum:
                thirds_min_first_sum = thirds_pkt3_sum
                thirds_min_first_corner = 3
                
            if thirds_pkt4_sum < thirds_min_first_sum:
                thirds_min_first_sum = thirds_pkt4_sum
                thirds_min_first_corner = 4
                
            # choose SECOND corner - if FIRST is "1"
            if thirds_min_first_corner == 1:
                thirds_min_second_sum = thirds_pkt2_sum
                thirds_min_second_corner = 2
                
                # UP sector (A)
                thirds_min_x = 0.0
                thirds_max_x = 1.0
                thirds_min_y = 0.0
                thirds_max_y = 0.5
                thirds_sector_value = 0.0
                
                if thirds_pkt4_sum < thirds_min_second_sum:
                    thirds_min_second_sum = thirds_pkt4_sum
                    thirds_min_second_corner = 4
                    
                    # LEFT sector (D)
                    thirds_min_x = 0.0
                    thirds_max_x = 0.5
                    thirds_min_y = 0.0
                    thirds_max_y = 1.0
                    thirds_sector_value = 0.75
            
            # choose SECOND corner - if FIRST is "2"
            elif thirds_min_first_corner == 2:
                thirds_min_second_sum = thirds_pkt1_sum
                thirds_min_second_corner = 1
                
                # UP sector (A)
                thirds_min_x = 0.0
                thirds_max_x = 1.0
                thirds_min_y = 0.0
                thirds_max_y = 0.5
                thirds_sector_value = 0.0
                
                if thirds_pkt3_sum < thirds_min_second_sum:
                    thirds_min_second_sum = thirds_pkt3_sum
                    thirds_min_second_corner = 3
                    
                    # RIGHT sector (B)
                    thirds_min_x = 0.5
                    thirds_max_x = 1.0
                    thirds_min_y = 0.0
                    thirds_max_y = 1.0
                    thirds_sector_value = 0.25
            
            # choose SECOND corner - if FIRST is "3"
            elif thirds_min_first_corner == 3:
                thirds_min_second_sum = thirds_pkt2_sum
                thirds_min_second_corner = 2
                
                # RIGHT sector (B)
                thirds_min_x = 0.5
                thirds_max_x = 1.0
                thirds_min_y = 0.0
                thirds_max_y = 1.0
                thirds_sector_value = 0.25
                
                if thirds_pkt4_sum < thirds_min_second_sum:
                    thirds_min_second_sum = thirds_pkt4_sum
                    thirds_min_second_corner = 4
                    
                    # DOWN sector (C)
                    thirds_min_x = 0.0
                    thirds_max_x = 1.0
                    thirds_min_y = 0.5
                    thirds_max_y = 1.0
                    thirds_sector_value = 0.5
            
            # choose SECOND corner - if FIRST is "4"
            elif thirds_min_first_corner == 4:
                thirds_min_second_sum = thirds_pkt1_sum
                thirds_min_second_corner = 1
                
                # LEFT sector (D)
                thirds_min_x = 0.0
                thirds_max_x = 0.5
                thirds_min_y = 0.0
                thirds_max_y = 1.0
                thirds_sector_value = 0.75
                
                if thirds_pkt3_sum < thirds_min_second_sum:
                    thirds_min_second_sum = thirds_pkt3_sum
                    thirds_min_second_corner = 3
                    
                    # DOWN sector (C)
                    thirds_min_x = 0.0
                    thirds_max_x = 1.0
                    thirds_min_y = 0.5
                    thirds_max_y = 1.0
                    thirds_sector_value = 0.5
                    
            if thirds_min_first_corner == 1:
                thirds_min_first_point = [0.33, 0.33]
            elif thirds_min_first_corner == 2:
                thirds_min_first_point = [0.66, 0.33]
            elif thirds_min_first_corner == 3:
                thirds_min_first_point = [0.66, 0.66]
            elif thirds_min_first_corner == 4:
                thirds_min_first_point = [0.33, 0.66]
                
            if thirds_min_second_corner == 1:
                thirds_min_second_point = [0.33, 0.33]
            elif thirds_min_second_corner == 2:
                thirds_min_second_point = [0.66, 0.33]
            elif thirds_min_second_corner == 3:
                thirds_min_second_point = [0.66, 0.66]
            elif thirds_min_second_corner == 4:
                thirds_min_second_point = [0.33, 0.66]
                            
            for object in frame.objects:
                # [object] COMPOSITY: Thirds             
                if object.x_center > thirds_min_x and object.x_center < thirds_max_x and object.y_center > thirds_min_y and object.y_center < thirds_max_y:
                    object_area = object.width * object.height
                     
                    object_distance_first = dist([object.x_center, object.y_center], thirds_min_first_point)
                    object_distance_second = dist([object.x_center, object.y_center], thirds_min_second_point)
                    if object_distance_first > object_distance_second:
                        object_distance_first = object_distance_second
                        
                    thirds_sum = thirds_sum + ( object_distance_first * object_area )
                    thirds_weights = thirds_weights + object_area
                    
                   
            # [frame] COMPOSITY: Thirds
            if thirds_weights == 0:
                frame.composity_thirds = 0
            else:
                frame.composity_thirds = ( ( 1 - ( (  thirds_sum / thirds_weights ) / dist([0.0, 0.0], [0.33, 0.33]) ) ) / 4 ) + thirds_sector_value
            mean_thirds_sum = mean_thirds_sum + frame.composity_thirds
            
            composity_thirds_list.append(frame.composity_thirds)
            
           
            

        # [video] COMPOSITY: Central
        video.mean_composity_central = mean_central_sum / len(video.frames)
        video.stdev_composity_central = stdev(composity_central_list)
        
        # [video] COMPOSITY: Thirds
        video.mean_composity_thirds = mean_thirds_sum / len(video.frames)
        video.stdev_composity_thirds = stdev(composity_thirds_list)
            
        # [video] COMPOSITY: Symmetrical
        video.mean_composity_symmetrical = mean_symmetrical_sum / len(video.frames)
        video.stdev_composity_symmetrical = stdev(composity_symmetrical_list)



    # ---
    # WORKING ON PREPARED OBJECTS
    # ---
    
    #for video in videos:
        #print(video.is_valid)
 
    old_directory = source.replace("_output.xml", "")
    new_directory = old_directory + "_ARUV"
    
    if Path(new_directory).exists():
        shutil.rmtree(new_directory)
    
    Path(new_directory).mkdir(parents=True, exist_ok=True)

    
    scene_number = 1
    new_scene_detected = True
    shot_number = 1
    
    # DIRECTORY FEATURES
    dir_stdev_detected_objects = -1
    dir_stdev_object_quality = -1
    dir_stdev_object_x_center = -1
    dir_stdev_object_y_center = -1
    dir_stdev_object_area = -1
    dir_stdev_objects_area = -1
    dir_stdev_sharpness_edges = -1
    dir_stdev_sharpness_laplacian = -1
    dir_stdev_sharpness_sobel_x = -1
    dir_stdev_sharpness_sobel_y = -1
    dir_stdev_saturation_red = -1
    dir_stdev_saturation_green = -1
    dir_stdev_saturation_blue = -1
    
    mean_detected_objects_list = []
    mean_object_name_list = []
    mean_object_quality_list = []
    mean_object_x_center_list = []
    mean_object_y_center_list = []
    mean_object_area_list = []
    mean_objects_area_list = []
    mean_sharpness_edges_list = []
    mean_sharpness_laplacian_list = []
    mean_sharpness_sobel_x_list = []
    mean_sharpness_sobel_y_list = []
    mean_saturation_red_list = []
    mean_saturation_green_list = []
    mean_saturation_blue_list = []
    
    ratio_saturation_red_list = []
    ratio_saturation_green_list = []
    ratio_saturation_blue_list = []
    
    files_list = []
    files_names = []
    
    # for file in files_number:
        # files_list.append(int(file) + 1)
        
        
    test_list_1 = []
    test_list_1a = []
    test_list_2 = []
    test_list_2a = []
    test_list_3 = []
    test_list_3a = []
    test_list_4 = []
    test_list_5 = []
    test_list_6 = []
    test_list_7 = []
    test_list_8 = []
    test_list_9 = []
    test_list_10 = []
    
    test_list_rate = []
    
    # COPYING TO _ARUV
    for video in videos:
    
        if not video.is_valid:
            continue

        #print(video.ratio_saturation_red)
        #print(video.ratio_saturation_green)
        #print(video.ratio_saturation_blue)
        #print("\n")
    
        # ratio_saturation_red_list.append(float(video.ratio_saturation_red))
        # ratio_saturation_green_list.append(float(video.ratio_saturation_green))
        # ratio_saturation_blue_list.append(float(video.ratio_saturation_blue))
        
        files_list.append(video_counter + 1)
        video_counter = video_counter + 1
        
        print("COMPOSITY: Central")
        print(video.mean_composity_central) # G
        print(video.stdev_composity_central) # G + R
        print("")
        print("COMPOSITY: Thirds")
        print(video.mean_composity_thirds) # G
        print(video.stdev_composity_thirds) # G + R
        print("")
        print("COMPOSITY: Simmetrical")
        print(video.mean_composity_symmetrical) # G
        print(video.stdev_composity_symmetrical) # G + R
        print("")
        
        print("-----")
        print("")
        
        test_list_1.append(float(video.mean_composity_central) * 0.5)
        test_list_1a.append(float(video.stdev_composity_central))
        test_list_2.append(float(video.mean_composity_thirds) * 0.5)
        test_list_2a.append(float(video.stdev_composity_thirds))
        test_list_3.append(float(video.mean_composity_symmetrical) * 0.5)
        test_list_3a.append(float(video.stdev_composity_symmetrical))
        test_list_4.append(float(video.mean_sharpness_edges)/24000)
        test_list_5.append(float(video.mean_sharpness_laplacian)/4000)
        test_list_6.append(float(video.mean_sharpness_sobel_x)/4000000)
        test_list_7.append(float(video.mean_sharpness_sobel_y)/4000000)
        test_list_8.append(float(video.ratio_saturation_red))
        test_list_9.append(float(video.ratio_saturation_green))
        test_list_10.append(float(video.ratio_saturation_blue))
                
        
        
    
        # video_to_copy = video.name + "." + video.type
        # shutil.copy(old_directory + "/" + video_to_copy, new_directory + "/" + "SCENA_" + str(scene_number) + "_DUBEL_" + str(shot_number) + "." + video.type)
        
        # if new_scene_detected:
            # scene_number = scene_number + 1

    #fig, ax = plt.subplots()
    
    
    for file in range(len(files_list)):
        test_list_rate.append(
            0.3 * max([test_list_1[file], test_list_2[file], test_list_3[file]]) + 
            0.5 * (sum([test_list_4[file], test_list_5[file], test_list_6[file], test_list_7[file]]) / 4) +
            0.2 * test_list_8[file])


    # plt.plot(files_list, test_list_2, color='orange', linewidth=2, label='Laplace Operator')
    # plt.plot(files_list, test_list_3, color='yellow', linewidth=3, label='Sobel Derivatives X')
    # plt.plot(files_list, test_list_4, color='gray', linewidth=1, label='Sobel Derivatives Y' )
    
    plt.plot(files_list, test_list_1, color='black', linewidth=3, label='Composity: Central')
    plt.plot(files_list, test_list_2, color='black', linewidth=1, label='Composity: Thirds')
    plt.plot(files_list, test_list_3, color='black', linewidth=1, linestyle=':', label='Composity: Symmetrical')
    
    # plt.plot(files_list, test_list_1a, color='red', linewidth=3, label='Standard Devation: Central')
    # plt.plot(files_list, test_list_2a, color='red', linewidth=1, label='Standard Devation: Thirds')
    # plt.plot(files_list, test_list_3a, color='red', linewidth=1, linestyle=':', label='Standard Devation: Symmetrical')
    
    plt.plot(files_list, test_list_4, color='yellow', linewidth=2, label='Sharpness: Edges')
    plt.plot(files_list, test_list_5, color='orange', linewidth=2, label='Sharpness: Laplacian')
    plt.plot(files_list, test_list_6, color='pink', linewidth=3, label='Sharpness: Sobel X')
    plt.plot(files_list, test_list_7, color='pink', linewidth=1, label='Sharpness: Sobel Y')

    plt.plot(files_list, test_list_8, color='red', linewidth=2, label='Saturation: Red')
    plt.plot(files_list, test_list_9, color='green', linewidth=2, label='Saturation: Green')
    plt.plot(files_list, test_list_10, color='blue', linewidth=2, label='Saturation: Blue')
    
    plt.plot(files_list, test_list_rate, color='aqua', linewidth=4, label='Estimated Rate')
    

    #plt.plot(files_list, ratio_saturation_blue_list, color='black', linewidth=2,)
    #plt.plot(files_list, ratio_saturation_blue_list, color='yellow', linewidth=2,)
    
    plt.xticks(files_list,files_list)

    # plt.title('Saturation analysis (ARUV)', fontsize=14)
    # plt.xlabel('Valid videos', fontsize=14)
    # plt.ylabel('Color ratio', fontsize=14)
    
    #plt.yscale("log") Press F to maximize

    plt.title('General Overview (ARUV)', fontsize=14)
    plt.xlabel('Valid videos', fontsize=14)
    plt.ylabel('Values', fontsize=14)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    plt.legend()
    #plt.grid(True)
    plt.show()

    


    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='output.xml', help='output.xml')
    opt = parser.parse_args()
    return opt

def main(opt):
    run_aruv(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
