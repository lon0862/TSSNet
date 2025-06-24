"""
    some functions called by itri2av1_map.py
    
"""

import numpy as np
import cv2
import json
from xml.dom import minidom

def create_xml(config):
    """create .xml to indicate the reference point and map information"""
    dom=minidom.Document()
    root_node=dom.createElement('ArgoverseVectorMap')
    dom.appendChild(root_node)
    
    
    with open(config['path_to_json'],'r') as f:
        data = json.load(f)['waypoints']


    ## collect coordinates of the points
    id_to_xypoints = {} # to connect lane id to node id of the points which form the lane
    point_index = -1
    for lane in data:
        xy_points = []
        for point in lane['points']:
            # every points will be assigned with a node id
            point_index += 1 
            xy_points.append(str(point_index))
            # create node node for each point
            node_node=dom.createElement('node')
            root_node.appendChild(node_node)
            node_node.setAttribute('id',str(point_index))
            node_node.setAttribute('x',str(point['x']))
            node_node.setAttribute('y',str(point['y']))

        id_to_xypoints[str(lane['lane_id'])] = xy_points # record all the points id of a lane
    
    ##
    for lane in data:
        # create way node
        way_node=dom.createElement('way')
        root_node.appendChild(way_node)
        way_node.setAttribute('lane_id',str(lane['lane_id']))
        # create a series of tag node
        tag_node1=dom.createElement('tag')
        way_node.appendChild(tag_node1)
        tag_node1.setAttribute('k',"has_traffic_control")
        tag_node1.setAttribute('v',"False")

        tag_node2=dom.createElement('tag')
        way_node.appendChild(tag_node2)
        tag_node2.setAttribute('k',"turn_direction")
        tag_node2.setAttribute('v',"False")

        tag_node3=dom.createElement('tag')
        way_node.appendChild(tag_node3)
        tag_node3.setAttribute('k',"is_intersection")
        tag_node3.setAttribute('v',"False")

        tag_node4=dom.createElement('tag')
        way_node.appendChild(tag_node4)
        tag_node4.setAttribute('k',"l_neighbor_id")
        tag_node4.setAttribute('v',"None")

        tag_node5=dom.createElement('tag')
        way_node.appendChild(tag_node5)
        tag_node5.setAttribute('k',"r_neighbor_id")
        tag_node5.setAttribute('v',"None")

        # append nd nodes for ref (reference of points which form the lane)
        for point_index in id_to_xypoints[str(lane['lane_id'])]:
            nd_node =dom.createElement('nd')
            way_node.appendChild(nd_node)
            nd_node.setAttribute('ref',point_index)

        # append tag node (don't have information from the itri dataset right now)
        # for j in range(3): # assume 3 predecessor a way
        #     tag_node=dom.createElement('tag')
        #     way_node.appendChild(tag_node)
        #     tag_node.setAttribute('k',"predecessor")
        #     tag_node.setAttribute('v',str(50000+j))
    return dom

def create_bbox_table_and_bbox_to_laneid(config):
    print("creating bbox_table(.npy) and bbox_to_laneid(.json)~~~~~")
    path_to_json = config['path_to_json']
    with open(path_to_json,'r') as f:
        data = json.load(f)['waypoints']
    my_dict = {}
    vectors = []
    ## create table_dict and find the bbox of all the lanes
    id_to_centerline_dict = {}
    for i, lane in enumerate(data):
        
        my_dict[i] = lane['lane_id']
        points = lane['points']
        max_x = -10000000
        min_x = 10000000
        max_y = -10000000
        min_y = 10000000
        centerline = []
        for point in points:
            x = point['x']
            y = point['y']

            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            centerline.append([x,y])
        centerline = np.array(centerline)
        id_to_centerline_dict[lane['lane_id']] = centerline
        vectors.append([min_x, min_y, max_x, max_y])
        
        ## To see how to get the following information
        
    vectors = np.array(vectors)
    return vectors, my_dict, id_to_centerline_dict

def create_rotation_mat_and_da_npy(config, seq_lane_props):
    """
        input:
            seq_lane_props:
                1. a dict
                2. dict[lane_id, centerline(nparray)]
    """
    ## find the map limit of x-y coordinates from centerlines information
    xmin = 10000
    xmax = -10000
    ymin = 10000
    ymax = -10000

    lane_centerlines = []
    for lane_id, lane_cl in seq_lane_props.items():

        local_max = np.max(lane_cl, axis = 0)
        local_min = np.min(lane_cl, axis = 0)
        xmax = local_max[0] if xmax < local_max[0] else xmax
        ymax = local_max[1] if ymax < local_max[1] else ymax
        xmin = local_min[0] if xmin > local_min[0] else xmin
        ymin = local_min[1] if ymin > local_min[1] else ymin
        
        lane_centerlines.append(lane_cl)

    lane_centerlines = [np.ceil(line).astype(int) for line in lane_centerlines]
    xmin = np.ceil(xmin)
    xmax = np.ceil(xmax)
    ymin = np.ceil(ymin)
    ymax = np.ceil(ymax)

    ## create the rotation matrix
    rot_mat = np.eye(3)
        ## assign the x-y coordinate origin(0,0) to the image position(10,10) 
    x_translation = np.ceil(10 - xmin).astype(int)
    y_translation = np.ceil(10 - ymin).astype(int)

    rot_mat[0,2] = x_translation
    rot_mat[1,2] = y_translation

    ## move the centerlines coordinate from the city coordinates to image coordinates
    translation = np.array([x_translation, y_translation]).reshape(2,1)
    after_translation_lane_centerlines = [line + translation.T for line in lane_centerlines]

    # check img limit
    xmin = 10000
    xmax = -10000
    ymin = 10000
    ymax = -10000

    for line in after_translation_lane_centerlines:
        local_max = np.max(line, axis = 0)
        local_min = np.min(line, axis = 0)
        xmax = local_max[0] if xmax < local_max[0] else xmax
        ymax = local_max[1] if ymax < local_max[1] else ymax
        xmin = local_min[0] if xmin > local_min[0] else xmin
        ymin = local_min[1] if ymin > local_min[1] else ymin
    # print(xmin, xmax, ymin, ymax)
    assert xmin == 10 and ymin == 10, "city coordinate origin doesn't map to the (10,10) of the map"

    ## create driveable area matrix
    img = np.zeros((ymax + 10, xmax + 10)).astype(np.uint8)

    for line in after_translation_lane_centerlines:
        cor_x = line[:,0]
        cor_y = line[:,1]
        img[cor_y,cor_x] = 1

        ## do the dilation(image processing) to the img since the 
        #  driveable area is too thin
    kernel = np.zeros((3,3), np.uint8)
    kernel[0,1] = 1
    kernel[1,0] = 1
    kernel[1,1] = 1
    kernel[1,2] = 1
    kernel[2,1] = 1
    da_img = cv2.dilate(img, kernel, iterations = 5)

    return rot_mat, da_img