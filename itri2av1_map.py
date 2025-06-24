import json
import os
import argparse
import numpy as np
from itri2av1_utils import create_xml, create_bbox_table_and_bbox_to_laneid, create_rotation_mat_and_da_npy


if __name__ == '__main__':
    ## set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../ITRI_Dataset')
    args = parser.parse_args()

    city_name = 'ITRI'
    raw_root = os.path.join(args.root, 'raw_data')
    map_root = os.path.join(raw_root, 'hd_map')
    map_file = os.path.join(map_root, 'waypoints.json')
    save_dir = os.path.join(args.root, 'av1_map')
    os.makedirs(save_dir, exist_ok=True)

    ## set config
    config = {
        'map_name':city_name,
        'map_wholename' : city_name,
        'map_id' : 0,
        'path_to_json': map_file,
        'save_dir': save_dir
    }

    ## create xml, bbox_table, bbox_to_laneid, rotatioin mat, da_img, height_img
    xml_file = create_xml(config)
    bbox_table, bbox_to_laneid, id_to_centerline_dict = create_bbox_table_and_bbox_to_laneid(config)
    rot_mat, da_img = create_rotation_mat_and_da_npy(config, id_to_centerline_dict)
    height_img = np.zeros_like(da_img).astype(np.uint8)
    ## save xml
    try:
        with open(os.path.join(config['save_dir'],f'pruned_argoverse_{config["map_name"]}_{config["map_id"]}_vector_map.xml'),'w',encoding='UTF-8') as fh:
            xml_file.writexml(fh,indent='',addindent='  ',newl='\n',encoding='UTF-8')
            print('write xml OK!')
    except Exception as err:
        print('錯誤信息：{0}'.format(err))

    ## save bbox files
    np.save(f'{config["save_dir"]}/{config["map_name"]}_{config["map_id"]}_halluc_bbox_table.npy', bbox_table)
    with open(f'{config["save_dir"]}/{config["map_name"]}_{config["map_id"]}_tableidx_to_laneid_map.json', "w") as fp:
        json.dump(bbox_to_laneid ,fp)

    ## save rotation matrix and driveable area
    np.save(f'{config["save_dir"]}/{config["map_name"]}_{config["map_id"]}_driveable_area_mat_2019_05_28.npy', da_img)
    np.save(f'{config["save_dir"]}/{config["map_name"]}_{config["map_id"]}_npyimage_to_city_se2_2019_05_28.npy', rot_mat)

    ## save height matrix
    np.save(f'{config["save_dir"]}/{config["map_name"]}_{config["map_id"]}_ground_height_mat_2019_05_28.npy', height_img)