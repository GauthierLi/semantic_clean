import os
import sys
import numpy as np
from typing import List, Union
from pathlib import Path

# Set protobuf implementation to pure Python to avoid version conflicts
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add current directory (project root) and proto_gen to path
current_dir = os.getcwd()
proto_gen_dir = os.path.join(current_dir, 'proto_gen')
sys.path.insert(0, current_dir)
sys.path.insert(0, proto_gen_dir)

try:
    import clean_subtype_pb2
    print("✅ clean_subtype_pb2 imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Available files in proto_gen:")
    if os.path.exists(proto_gen_dir):
        for f in os.listdir(proto_gen_dir):
            print(f"  {f}")
    sys.exit(1)
from google.protobuf.json_format import MessageToJson

def load_npy_files(input_path: str) -> List[np.ndarray]:
    """Load npy files from different input types"""
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Input is a directory - load all .npy files
        npy_files = list(input_path.glob('*.npy'))
        if not npy_files:
            raise ValueError(f"No .npy files found in directory: {input_path}")
        return [np.load(str(f), allow_pickle=True) for f in npy_files]
    
    elif input_path.suffix == '.npy':
        # Input is a single .npy file
        return [np.load(str(input_path), allow_pickle=True)]
    
    elif input_path.suffix == '.txt':
        # Input is a text file with .npy paths
        with open(input_path, 'r') as f:
            npy_paths = [line.strip() for line in f if line.strip()]
        if not npy_paths:
            raise ValueError(f"No valid paths found in text file: {input_path}")
        return [np.load(p, allow_pickle=True) for p in npy_paths]
    
    else:
        raise ValueError(f"Unsupported input type: {input_path}. Must be .npy file, directory, or .txt file")

def get_category_labels_from_sample(sample) -> List[int]:
    """Convert npy sample data to multiple category labels based on subtype_dataset logic"""
    categories = []
    
    # Sample structure:
    # [image_id, bbox, gt_class, sub_type, sp_vehicle, profession, 
    #  car_door_state, door_open_position, segment, face_toward, 
    #  robotaxi, child, crossing, stop_sign?]
    
    if len(sample) < 2:
        return categories
    
    try:
        gt_class = int(sample[2]) if len(sample) > 2 else -1
        sub_type = int(sample[3]) if len(sample) > 3 and sample[3] != '' else -1
        sp_vehicle = int(sample[4]) if len(sample) > 4 and sample[4] != '' else -1
        profession = int(sample[5]) if len(sample) > 5 and sample[5] != '' else -1
        car_door_state = int(sample[6]) if len(sample) > 6 and sample[6] != '' else -1
        door_open_position = sample[7] if len(sample) > 7 and sample[7] != '' else []
        segment = sample[8] if len(sample) > 8 and sample[8] != '' else set()
        face_toward = sample[9] if len(sample) > 9 and sample[9] != '' else ''
        robotaxi = int(sample[10]) if len(sample) > 10 and sample[10] != '' else -1
        child = int(sample[11]) if len(sample) > 11 and sample[11] != '' else -1
        crossing = sample[12] if len(sample) > 12 and sample[12] != '' else ''
        stop_sign = int(sample[13]) if len(sample) > 13 and sample[13] != '' else -1
        
        # Main vehicle category mapping - 注意：proto enum已经调整为BACKGROUND=0开始
        # 根据utils.py中的GTClass和proto中的ImageCategory对应关系
        if gt_class == 1:  # CAR
            categories.append(1)  # CAR
        elif gt_class == 2:  # TRUCK
            categories.append(2)  # TRUCK
        elif gt_class == 3:  # VAN
            categories.append(3)  # VAN
        elif gt_class == 4:  # BUS
            categories.append(4)  # BUS
        elif gt_class == 13:  # CONSTRUCTION_VEHICLE
            categories.append(5)  # CONSTRUCTION_VEHICLE
        elif gt_class == 14:  # CAT
            categories.append(39)  # CAT
        elif gt_class == 9:  # TRAFFICCONE
            categories.append(41)  # TRAFFIC_CONE
        elif gt_class == 15:  # BARRIER_DELINEATOR
            categories.append(42)  # BARRIER_DELINERATOR
        elif gt_class == 0:  # BACKGROUND
            categories.append(0)  # BACKGROUND
        
        # Special vehicle categories - 所有值+1因为proto调整了
        if sp_vehicle >= 0:
            sp_mapping = {
                0: 6,   # SV_NONE
                1: 9,   # SV_POLICE_CAR
                3: 7,   # SV_FIRE_ENGINE
                4: 8,   # SV_AMBULANCE
                6: 13,  # SV_SCHOOL_BUS
                9: 5,   # SV_CONSTRUCTION_VEHICLE (使用主类别)
                10: 10, # SV_SPRINKLER
                13: 11, # SV_SPRINKLER_WO_BRUSH
                16: 12  # SV_SANITATION_VEHICLE
            }
            if sp_vehicle in sp_mapping:
                categories.append(sp_mapping[sp_vehicle])
        
        # Slope categories (for trucks)
        if gt_class == 2:  # TRUCK
            if sp_vehicle == 14:  # SV_TRUCK_WITH_SLOPE
                categories.append(14)  # WITH_SLOPE
            elif sp_vehicle == 15:  # SV_TRUCK_WITHOUT_SLOPE
                categories.append(15)  # WO_SLOPE
        
        # Door status categories - 所有值+1
        if car_door_state == 1 and door_open_position:  # door is open
            if isinstance(door_open_position, (list, set)):
                for position in door_open_position:
                    if 'back' in str(position):
                        categories.append(16)  # DOOR_STATUS_BACK
                    elif 'left' in str(position):
                        categories.append(17)  # DOOR_STATUS_LEFT
                    elif 'right' in str(position):
                        categories.append(18)  # DOOR_STATUS_RIGHT
                    elif 'front' in str(position):
                        categories.append(19)  # DOOR_STATUS_FRONT
        
        # Segment categories - 所有值+1
        if isinstance(segment, (set, str)):
            segment_str = str(segment)
            if 'head' in segment_str:
                categories.append(20)  # SEGMENT_HEAD
            if 'tail' in segment_str:
                categories.append(21)  # SEGMENT_TAIL
            if 'left' in segment_str:
                categories.append(22)  # SEGMENT_LEFT
            if 'right' in segment_str:
                categories.append(23)  # SEGMENT_RIGHT
        
        # Robotaxi categories - 所有值+1
        if robotaxi >= 0:
            if robotaxi == 0:
                categories.append(24)  # ROBOTAXI_NO
            elif robotaxi == 1:
                categories.append(25)  # ROBOTAXI_RT5
            elif robotaxi == 2:
                categories.append(26)  # ROBOTAXI_RT6
        
        # Stop sign categories - 所有值+1
        if sp_vehicle == 6 and stop_sign >= 0:  # SV_SCHOOL_BUS
            if stop_sign == 0:
                categories.append(27)  # STOP_SIGN_UNSEEN
            elif stop_sign == 1:
                categories.append(28)  # STOP_SIGN_ON
            elif stop_sign == 2:
                categories.append(29)  # STOP_SIGN_OFF
        
        # Profession categories - 所有值+1
        if gt_class == 8:  # PEDESTRIAN
            if profession == 0:
                categories.append(30)  # PROFESSION_NONE
            elif profession == 1:
                categories.append(31)  # PROFESSION_POLICE
            elif profession == 2:
                categories.append(32)  # PROFESSION_CONSTRUCTOR
        
        # Face toward categories - 所有值+1
        if face_toward:
            if face_toward == 'front':
                categories.append(33)  # FACE_TOWARD_FRONT
            else:
                categories.append(34)  # FACE_TOWARD_NO_FRONT
        
        # Child categories - 所有值+1
        if gt_class in [5, 6, 8]:  # CYCLIST, MOTO_CYCLIST, PEDESTRIAN
            if child >= 0:
                if child == 0:
                    categories.append(35)  # ADULT
                else:
                    categories.append(36)  # CHILD
        
        # Crossing categories - 所有值+1
        if crossing:
            if crossing == 'crossing':
                categories.append(37)  # CROSS
            else:
                categories.append(38)  # NO_CROSS
        
        # Remove duplicates
        categories = list(set(categories))
        
    except (ValueError, TypeError) as e:
        print(f"Warning: Error processing sample {sample[0] if len(sample) > 0 else 'unknown'}: {e}")
        categories = [40]  # UNDEFINED
    
    return categories

def convert_to_proto_format(data: np.ndarray, img_root: str) -> List[clean_subtype_pb2.ImageData]:
    """Convert npy data to protobuf ImageData objects using multi-label logic"""
    result = []
    
    for sample in data:
        # Create protobuf message
        image_data = clean_subtype_pb2.ImageData()
        
        image_id = sample[0] if len(sample) > 0 else ""
        image_data.id = image_id
        image_data.image_path = os.path.join(img_root, str(image_id) +'.jpg')
        
        # Get multiple category labels
        categories = get_category_labels_from_sample(sample)
        
        # Add all applicable categories
        for category in categories:
            image_data.category.append(category)
        
        # If no categories found, add UNDEFINED
        if not image_data.category:
            image_data.category.append(40)  # UNDEFINED
        
        # Features are left empty as requested
        # image_data.features.extend([])  # Already empty by default
        
        result.append(image_data)
    
    return result

def process_input_to_json(input_path: str, output_path: str, img_root: str) -> None:
    """Main processing function - output JSON format"""
    try:
        # Load all npy data
        npy_data_list = load_npy_files(input_path)
        
        # Process each npy file and combine results
        proto_messages = []
        for data in npy_data_list:
            proto_messages.extend(convert_to_proto_format(data, img_root))
        
        # Convert protobuf messages to JSON and write
        json_list = []
        for msg in proto_messages:
            json_str = MessageToJson(msg, preserving_proto_field_name=True)
            import json
            json_list.append(json.loads(json_str))
        
        with open(output_path, 'w') as f:
            json.dump(json_list, f, indent=2)
            
        print(f"Successfully converted {len(proto_messages)} items to {output_path}")
    
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise

def process_input_to_binary(input_path: str, output_path: str, img_root: str) -> None:
    """Main processing function - output binary format"""
    try:
        # Load all npy data
        npy_data_list = load_npy_files(input_path)
        
        # Process each npy file and combine results
        proto_messages = []
        for data in npy_data_list:
            proto_messages.extend(convert_to_proto_format(data, img_root))
        
        # Write protobuf binary format
        with open(output_path, 'wb') as f:
            for msg in proto_messages:
                f.write(msg.SerializeToString())
                
        print(f"Successfully converted {len(proto_messages)} items to {output_path} (binary format)")
    
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert npy files to protobuf format"
    )
    parser.add_argument(
        "input_path",
        help="Input path (can be .npy file, directory of .npy files, or .txt file with .npy paths)"
    )
    parser.add_argument(
        "output_file",
        help="Path to output file (.json for JSON format, .bin for binary format)"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Output in protobuf binary format instead of JSON"
    )
    parser.add_argument(
        "--img-root",
        type=str,
        required=True,
        help="Root directory for images"
    )

    args = parser.parse_args()
    
    if args.binary:
        process_input_to_binary(args.input_path, args.output_file, args.img_root)
    else:
        process_input_to_json(args.input_path, args.output_file, args.img_root)