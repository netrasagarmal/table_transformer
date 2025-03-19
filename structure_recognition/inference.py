import os
import logging
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base path loaded from environment variable (fallback to default if not set)
BASE_PATH = os.getenv('BASE_PATH', '/home/jovyan/work/Sagar/')

class LoadModels:
    """
    Class to load different AI models including YOLO and Table Transformer.
    """
    def __init__(self):
        logging.info("Initializing LoadModels class...")

    def load_yolo_model(self, model_path: str = None) -> YOLO:
        """
        Load YOLO model from the specified path.
        :param model_path: Path to the YOLO model weights file.
        :return: Loaded YOLO model.
        """
        if model_path is None:
            model_path = os.path.join(BASE_PATH, 'yolo_layout_model/yolov11x_best.pt')
        
        try:
            model = YOLO(model_path)
            logging.info("YOLO Model Loaded Successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise
    
    def load_table_model(self, model_name: str = None):
        """
        Load the Table Transformer model.
        :param model_name: Path to the pre-trained model directory.
        :return: Tuple containing the feature extractor and model.
        """
        if model_name is None:
            model_name = os.path.join(BASE_PATH, 'table_transformer_structure_recog_model')
        
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=model_name,
                use_pretrained_backbone=False,
                local_files_only=True,
                cache_dir=model_name
            )
            logging.info("Table Model Loaded Successfully.")
            return feature_extractor, model
        except Exception as e:
            logging.error(f"Error loading Table model: {e}")
            raise

# Example usage
if __name__ == "__main__":
    loader = LoadModels()
    yolo_model = loader.load_yolo_model()
    table_feature_extractor, table_model = loader.load_table_model()

import logging
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtractTable:
    def __init__(self, model_loader):
        """Initialize ExtractTable class with preloaded models."""
        try:
            self.feature_extractor, self.model = model_loader.load_table_model()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info("Table model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading table model: {e}")
            raise

    def crop_image(self, image: Image.Image, coordinates: List[int]) -> Image.Image:
        """Crops an image based on given coordinates."""
        if len(coordinates) != 4:
            raise ValueError("Coordinates list must contain exactly 4 values: [x1, y1, x2, y2]")
        return image.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))

    def file_to_base64(self, file_path: str) -> str:
        """Convert a file to a Base64 string."""
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Error encoding file to base64: {e}")
            raise

    def base64_to_file(self, base64_string: str) -> bytes:
        """Convert a Base64 string back to bytes."""
        return base64.b64decode(base64_string)

    def get_table_cells(self, results: Dict[str, Any], table_bbox_list: List[int]) -> Dict[str, Any]:
        """Extracts table cell coordinates from detection results."""
        try:
            rows, columns, headers = [], [], []
            table_bounds = None

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [int(i) for i in box.tolist()]
                label_name = self.model.config.id2label[label.item()]

                if label_name == "table":
                    table_bounds = box
                elif label_name == "table row":
                    rows.append(box)
                elif label_name == "table column":
                    columns.append(box)
                elif label_name == "table column header":
                    headers.append(box)

            rows.sort(key=lambda x: x[1])  # Sort rows by y-coordinate
            columns.sort(key=lambda x: x[0])  # Sort columns by x-coordinate

            cells = []
            width_x, height_y = table_bbox_list[:2]

            for i in range(len(rows)-1):
                for j in range(len(columns)-1):
                    cell = {
                        'row': i, 'col': j,
                        'coordinates': [columns[j][0] + width_x, rows[i][1] + height_y,
                                        columns[j+1][0] + width_x, rows[i+1][1] + height_y]
                    }
                    cells.append(cell)
            return {'cells': cells, 'table_bounds': table_bounds, 'rows': rows, 'columns': columns, 'headers': headers}
        except Exception as e:
            logging.error(f"Error extracting table cells: {e}")
            raise

    def extract_table(self, page_image: Image.Image, table_bbox_list: List[int], words_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts table structure and assigns words to their respective cells."""
        try:
            w, h = page_image.size
            cropped_image = self.crop_image(page_image, table_bbox_list)
            new_size = (table_bbox_list[2] - table_bbox_list[0], table_bbox_list[3] - table_bbox_list[1])

            inputs = self.feature_extractor(images=np.array(cropped_image.convert('RGB')), return_tensors="pt", size=new_size)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([cropped_image.size[::-1]]).to(self.device)
            results = self.feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

            table_data = self.get_table_cells(results, table_bbox_list)
            cells = [{'rowIndex': cell['row']+1, 'columnIndex': cell['col']+1, 'context': "", 'bbox': cell['coordinates']} for cell in table_data['cells']]

            box_map = {i: [] for i in range(len(table_data['cells']))}
            for word in words_list:
                word_bbox = word['bbox']
                for i, cell in enumerate(table_data['cells']):
                    if self.is_word_in_box(word_bbox, cell['coordinates']):
                        box_map[i].append(word['content'])
                        break

            for box_idx, words in box_map.items():
                cells[box_idx]['context'] = ' '.join(words)
            
            return cells
        except Exception as e:
            logging.error(f"Error extracting table: {e}")
            raise

    def is_word_in_box(self, word_bbox: List[int], box_bbox: List[int], threshold: float = 0.6) -> bool:
        """Check if a word lies at least `threshold`% inside the given box."""
        x1_w, y1_w, x2_w, y2_w = word_bbox
        x1_b, y1_b, x2_b, y2_b = box_bbox

        inter_x1, inter_y1 = max(x1_w, x1_b), max(y1_w, y1_b)
        inter_x2, inter_y2 = min(x2_w, x2_b), min(y2_w, y2_b)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            word_area = (x2_w - x1_w) * (y2_w - y1_w)
            return (inter_area / word_area) >= threshold
        return False

    






















###############################################################################################

import os
import io
from pdf2image import convert_from_path, convert_from_bytes
import json
from PIL import Image
import base64
import cv2
import supervision as sv # pip install supervision
from ultralytics import YOLO
import fnmatch
import torch
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import csv

BASE_PATH = '/home/jovyan/work/Sagar/'


class LoadModels:
    
    def __init__(self):
        pass
    
    def load_yolo_model(self,model_path = f'{BASE_PATH}yolo_layout_model/yolov11x_best.pt'):
        # Load YOLO Model
        model = YOLO(model_path)
        print("YOLO Model Loaded")
        return model
    
    def load_table_model(self , model_name = f'{BASE_PATH}table_transformer_structure_recog_model'):
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(pretrained_model_name_or_path = model_name,use_pretrained_backbone=False ,local_files_only=True,cache_dir=model_name)
        print("Table Model Loaded")
        return feature_extractor , model
        
           
        
class ExtractTable:
    
    def __init__(self):
        self.feature_extractor , self.model = model_loader.load_table_model()
    

    def crop_image(self,image_bytes: bytes, coordinates: list) -> Image.Image:
        """
        Crops an image around the given coordinates.

        :param image_bytes: The image in bytes format.
        :param coordinates: A list containing 4 coordinates [x1, y1, x2, y2].
        :return: Cropped PIL Image.
        """
        if len(coordinates) != 4:
            raise ValueError("Coordinates list must contain exactly 4 values: [x1, y1, x2, y2]")

        # Open the image from bytes
        # image = Image.open(io.BytesIO(image_bytes))
        image = image_bytes


        # Crop the image
        cropped_image = image.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))

        return cropped_image
    
    
    def file_to_base64(self,file_path):
        """Convert a file (PDF or image) to a Base64 string."""
        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("utf-8")
        return encoded_string
    
    
    def base64_to_file(self,base64_string):
        """Convert a Base64 string back to a file."""
        decoded_data = base64.b64decode(base64_string)
        return decoded_data
    
    
    def calculate_new_size_relative(self,coords: list, original_size: tuple) -> tuple:
        """
        Calculates new size of the cropped image relative to the original image.

        :param coords: List of 4 coordinates [x1, y1, x2, y2].
        :param original_size: Tuple of original image size (width, height).
        :param target_size: Tuple of target image size (width, height).
        :return: New (width, height) of the resized image.
        """
        orig_width, orig_height = original_size
        # target_width, target_height = target_size

        x1, y1, x2, y2 = coords
        cropped_width = x2 - x1
        cropped_height = y2 - y1

        # Scaling factors based on the original vs target size
        # width_scale = target_width / orig_width
        # height_scale = target_height / orig_height

        new_width = int(cropped_width * 1)
        new_height = int(cropped_height * 1)

        return (new_width, new_height)
    
    
    def get_table_cells(self,results, table_bbox_list):
        # Initialize lists for rows, columns and headers
        rows = []
        columns = []
        headers = []
        table_bounds = None

        # Sort detections into categories
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]  # Convert to integers
            label_name = self.model.config.id2label[label.item()]

            if label_name == "table":
                table_bounds = box
            elif label_name == "table row":
                rows.append(box)
            elif label_name == "table column":
                columns.append(box)
            elif label_name == "table column header":
                headers.append(box)

        # Sort rows and columns by their y and x coordinates respectively
        rows.sort(key=lambda x: x[1])  # Sort rows by y-coordinate
        columns.sort(key=lambda x: x[0])  # Sort columns by x-coordinate

        # Create cell matrix
        cells = []
        cell_coordinates = []

        #assiging x1 , y1 , x2 , y2
        # width_x = result["pages"][0]['tables'][1]['bbox'][0]
        # height_y = result["pages"][0]['tables'][1]['bbox'][1]
        width_x = table_bbox_list[0]
        height_y = table_bbox_list[1]

        # For each pair of consecutive rows
        for i in range(len(rows)-1):
            row_cells = []
            row_coords = []

            # For each pair of consecutive columns
            for j in range(len(columns)-1):
                # Cell coordinates
                cell = {
                    'x1': columns[j][0] + width_x,       # Left from current column
                    'y1': rows[i][1] + height_y,          # Top from current row
                    'x2': columns[j+1][0] + width_x,     # Right from next column
                    'y2': rows[i+1][1] + height_y,        # Bottom from next row
                    'row': i,
                    'col': j,
                    'coordinates': [
                        columns[j][0] + width_x,         # x1
                        rows[i][1] + height_y,           # y1
                        columns[j+1][0] + width_x,      # x2
                        rows[i+1][1] + height_y          # y2
                    ]
                }
                row_cells.append(f"r{i+1}c{j+1}")
                row_coords.append(cell)

            cells.append(row_cells)
            cell_coordinates.append(row_coords)

        return {
            'matrix': cells,
            'coordinates': cell_coordinates,
            'table_bounds': table_bounds,
            'rows': rows,
            'columns': columns,
            'headers': headers
        }
    
    
    
    def adjust_coordinates(self,word_bbox, orig_width, orig_height, table_bbox):
        """Adjust word coordinates relative to the cropped table."""
        x1_w, y1_w, x2_w, y2_w = word_bbox
        x1_t, y1_t, x2_t, y2_t = table_bbox

        scale_x = (x2_t - x1_t) / orig_width
        scale_y = (y2_t - y1_t) / orig_height

        new_x1 = (x1_w - x1_t) * scale_x
        new_y1 = (y1_w - y1_t) * scale_y
        new_x2 = (x2_w - x1_t) * scale_x
        new_y2 = (y2_w - y1_t) * scale_y

        # new_x1 = (x1_w - x1_t)
        # new_y1 = (y1_w - y1_t)
        # new_x2 = (x2_w - x1_t)
        # new_y2 = (y2_w - y1_t)

        return [new_x1, new_y1, new_x2, new_y2]


    def is_word_in_box(self,word_bbox, box_bbox, threshold=0.6):
        """Check if a word lies at least `threshold`% inside the given box."""
        x1_w, y1_w, x2_w, y2_w = word_bbox
        x1_b, y1_b, x2_b, y2_b = box_bbox

        # Calculate intersection area
        inter_x1 = max(x1_w, x1_b)
        inter_y1 = max(y1_w, y1_b)
        inter_x2 = min(x2_w, x2_b)
        inter_y2 = min(y2_w, y2_b)

    #     if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
    #         return False  # No overlap

    #     inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    #     word_area = (x2_w - x1_w) * (y2_w - y1_w)

    #     return (inter_area / word_area) >= threshold

            # Check if there is an intersection
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            word_area = (x2_w - x1_w) * (y2_w - y1_w)
            overlap_ratio = inter_area / word_area

            return overlap_ratio >= threshold

        return False


    def group_words_into_boxes(self,words, boxes, orig_width, orig_height, table_bbox, threshold=0.6):
        """Assign words to their respective bounding boxes based on the given threshold, adjusting for cropped tables."""
        box_map = {i: [] for i in range(len(boxes))}

        for word in words:
            # word_bbox = adjust_coordinates(word["bbox"], orig_width, orig_height, table_bbox)
            word_bbox = word["bbox"]
            assigned = False

            for i, box in enumerate(boxes):
                if self.is_word_in_box(word_bbox, box, threshold):
                    box_map[i].append(word["content"])
                    assigned = True
                    break

        return box_map
    
    
    def drawing_printing(self,image,results):
        
        image1 = image
        # Visualization
        draw = ImageDraw.Draw(image1)
        font = ImageFont.load_default()

        # Color mapping for different table structure elements
        color_map = {
            0: "red",    # table
            1: "green",  # table column
            2: "blue",   # table row
            3: "purple", # table column header
            4: "orange"  # table body
        }

        # Draw bounding boxes
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Convert to integer coordinates
            box = [int(b) for b in box.tolist()]

            # Get label name (optional)
            label_name = self.model.config.id2label[label.item()]

            # Choose color based on label
            draw_color = color_map.get(label.item(), "yellow")

            # Draw bounding box
            draw.rectangle(box, outline=draw_color, width=2)

            # Draw label with score
            label_text = f"{label_name}: {score.item():.2f}"
            draw.text((box[0], max(0, box[1]-10)), label_text, fill=draw_color, font=font)

        # Save the annotated image
        output_path = "/home/jovyan/work/Sagar/table-transformer-playground/structure-recog/r1/test1.png"
        image1.save(output_path)

        # Print detected elements
        # print("Detected Table Structures:")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            # print(f"{label_name}: Score {score.item():.2f}, Box {box.tolist()}")

        # print("done")
    
    
    def extract_table(self , page_image , table_bbox_list , words_list):
        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to GPU
        self.model.to(device)
        
        # Convert file to Base64 string
        # base64_str = self.file_to_base64(input_file)
        
        # decoded_data = self.base64_to_file(base64_str)

        #for pdf
        # pages = convert_from_bytes(decoded_data)

        #for image
        # image = Image.open(io.BytesIO(decoded_data))
        pages = [page_image]
        
        w, h = pages[0].size
        # print(w,h)
        
        
        #get cropped image
        # image = self.crop_image(image_bytes = pages[0], coordinates = list(result["pages"][0]['tables'][1]['bbox']))
        image = self.crop_image(image_bytes = pages[0], coordinates = table_bbox_list)
        
        # new_size = self.calculate_new_size_relative(coords=list(result["pages"][0]['tables'][0]['bbox']), original_size=(w,h))
        new_size = self.calculate_new_size_relative(coords= table_bbox_list, original_size=(w,h))
        # print(new_size)
        
        # Convert image to RGB if it's not already
        image1 = image.convert('RGB')

        image_np = np.array(image1)
        
        '''# Prepare inputs
        inputs = self.feature_extractor(images=image_np, return_tensors="pt", do_resize=True, size={"height": new_size[1], "width": new_size[0]})

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process the results
        target_sizes = torch.tensor([image1.size[::-1]])
        results = self.feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]'''
        
        # Prepare inputs and move them to GPU
        inputs = self.feature_extractor(images=image_np, return_tensors="pt", do_resize=True, size={"height": new_size[1], "width": new_size[0]})
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move all input tensors to GPU

        # Run inference on GPU
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the results
        target_sizes = torch.tensor([image1.size[::-1]]).to(device)  # Move target_sizes to GPU
        results = self.feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        # self.drawing_printing(image1 , results)
        
        
        # Load the image
        image_path = "/home/jovyan/work/Sagar/table-transformer-playground/structure-recog/r1/test1.png"
        # image = Image.open(image)

        # Convert image to RGB if it's not already
        image2 = image.convert('RGB')
        
        # Use the function with your results
        table_data = self.get_table_cells(results , table_bbox_list)
        
        
        # print(table_data)

        # Print the cell matrix
        # print("\nTable Cell Matrix:")
        # for row in table_data['matrix']:
        #     print(row)
        cells = []
        crds = []
        # Print detailed cell coordinates
        # print("\nDetailed Cell Coordinates:")
        for i, row in enumerate(table_data['coordinates']):
            for cell in row:
                # print(f"Cell {cell['row']+1},{cell['col']+1}: {cell['coordinates']}")
                crds.append(cell['coordinates'])
                
                cell_info = {
                  "rowIndex":cell['row']+1,
                  "columnIndex":cell['col']+1,
                  "context":"",
                  "bbox":cell['coordinates']
                }
                cells.append(cell_info)

        '''# Visualize the cells on the image
        draw = ImageDraw.Draw(image2)
        font = ImageFont.load_default()

        # Draw cells
        for row in table_data['coordinates']:
            for cell in row:
                # Draw cell rectangle
                draw.rectangle([
                    (cell['x1'], cell['y1']),
                    (cell['x2'], cell['y2'])
                ], outline="green", width=5)

                # Draw cell label
                label = f"r{cell['row']+1}c{cell['col']+1}"
                draw.text(
                    (cell['x1'] + 5, cell['y1'] + 5),
                    label,
                    fill="blue",
                    font=font
                )

        # Save the annotated image
        output_path = "/home/jovyan/work/Sagar/table-transformer-playground/structure-recog/r1/test2.png"
        image2.save(output_path)'''
        
        # Example usage:
        # words = result["pages"][0]["words"]
        words = words_list

        # table_bbox = list(result["pages"][0]['tables'][0]['bbox'])  # Table location in the original image
        table_bbox = table_bbox_list
        orig_width, orig_height = w,h  # Original image dimensions

        boxes = crds

        grouped_words = self.group_words_into_boxes(words, boxes, orig_width, orig_height, table_bbox, threshold=.6)
        for box_idx, words in grouped_words.items():
            # print(f"Box {box_idx}: {' '.join(words)}")
            cells[box_idx]["context"] = ' '.join(words)
            
        return cells

    
    
class AnalyseFileLayout:
    def __init__(self, model):
        self.model = model
        self.extractor = ExtractTable()

    def file_to_base64(self, file_path):
        """Convert a file (PDF or image) to a Base64 string."""
        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("utf-8")
        return encoded_string
    
    def base64_to_file(self, base64_string):
        """Convert a Base64 string back to a file."""
        decoded_data = base64.b64decode(base64_string)
        return decoded_data
    
    
    def crop_image(image_bytes: bytes, coordinates: list) -> Image.Image:
        """
        Crops an image around the given coordinates.

        :param image_bytes: The image in bytes format.
        :param coordinates: A list containing 4 coordinates [x1, y1, x2, y2].
        :return: Cropped PIL Image.
        """
        if len(coordinates) != 4:
            raise ValueError("Coordinates list must contain exactly 4 values: [x1, y1, x2, y2]")

        # Open the image from bytes
        # image = Image.open(io.BytesIO(image_bytes))
        image = image_bytes


        # Crop the image
        cropped_image = image.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))

        return cropped_image

    def get_model_layout_analysis(self, image, get_annotated_image = False):
    
        annotated_image = None
    
        results = self.model(source=image, conf=0.2, iou=0.8)[0]
        detections = sv.Detections.from_ultralytics(results)
    
        boxes = detections.xyxy.tolist()
        boxes = [[round(x,3) for x in box] for box in boxes]
        # boxes = sort_bounding_box_top_to_bottom(bbox_list = boxes)
        confidence = detections.confidence.tolist()
        class_id = detections.class_id.tolist()
        class_labels = detections.data['class_name'].tolist()
    
        # cleaned_detections = [boxes, confidence, class_id, class_labels]
    
        cleaned_detections = {
            "bboxes":boxes,
            "confidence":confidence,
            "class_id":class_id,
            "class_labels":class_labels
        }
        
    
        if get_annotated_image:
            detections = sv.Detections.from_ultralytics(results)
    
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)
    
        return cleaned_detections, annotated_image

    def adjust_bounding_boxes(self, bboxes, n_percent= None, value = None):
        """
        Expands bounding boxes by n percent.
        
        Parameters:
        - bboxes: List of bounding boxes [(x_min, y_min, x_max, y_max)]
        - n_percent: Percentage to expand (e.g., 10 for 10%)
    
        Returns:
        - List of adjusted bounding boxes [(new_x_min, new_y_min, new_x_max, new_y_max)]
        """
        adjusted_bboxes = []
    
        for x_min, y_min, x_max, y_max in bboxes:
            
            
            if n_percent is not None:
                width = x_max - x_min
                height = y_max - y_min
            
                # Calculate expansion values
                expand_x = (width * n_percent) / 100
                expand_y = (height * n_percent) / 100
            else:
                # Calculate expansion values
                expand_x = value
                expand_y = value
    
            # Adjust coordinates
            new_x_min = x_min - expand_x
            new_y_min = y_min - expand_y
            new_x_max = x_max + expand_x
            new_y_max = y_max + expand_y
    
            adjusted_bboxes.append((new_x_min, new_y_min, new_x_max, new_y_max))
    
        return adjusted_bboxes
    
    def inch_to_pixel(self, word_polygon, width_dpi, height_dpi):
        
        new_bbox = [val * (width_dpi if i % 2 == 0 else height_dpi) for i, val in enumerate(word_polygon)]
    
        new_bbox = [round(val,3) for val in new_bbox]
    
        return new_bbox

    def class_bbox_mapping(self, page_detections):
        mappings = {}
    
        for i, class_label in enumerate(page_detections["class_labels"]):
    
            if class_label not in mappings:
                mappings[str(class_label)] = []
                mappings[str(class_label)].append((page_detections["bboxes"][i], 
                                              page_detections["class_id"][i], 
                                              page_detections["confidence"][i])
                                            )
            else:
                mappings[str(class_label)].append((page_detections["bboxes"][i], 
                                              page_detections["class_id"][i], 
                                              page_detections["confidence"][i])
                                            )
            def sort_bounding_box_top_to_bottom(bbox_list):
                d = {}
                for box in bbox_list:
                    d[box[0][1]] = box
            
                keys_lst = list(d.keys())
                keys_lst.sort()
                new_bbox_list = [d[x] for x in keys_lst]
            
                return new_bbox_list
            
            for key in mappings:
                mappings[key] = sort_bounding_box_top_to_bottom(bbox_list = mappings[key])

        return mappings
        
    def is_inside_bbox(self, word_polygon, bbox):
        """
        Check if the word's bounding box (polygon) is inside the given YOLO bounding box (bbox).
        word_polygon: List of 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
        bbox: YOLO bounding box (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = bbox
        # print(x_min, y_min, x_max, y_max)
        word_x_min = min(word_polygon[0], word_polygon[2], word_polygon[4], word_polygon[6])
        word_y_min = min(word_polygon[1], word_polygon[3], word_polygon[5], word_polygon[7])
        word_x_max = max(word_polygon[0], word_polygon[2], word_polygon[4], word_polygon[6])
        word_y_max = max(word_polygon[1], word_polygon[3], word_polygon[5], word_polygon[7])
        
        return (word_x_min >= x_min and word_x_max <= x_max and 
                word_y_min >= y_min and word_y_max <= y_max)
    
    def get_full_page_paragraphs(self, yolo_detections, page_ocr, width_ppi = None, height_ppi = None):
        """
        Groups OCR words into paragraphs based on YOLO bounding boxes.
        yolo_detections: List of bounding boxes [(x_min, y_min, x_max, y_max)]
        ocr_data: OCR words with their polygon positions
        """
        paragraphs = []
        paragraph_words = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
        
            paragraph_words = []
    
            unit = page_ocr["unit"]
            
            for word in page_ocr["words"]:
                word_polygon = word["polygon"]
                
                if unit == "inch" and width_ppi is not None and height_ppi is not None:
                    # word_polygon = [val * (width_dpi if i % 2 == 0 else height_dpi) for i, val in enumerate(word_polygon)]
                    word_polygon = self.inch_to_pixel(word_polygon, width_ppi, height_ppi)
                
                if self.is_inside_bbox(word_polygon, bbox):
                    paragraph_words.append((word_polygon, word["content"]))  # Store x-coordinate for sorting
    
            if paragraph_words:
                x1 = paragraph_words[0][0][0]
                y1 = paragraph_words[0][0][1]
                x2 = paragraph_words[-1][0][3]
                y2 = paragraph_words[-1][0][4]
        
                bbox = [x1,y1,x2,y2]
                
                # Extract only the words
                paragraph_text = " ".join([w[1] for w in paragraph_words])
        
        
                d = {
                    "content":paragraph_text,
                    "bbox": bbox,
                    "confidence": confidence
                }
                
                if paragraph_text:
                    paragraphs.append(d)
    
        return paragraphs
    
    
    def get_page_footers(self, yolo_detections, page_ocr, width_ppi = None, height_ppi = None):
        """
        Groups OCR words into paragraphs based on YOLO bounding boxes.
        yolo_detections: List of bounding boxes [(x_min, y_min, x_max, y_max)]
        ocr_data: OCR words with their polygon positions
        """
        footers = []
        footer_words = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
        
            footer_words = []
    
            unit = page_ocr["unit"]
            
            for word in page_ocr["words"]:
                word_polygon = word["polygon"]
                
                if unit == "inch" and width_ppi is not None and height_ppi is not None:
                    # word_polygon = [val * (width_dpi if i % 2 == 0 else height_dpi) for i, val in enumerate(word_polygon)]
                    word_polygon = self.inch_to_pixel(word_polygon, width_ppi, height_ppi)
                
                if self.is_inside_bbox(word_polygon, bbox):
                    footer_words.append((word_polygon, word["content"]))  # Store x-coordinate for sorting
    
            if footer_words:
                x1 = footer_words[0][0][0]
                y1 = footer_words[0][0][1]
                x2 = footer_words[-1][0][0]
                y2 = footer_words[-1][0][1]
        
                bbox = [x1,y1,x2,y2]
                
                # Extract only the words
                footer_text = " ".join([w[1] for w in footer_words])
        
        
                d = {}
                d = {
                    "content":footer_text,
                    "bbox": bbox,
                    "confidence": confidence
                }
                
                if footer_text:
                    footers.append(d)
    
        return footers
    
    
    def get_page_headers(self, yolo_detections, page_ocr, width_ppi = None, height_ppi = None):
        """
        Groups OCR words into paragraphs based on YOLO bounding boxes.
        yolo_detections: List of bounding boxes [(x_min, y_min, x_max, y_max)]
        ocr_data: OCR words with their polygon positions
        """
        headers = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
        
            header_words = []
    
            unit = page_ocr["unit"]
            
            for word in page_ocr["words"]:
                word_polygon = word["polygon"]
                
                if unit == "inch" and width_ppi is not None and height_ppi is not None:
                    # word_polygon = [val * (width_dpi if i % 2 == 0 else height_dpi) for i, val in enumerate(word_polygon)]
                    word_polygon = self.inch_to_pixel(word_polygon, width_ppi, height_ppi)
                
                if self.is_inside_bbox(word_polygon, bbox):
                    header_words.append((word_polygon, word["content"]))  # Store x-coordinate for sorting
    
            if header_words:
                x1 = header_words[0][0][0]
                y1 = header_words[0][0][1]
                x2 = header_words[-1][0][0]
                y2 = header_words[-1][0][1]
        
                bbox = [x1,y1,x2,y2]
    
                header_text = " ".join([w[1] for w in header_words])
        
                d = {}
                d = {
                    "content":header_text,
                    "bbox": bbox,
                    "confidence": confidence
                }
                
                if header_text:
                    headers.append(d)
    
        return headers
    
    def get_page_section_headers(self, yolo_detections, page_ocr, width_ppi = None, height_ppi = None):
        """
        Groups OCR words into paragraphs based on YOLO bounding boxes.
        yolo_detections: List of bounding boxes [(x_min, y_min, x_max, y_max)]
        ocr_data: OCR words with their polygon positions
        """
        headers = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
        
            section_header_words = []
    
            unit = page_ocr["unit"]
            
            for word in page_ocr["words"]:
                
                word_polygon = word["polygon"]
                
                if unit == "inch" and width_ppi is not None and height_ppi is not None:
                    # word_polygon = [val * (width_dpi if i % 2 == 0 else height_dpi) for i, val in enumerate(word_polygon)]
                    word_polygon = self.inch_to_pixel(word_polygon, width_ppi, height_ppi)
                
                if self.is_inside_bbox(word_polygon, bbox):
                    section_header_words.append((word_polygon, word["content"]))  # Store x-coordinate for sorting
    
            if section_header_words:
                x1 = section_header_words[0][0][0] if section_header_words else 0
                y1 = section_header_words[0][0][1]
                x2 = section_header_words[-1][0][2]
                y2 = section_header_words[-1][0][3]
        
                bbox = [x1,y1,x2,y2]
                
                # Extract only the words
                section_header_text = " ".join([w[1] for w in section_header_words])
                d = {}
                d = {
                    "content":section_header_text,
                    "bbox": bbox,
                    "confidence": confidence
                }
                
                if section_header_text:
                    headers.append(d)
    
        return headers
        
    
    def get_full_page_text(self, page_ocr):
    
        page_text = ""
        for word in page_ocr["words"]:
            page_text = " \n ".join([w["content"] for w in page_ocr["lines"]])

        return page_text
    
    def get_full_page_words(self, page_ocr, width_ppi = None, height_ppi = None):
    
        page_words = []
        unit = page_ocr["unit"]
        for word in page_ocr["words"]:
            content = word["content"]
            

            
            line_polygon = word["polygon"]
            if unit == "inch" and width_ppi is not None and height_ppi is not None:
                line_polygon = self.inch_to_pixel(word["polygon"], width_ppi, height_ppi)
            confidence = word["confidence"]
            if len(line_polygon) == 8:
                line_polygon = [line_polygon[0],line_polygon[1],line_polygon[4],line_polygon[5]]
            d = {
                "content":content,
                "bbox": line_polygon,
                "confidence":confidence
            }
            page_words.append(d)
            
        return page_words
    
    def get_full_page_lines(self, page_ocr, width_ppi = None, height_ppi = None):
    
        page_lines = []
        
        
        unit = page_ocr["unit"]
        
        for line in page_ocr["lines"]:
            content = line["content"]
            line_polygon = line["polygon"]
            if unit == "inch" and width_ppi is not None and height_ppi is not None:
                line_polygon = self.inch_to_pixel(line["polygon"], width_ppi, height_ppi)
            if len(line_polygon) == 8:
                line_polygon = [line_polygon[0],line_polygon[1],line_polygon[4],line_polygon[5]]
            d = {
                "content":content,
                "bbox": line_polygon
            }
            page_lines.append(d)
            
    
        # print(page_lines)
    
        return page_lines
            
    
    def get_page_tables(self, yolo_detections , page_words , page_image):
    
        page_tables = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
            
            start_time = time.time()
            cells = self.extractor.extract_table(page_image , bbox , page_words)
            end_time = time.time()
            
            d = {
                "cells":cells,
                "bbox":bbox,
                "confidence":confidence,
                "time_taken":end_time-start_time
            }
            
            # print("for loop , get_page_table",d)
            
            page_tables.append(d)
            
        # print("Inside get_page_tables", page_tables)
    
        return page_tables
    
    def get_page_figures(self, yolo_detections):
    
        page_figures = []
    
        for i, (bbox, id, confidence) in enumerate(yolo_detections):
            d = {
                "bbox":bbox,
                "confidence":confidence
            }
            page_figures.append(d)
    
        return page_figures
    
    def analyze_page_layout(self, input_file, page, page_ocr, page_number):

        page_width_pixels, page_height_pixels = page.size
    
        # print(page_width_pixels, page_height_pixels)

        # print("page keys",page_ocr.keys())
    
        width_ppi = 0
        height_ppi = 0
    
        page_paragraph = []
        page_tables = []
        page_figures = []
        page_footers = []
        page_headers = []
        page_section_headers = []
    
        if page_ocr["unit"] == "inch":
            page_width_inch = page_ocr["width"]
            page_height_inch = page_ocr["height"]
    
    
            width_ppi = round(page_width_pixels / page_width_inch,3)
            height_ppi = round(page_height_pixels / page_height_inch,3)

        # else:
            # width_ppi = round(page_width_pixels / page_width_inch,3)
            # height_ppi = round(page_height_pixels / page_height_inch,3)
        
        start_time = time.time()
        page_detections, annotated_image = self.get_model_layout_analysis(image = page, get_annotated_image = False)
        end_time = time.time()
        # print(page_detections)
    
        page_detections["bboxes"] = self.adjust_bounding_boxes(page_detections["bboxes"], value = 20)
    
        mappings = self.class_bbox_mapping(page_detections = page_detections)
        
        # print(mappings)
    
        page_text = self.get_full_page_text(page_ocr = page_ocr)
    
        page_lines = self.get_full_page_lines(page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)
        
        page_words = self.get_full_page_words(page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)
    
        if "Text" in mappings:
            page_paragraph = self.get_full_page_paragraphs(yolo_detections = mappings["Text"], page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)
    
        if "Table" in mappings:
            page_tables = self.get_page_tables(yolo_detections = mappings["Table"] , page_words = page_words , page_image = page)
    
        if "Picture" in mappings:
            page_figures = self.get_page_figures(yolo_detections = mappings["Picture"])
    
        if "Page-header" in mappings:
            page_headers = self.get_page_headers(yolo_detections = mappings["Page-header"], page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)
    
        if "Page-footer" in mappings:
            page_footers = self.get_page_footers(yolo_detections = mappings["Page-footer"], page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)
    
        if "Section-header" in mappings:
            page_section_headers = self.get_page_section_headers(yolo_detections = mappings["Section-header"], page_ocr = page_ocr, width_ppi = width_ppi, height_ppi = height_ppi)

    
        d = {
            "page_number":page_number,
            "page_width":page_width_pixels,
            "page_height":page_height_pixels,
            "dimension_unit":"pixels",
            "content":page_text,
            "words":page_words,
            "lines":page_lines,
            "paragraphs":page_paragraph,
            "section_headers":page_section_headers,
            "page_headers":page_headers,
            "page_footers":page_footers,
            "tables":page_tables,
            "figure":page_figures,
            "yolo_model_time":end_time-start_time
        }
    
        return d
    
    def analyze_file_layout(self, input_file, file_ocr):
        
        base64_str = self.file_to_base64(input_file)
        decoded_data = self.base64_to_file(base64_str)

        pages = []

        pdf_initials = b'%PDF-'
        png_initials = b'\x89PNG\r\n\x1a\n'
        jpeg_initials = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        
        file_type = None
        if decoded_data.startswith(pdf_initials):
            file_type = "PDF"
            pages = convert_from_bytes(decoded_data)
            
        elif decoded_data.startswith(png_initials):
            file_type = "PNG"
            image_buffer = io.BytesIO(decoded_data)
            
            # Open the image using PIL
            image = Image.open(image_buffer)
            pages.append(image)
            
        elif decoded_data.startswith(jpeg_initials):
            file_type = "JPEG"
            image_buffer = io.BytesIO(decoded_data)
            
            # Open the image using PIL
            image = Image.open(image_buffer)
            pages.append(image)
        else:
            file_type = "Invalid"
    
        extracted_pages = []
        
        for i in range(len(pages)):
    
            # print("\n\npage: ", i+1)
    
            page_layout = self.analyze_page_layout(input_file = input_file, page = pages[i], page_ocr = file_ocr[i], page_number = i+1 )
            extracted_pages.append(page_layout)
        
        d = {
            "total_pages":len(pages),
            "file_type":file_type,
            "pages":extracted_pages
            
        }
    
        return d
            
        
def read_json_file(file_path):
    with open(file_path, "r") as f:
        data_file = json.load(f)
        f.close()
    return data_file



def process_files(input_directory, output_directory, azure_ocr, log_file, analyse_layout):
    """
    Process image and PDF files in the input directory and save each result as a JSON file in the output directory.

    Args:
        input_directory (str): The path to the directory containing files to process.
        output_directory (str): The path to the directory where JSON files will be saved.
    """
    # Define patterns for image and PDF files
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    pdf_pattern = '*.pdf'

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        
        ocr_filename = f'{os.path.splitext(filename)[0]}.json'
        ocr_file_path = os.path.join(ocr_directory, ocr_filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Check for image files
            if any(fnmatch.fnmatch(filename.lower(), pattern) for pattern in image_patterns):
                # Process the image file
                start_time = time.time()
                ocr_data = read_json_file(ocr_file_path)
                processed_data = analyse_layout.analyze_file_layout(input_file = file_path, file_ocr = ocr_data)
                end_time = time.time()
            # Check for PDF files
            elif fnmatch.fnmatch(filename.lower(), pdf_pattern):
                # Process the PDF file
                start_time = time.time()
                ocr_data = read_json_file(ocr_file_path)
                processed_data = analyse_layout.analyze_file_layout(input_file = file_path, file_ocr = ocr_data)
                end_time = time.time()
            else:
                # Skip files that are neither images nor PDFs
                continue
                
                
                
            time_taken = end_time - start_time
            
            # Log the details to the CSV file
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename,
                                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
                                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                                 f"{time_taken:.6f}"])

            # Define the output JSON file path
            json_filename = f'{os.path.splitext(filename)[0]}.json'
            json_path = os.path.join(output_directory, json_filename)

            # Write the processed data to the JSON file
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(processed_data, json_file, indent=4)
            print(f'Saved processed data to {json_path}')
            
            
            
# Define the input and output directory paths
input_directory = f'{BASE_PATH}table-transformer-playground/final_pipline_table/input_files'
output_directory = f'{BASE_PATH}table-transformer-playground/final_pipline_table/output_files'
ocr_directory = f'{BASE_PATH}table-transformer-playground/final_pipline_table/ocr_files'

model_loader = LoadModels()
#Load YOLO model
detection_model = model_loader.load_yolo_model()
analyse_layout = AnalyseFileLayout(model = detection_model)

log_file = f'{BASE_PATH}table-transformer-playground/final_pipline_table/log_file.csv'
# Write header to the CSV log file if it doesn't exist
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Start Time', 'End Time', 'Time Taken (seconds)'])
    
    
if __name__ == "__main__":
    process_files(input_directory, output_directory , ocr_directory, log_file, analyse_layout)

    
    