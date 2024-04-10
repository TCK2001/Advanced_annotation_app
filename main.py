import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import os
import random
from PIL import Image, ImageTk

import torch
from ultralytics import YOLO

class ObjectDetectionAnnotationGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Object Detection Annotation")
        self.master.geometry("1200x800")
        
        self.predicted_image = None  # Store the predicted image
        
        self.categories = ["background"]  # Add your categories here
        self.models = ["yolo v8"]
        
        self.load_image_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_image_button.pack(anchor=tk.NW, side=tk.LEFT, padx=10, pady=10)

        self.image_path_frame = tk.Frame(self.master, width=200, height=50, bd=2, relief=tk.SOLID)
        self.image_path_frame.place(x=700, y=100)
        
        self.image_path_label = tk.Label(self.master, text="Current Image Path: None")
        self.image_path_label.pack(anchor=tk.NW, side=tk.LEFT, padx=10, pady=10)
        
        self.current_image_coordinate_label = tk.Label(self.image_path_frame, text="Current Image Path: None")
        self.current_image_coordinate_label.pack(anchor=tk.W, side=tk.LEFT, pady=10)

        # Left side: Image display area
        self.image_frame = tk.Frame(self.master, width=600, height=600, bg="white", bd=2, relief=tk.SOLID)
        self.image_frame.place(x=50, y=100)

        # Right side: Tool panel
        self.tool_panel = tk.Frame(self.master, width=200, height=800, bg="green", bd=2, relief=tk.SOLID)
        self.tool_panel.place(x=1000, y=100)

        self.model_var = tk.StringVar()
        self.model_var.set(self.models[0])
        
        self.model_menu = tk.OptionMenu(self.tool_panel, self.model_var, *self.models)
        self.model_menu.pack(anchor=tk.N, padx=10, pady=10)
        
        self.predict_button = ttk.Button(self.tool_panel, text="Predict", command=self.predict)
        self.predict_button.pack(anchor=tk.N, padx=10, pady=10)
        
        self.draw_button = tk.Button(self.tool_panel, text="Draw", command=self.enable_drawing)
        self.draw_button.pack(anchor=tk.N, padx=10, pady=10)
        
        style = ttk.Style()
        style.map("TButton", foreground=[('disabled', 'black')], background=[('disabled', 'black')])
        
        self.undo_button = ttk.Button(self.tool_panel, text="<-", command=self.undo)
        self.undo_button.pack(anchor=tk.N, padx=10)
        self.undo_button.config(state="disabled")
        
        self.redo_button = ttk.Button(self.tool_panel, text="->", command=self.redo)
        self.redo_button.pack(anchor=tk.N, padx=10)
        self.redo_button.config(state="disabled")
        
        self.category_var = tk.StringVar()
        self.category_var.set(self.categories[0])
        
        self.category_entry = tk.Entry(self.tool_panel, textvariable=self.category_var)
        self.category_entry.pack(anchor=tk.N, padx=10, pady=10)
        
        self.confirm_category_button = tk.Button(self.tool_panel, text="Add catrgory", command=self.confirm_category)
        self.confirm_category_button.pack(anchor=tk.N, padx=10, pady=10)
        
        self.category_menu = tk.OptionMenu(self.tool_panel, self.category_var, *self.categories)
        self.category_menu.pack(anchor=tk.N, padx=10, pady=10)

        # Button to add model
        self.add_model_button = tk.Button(self.tool_panel, text="Custom Weight", command=self.add_model)
        self.add_model_button.pack(anchor=tk.N, padx=10, pady=10)
        self.weights_path = None 
        
        self.canvas = None
        self.drawing = False
        self.photo = None  # Store PhotoImage object as an instance variable
        self.bbox_start = None  # Store the starting point of the bounding box
        self.bbox_end = None  # Store the ending point of the bounding box
        self.image_path = None
        
        self.undo = 1
        self.redo = 1
        
        # Dictionary to store the assigned category for each bounding box
        self.current_bboxes = [] 
        self.draw_boxes = []  # Initialize current_bbox attribute
        self.deleted_bboxes = []  # Store deleted bounding boxes for restoration
        self.category_colors = {"background": "red"}
        self.draw_btn_isclick = False
        self.draw_boxes_coords = {}  # Dictionary to store coordinates of drawn boxes
        
    def add_model(self):
        # Load weights
        self.weights_path = filedialog.askopenfilename(filetypes=[("Weight files", "*.pt")])
    
        # Update the model selection menu
        weight_name = os.path.basename(self.weights_path).split('.')[0]
        self.models.append(weight_name)
        self.model_menu.destroy()
        self.model_menu = tk.OptionMenu(self.tool_panel, self.model_var, *self.models)
        self.model_menu.pack(anchor=tk.N, padx=10, pady=10)
        
    def predict(self):
        self.undo = 0
        self.redo = 0
        if self.photo is None:
            return
        image = Image.open(self.image_path)
        image = image.resize((600, 600))  # Resize the image
        if self.weights_path  != None :
            weight_name = self.model_var.get()
            check_now_select_weight = os.path.basename(self.weights_path).split('.')[0]
            if weight_name == "yolo v8":
                self.model = YOLO("yolov8n.pt")
            elif weight_name == check_now_select_weight:
                self.model = YOLO(self.weights_path)
            else :
                self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO("yolov8n.pt")
            
        # Perform prediction using the YOLOv8 model
        results = self.model(image)
        
        # Add the predicted bounding boxes to the current_bboxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                
                # Extract box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0]
                c = box.cls
                
                bbox_color = self.get_category_color(self.model.names[int(c)])
                if isinstance(x1, torch.Tensor) and isinstance(y1, torch.Tensor):
                    x1 = int(x1.tolist())  
                    y1 = int(y1.tolist())  
                    x2 = int(x2.tolist())  
                    y2 = int(y2.tolist()) 
                    
                self.current_bboxes.append([[x1, y1], [x2, y2], [bbox_color]])
                
                self.current_bboxes_temp = self.current_bboxes
                self.current_bboxes = []
                for inner in self.current_bboxes_temp:
                    inner_tuple = tuple(tuple(x) for x in inner)
                    if inner_tuple not in self.current_bboxes:
                        self.current_bboxes.append(inner_tuple)
                self.current_bboxes = [list(inner) for inner in self.current_bboxes]

        self.draw_bounding_boxes()
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
    def confirm_category(self):
        new_category = self.category_var.get()
        if new_category not in self.categories:
            self.category_colors[new_category] = self.get_category_color(new_category)
            self.category_menu.destroy()  # Destroy the old menu
            self.category_menu = tk.OptionMenu(self.tool_panel, self.category_var, *self.categories)
            self.category_menu.pack(anchor=tk.N, padx=10, pady=10)
        
    def load_image(self):
        self.draw_boxes = [] 
        self.deleted_bboxes = []
        
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if self.image_path:
            self.image_path_label.config(text=f"Current Image Path: {self.image_path}")
            image = Image.open(self.image_path)
            image = image.resize((600, 600))  # Resize the image
            self.photo = ImageTk.PhotoImage(image)

            # Clear any existing drawings on the canvas
            if self.canvas:
                self.canvas.destroy()

            # Create a new canvas on top of the image
            self.canvas = tk.Canvas(self.image_frame, width=600, height=600, bg="white", bd=0, highlightthickness=0)
            self.canvas.pack()
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Bind mouse events for drawing
            self.canvas.bind("<Button-1>", self.start_draw)
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def enable_drawing(self):
        self.drawing = True
        self.draw_btn_isclick = True

    def start_draw(self, event):
        if self.draw_btn_isclick == True:
            self.drawing = True
        if self.drawing:
            self.bbox_start = (event.x, event.y)

    def draw(self, event):
        if self.drawing:
            if self.draw_boxes:
                # Delete the previous rectangle if it exists
                for bbox_id in self.draw_boxes:
                    self.canvas.delete(bbox_id)
                
            # Draw a new rectangle with the updated coordinates
            bbox_color = self.get_category_color(self.category_var.get())
            bbox_id = self.canvas.create_rectangle(self.bbox_start[0], self.bbox_start[1],
                                                            event.x, event.y,
                                                            outline=bbox_color, width=3, tag="bbox")
            self.draw_boxes.append(bbox_id)
            self.bbox_end = (event.x, event.y)

    def stop_draw(self, event):
        if self.drawing:
            self.drawing = False
            # Add the final bounding box to the list of current bounding boxes
            self.current_bboxes.append([[self.bbox_start[0], self.bbox_start[1]],[self.bbox_end[0], self.bbox_end[1]],[self.get_category_color(self.category_var.get())]])
            self.draw_bounding_boxes()
        
    def draw_bounding_boxes(self):
        bbox_coords_text = ""
        draw_boxes_coords_temp = {}  # Temporary dictionary to store new coordinates

        if len(self.current_bboxes) >= 0:
            if self.undo == 1:
                self.canvas.delete("bbox")
            self.undo_button.config(state="normal")
            # Draw each bounding box
            for bbox_info in self.current_bboxes:
                if isinstance(bbox_info[0][0], torch.Tensor) and isinstance(bbox_info[0][1], torch.Tensor):
                    start_x = int(bbox_info[0][0].tolist()) 
                    start_y = int(bbox_info[0][1].tolist())  
                    end_x = int(bbox_info[1][0].tolist())  
                    end_y = int(bbox_info[1][1].tolist()) 
                else:
                    start_x, start_y = bbox_info[0]
                    end_x, end_y = bbox_info[1]
                
                bbox_coords = f"Bounding Box: ({start_x}, {start_y}) - ({end_x}, {end_y})\n"
    
                # Check if the coordinates are not in draw_boxes_coords_temp (remove duplicates)
                if bbox_coords not in draw_boxes_coords_temp.values():
                    bbox_color = bbox_info[2]
                    bbox_id = self.canvas.create_rectangle(start_x, start_y, end_x, end_y, outline=bbox_color, width=3, tag="bbox")
                    self.draw_boxes.append(bbox_id)
                    draw_boxes_coords_temp[bbox_id] = bbox_coords
                    bbox_coords_text += bbox_coords
            self.current_image_coordinate_label.config(text=bbox_coords_text)
        else:
            self.undo_button.config(state="disabled")

    def get_category_color(self, category):
        # Assign a unique color to each category
        if category in self.category_colors:
            return self.category_colors[category]
        else:
            # Generate a random color if category is not found
            color = "#%06x" % random.randint(0, 0xFFFFFF)
            self.categories.append(category)
            self.category_colors[category] = color
            self.category_menu.destroy()  # Destroy the old menu
            self.category_menu = tk.OptionMenu(self.tool_panel, self.category_var, *self.categories)
            self.category_menu.pack(anchor=tk.N, padx=10, pady=10)
            return self.category_colors[category]
    
    def undo(self):
        self.undo = 1
        self.redo = 1
        if self.draw_boxes and self.current_bboxes:
            bbox_id = self.draw_boxes.pop()
            self.canvas.delete(bbox_id)
            self.deleted_bboxes.append(self.current_bboxes.pop())
            self.draw_bounding_boxes()
        if self.deleted_bboxes:
            self.redo_button.config(state="normal")
    
    def redo(self):
        if len(self.deleted_bboxes) > 0:
            if self.deleted_bboxes:
                bbox_info = self.deleted_bboxes.pop()
                start_x, start_y = bbox_info[0]
                end_x, end_y = bbox_info[1]
                bbox_color = bbox_info[2]
                bbox_id = self.canvas.create_rectangle(start_x, start_y, end_x, end_y, outline=bbox_color, width=3, tag="bbox")
                self.draw_boxes.append(bbox_id)
                self.current_bboxes.append(bbox_info)
                self.draw_bounding_boxes()
                if len(self.deleted_bboxes) == 0:
                    self.redo_button.config(state="disabled")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionAnnotationGUI(root)
    root.mainloop()
