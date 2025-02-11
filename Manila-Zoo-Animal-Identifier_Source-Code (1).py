
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import onnxruntime
import yaml
from ultralytics import YOLO

class YOLOPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manila Zoo Animals Identification System")

        # Load ONNX model
        self.onnx_model_path = 'C:\\Users\\Merrell\\Documents\\Source Codes\\Python projects\\pythonProject\\Image-classification.onnx'
        self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path)

        # Load data
        with open("C:\\Users\\Merrell\\Documents\\Source Codes\\Python projects\\pythonProject\\data.yaml", 'r') as stream:
            self.num_classes = str(yaml.safe_load(stream)['nc'])

        # Load new model
        self.my_new_model = YOLO('C:\\Users\\Merrell\\Documents\\Source Codes\\Python projects\\pythonProject\\last.pt')

        self.create_widgets()

    def create_widgets(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        # Button to open image 
        self.btn_open_image = tk.Button(button_frame, text="Open Image", command=self.open_image)
        self.btn_open_image.pack(side=tk.LEFT)

        # Button to predict 
        self.btn_predict = tk.Button(button_frame, text="Predict", command=self.predict)
        self.btn_predict.pack(side=tk.LEFT)

        # display image
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()

        # Label to display predicted class and confidence
        self.label_predictions = tk.Label(self.root, text="")
        self.label_predictions.pack()

    def open_image(self):
        # Open file 
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                                filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
        if file_path:
            self.image_path = file_path

            # Display 
            image = Image.open(self.image_path)
            image = image.resize((500, 500))  # Resize for display
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def predict(self):
        if hasattr(self, 'image_path'):
            new_results = self.my_new_model.predict(self.image_path, conf=0.2)

            new_result = new_results[0]
            detected_boxes = new_result.boxes.data
            class_labels = detected_boxes[:, -1].int().tolist()

            if class_labels:
                predicted_class_id = class_labels[0]
                predicted_class_name = new_result.names[predicted_class_id]
                confidence_level = detected_boxes[0, 4].item()

                if confidence_level >= 0.55:
                    self.label_predictions.config(text=f"Predicted Class: {predicted_class_name}, Confidence: {confidence_level:.2f}")
                else:
                    self.label_predictions.config(text="Attached image cannot be identified.")
            else:
                self.label_predictions.config(text="No object detected.")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOPredictorApp(root)
    root.mainloop()
    
