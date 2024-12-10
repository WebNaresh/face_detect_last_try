import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from deepface import DeepFace
import logging
import threading
from tkinter import ttk
from zipfile import ZipFile
import time
import psutil

logging.basicConfig(level=logging.INFO)
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

def log_memory_usage(message):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"{message} - Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def select_thief_image():
    global thief_image_path, thief_face, thief_face_ready
    thief_image_path = filedialog.askopenfilename()
    thief_image_label.config(text=f"Thief Image: {thief_image_path}")
    logging.info(f"Selected thief image: {thief_image_path}")

    def obtain_face_representation():
        global thief_face
        try:
            thief_face = DeepFace.represent(img_path=thief_image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            logging.info("Thief face representation obtained.")
        except ValueError as e:
            result_label.config(text=f"No face detected in the thief image: {e}")
            logging.error(f"No face detected in the thief image: {e}")
            thief_face = None
        finally:
            thief_face_ready.set()

    thief_face_ready = threading.Event()
    threading.Thread(target=obtain_face_representation).start()

def select_image_folder():
    global folder_path, total_images
    folder_path = filedialog.askdirectory()
    folder_label.config(text=f"Folder: {folder_path}")
    logging.info(f"Selected image folder: {folder_path}")
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_paths)
    progress_bar.config(maximum=total_images)
    progress_label.config(text=f"Total images: {total_images}")
    logging.info(f"Total images in folder: {total_images}")

def download_images():
    global detected_images
    if not detected_images:
        logging.warning("No images to download.")
        result_label.config(text="No images to download.")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
    if not save_path:
        logging.info("Save operation cancelled.")
        return

    with ZipFile(save_path, 'w') as zipf:
        for img_path in detected_images:
            zipf.write(img_path, os.path.basename(img_path))
    logging.info(f"Images downloaded successfully to {save_path}.")
    result_label.config(text=f"Images downloaded successfully to {save_path}.")

def process_image(image_path):
    try:
        # Try to read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        
        # Detect faces in the image
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend='opencv', enforce_detection=False)
        if len(faces) == 0:
            logging.info(f"No faces detected in image: {image_path}")
            return None

        # Verify each detected face
        for face in faces:
            result = DeepFace.verify(img1_path=thief_image_path, img2_path=image_path,  model_name = models[2])  # Adjusted threshold
            logging.info(f"Verification result: {result["distance"]}, verified: {result['verified']} and also {result['distance'] <= 0.008}")
            logging.info(f"Path of Image {image_path}")
            logging.info(result)
            
            if  result["distance"] <= 0.008:  # Adding confidence check
                logging.info(f"Thief detected")
                return image_path
    except cv2.error as e:
        logging.error(f"OpenCV error processing image {image_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
    return None

def update_progress_bar(progress, total):
    progress_var.set(progress)
    progress_label.config(text=f"Compared {progress} of {total} images")

def detect_thief():
    global detected_images
    detected_images = []
    def run_detection():
        try:
            if not thief_image_path or not folder_path:
                result_label.config(text="Please select both thief image and folder.")
                logging.warning("Thief image or folder not selected.")
                return

            logging.info("Waiting for thief face representation...")
            thief_face_ready.wait()  # Wait for the face representation to be ready

            if thief_face is None:
                result_label.config(text="No face detected in the thief image.")
                logging.warning("No face detected in the thief image.")
                return

            logging.info("Starting thief detection...")
            thief_image = cv2.imread(thief_image_path)
            if thief_image is None:
                raise ValueError(f"Unable to read thief image: {thief_image_path}")

            image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
            progress_var.set(0)

            start_time = time.time()

            def process_images_sequentially():
                for i, img_path in enumerate(image_paths):
                    try:
                        logging.info(f"Processing image {i + 1} of {total_images}")
                        result = process_image(img_path)
                        if result:
                            detected_images.append(result)
                    except Exception as e:
                        logging.error(f"Error processing image {img_path}: {e}")
                    update_progress_bar(i + 1, total_images)

            logging.info("Processing images sequentially...")
            process_images_sequentially()

            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"Total time taken for comparison: {total_time:.2f} seconds")

            result_label.config(text=f"Detected {len(detected_images)} images with thief. Time taken: {total_time:.2f} seconds")
            logging.info(f"Detection complete. {len(detected_images)} images detected with thief.")
            for img_path in detected_images:
                try:
                    img = Image.open(img_path)
                    img.thumbnail((100, 100))
                    img = ImageTk.PhotoImage(img)
                    panel = tk.Label(scrollable_frame, image=img)
                    panel.image = img
                    panel.pack()
                except Exception as e:
                    logging.error(f"Error displaying image {img_path}: {e}")
        except Exception as e:
            logging.error(f"Error in run_detection: {e}")

    threading.Thread(target=run_detection).start()

def reset_state():
    global thief_image_path, folder_path, detected_images, thief_face, thief_face_ready, total_images
    thief_image_path = ""
    folder_path = ""
    detected_images = []
    thief_face = None
    thief_face_ready = threading.Event()
    total_images = 0

    thief_image_label.config(text="Thief Image: Not selected")
    folder_label.config(text="Folder: Not selected")
    result_label.config(text="")
    progress_var.set(0)
    progress_label.config(text="Compared 0 of 0 images")

    for widget in scrollable_frame.winfo_children():
        widget.destroy()

root = tk.Tk()
root.title("Thief Detection Application")

thief_image_path = ""
folder_path = ""


select_thief_button = tk.Button(root, text="Select Thief Image", command=select_thief_image)
select_thief_button.pack()

thief_image_label = tk.Label(root, text="Thief Image: Not selected")
thief_image_label.pack()

select_folder_button = tk.Button(root, text="Select Image Folder", command=select_image_folder)
select_folder_button.pack()

folder_label = tk.Label(root, text="Folder: Not selected")
folder_label.pack()

detect_button = tk.Button(root, text="Detect Thief", command=detect_thief)
detect_button.pack()

reset_button = tk.Button(root, text="Reset", command=reset_state)
reset_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

image_frame = ttk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(image_frame)
scrollbar = ttk.Scrollbar(image_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

# Load the model using DeepFace.build_model
model = DeepFace.build_model(models[1])


canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

download_button = tk.Button(root, text="Download Detected Images", command=download_images)
download_button.pack()

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(root, variable=progress_var)
progress_bar.pack()

progress_label = tk.Label(root, text="Compared 0 of 0 images")
progress_label.pack()

root.mainloop()
