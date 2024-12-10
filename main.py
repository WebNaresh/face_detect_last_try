import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from deepface import DeepFace
import logging
import threading

logging.basicConfig(level=logging.INFO)

def select_thief_image():
    global thief_image_path
    thief_image_path = filedialog.askopenfilename()
    thief_image_label.config(text=f"Thief Image: {thief_image_path}")
    logging.info(f"Selected thief image: {thief_image_path}")

def select_image_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    folder_label.config(text=f"Folder: {folder_path}")
    logging.info(f"Selected image folder: {folder_path}")

def detect_thief():
    def run_detection():
        if not thief_image_path or not folder_path:
            result_label.config(text="Please select both thief image and folder.")
            logging.warning("Thief image or folder not selected.")
            return

        logging.info("Starting thief detection...")
        thief_image = cv2.imread(thief_image_path)
        try:
            thief_face = DeepFace.represent(img_path=thief_image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            logging.info("Thief face representation obtained.")
        except IndexError:
            result_label.config(text="No face detected in the thief image.")
            logging.error("No face detected in the thief image.")
            return

        detected_images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                try:
                    result = DeepFace.verify(img1_path=thief_image_path, img2_path=image_path, model_name="Facenet", enforce_detection=False)
                    if result["verified"]:
                        detected_images.append(image_path)
                        logging.info(f"Thief detected in image: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")
                    continue

        result_label.config(text=f"Detected {len(detected_images)} images with thief.")
        logging.info(f"Detection complete. {len(detected_images)} images detected with thief.")
        for img_path in detected_images:
            img = Image.open(img_path)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)
            panel = tk.Label(root, image=img)
            panel.image = img
            panel.pack()

    threading.Thread(target=run_detection).start()

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

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
