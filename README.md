# Thief Detection Application

This application detects a thief's face in a folder of images using a provided thief image. It uses Tkinter for the GUI and `deepface` for face detection.

## Installation

1. Clone the repository:

   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Install `face_recognition_models`:

   ```sh
   pip install git+https://github.com/ageitgey/face_recognition_models
   ```

5. Run the application:
   ```sh
   python main.py
   ```

## Requirements

- Python 3.6+
- Tkinter
- opencv-python
- Pillow
- deepface
- tf-keras

## Usage

1. Click on "Select Thief Image" to choose the image of the thief.
2. Click on "Select Image Folder" to choose the folder containing images to scan.
3. Click on "Detect Thief" to start the detection process.
4. The application will display the number of images where the thief's face is detected and show thumbnails of those images.

## Notes

- Ensure that the images in the folder are in `.png`, `.jpg`, or `.jpeg` format.
- The application may take some time to process a large number of images.
