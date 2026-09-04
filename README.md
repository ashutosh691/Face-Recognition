# Face Recognition System

A basic real-time face recognition system built using **Python**, **OpenCV**, and the **face_recognition** library.

The application uses a webcam to detect faces, compare them with previously stored face profiles, and display the recognized person's name in real time.

## Features

- Real-time face detection using a webcam
- Face recognition using facial encodings
- Registration of new faces
- Local storage of face profiles
- Recognition of known and unknown faces
- Real-time FPS display
- Optimized frame processing
- Simple keyboard-based controls

## Technologies Used

- **Python**
- **OpenCV**
- **face_recognition**
- **Computer Vision**

## Project Structure

```text
Face-Recognition/
│
├── main.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── known_faces/
│   └── .gitkeep
│
└── saved_faces/
    └── .gitkeep
```

## How the System Works

The system follows these basic steps:

```text
Webcam
   ↓
Capture Video Frame
   ↓
Resize Frame
   ↓
Convert BGR to RGB
   ↓
Detect Faces
   ↓
Generate Face Encodings
   ↓
Compare with Stored Faces
   ↓
Identify Person
   ↓
Display Name and Bounding Box
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Face-Recognition.git
```

### 2. Navigate to the Project Directory

```bash
cd Face-Recognition
```

### 3. Create a Virtual Environment

```bash
python3 -m venv face_env
```

### 4. Activate the Virtual Environment

For macOS/Linux:

```bash
source face_env/bin/activate
```

For Windows:

```bash
face_env\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

## Running the Project

Start the application using:

```bash
python3 main.py
```

The application will access the computer's webcam and open the **Face Recognition System Dashboard**.

## Controls

| Key | Function |
|-----|----------|
| `s` | Save/Register the detected face |
| `q` | Quit the application |

## Registering a New Face

1. Start the application.
2. Make sure a face is visible in the webcam.
3. Press `s`.
4. Enter the person's name in the terminal.
5. The face image will be saved locally.
6. The stored face profiles will be loaded again for recognition.

Example:

```text
Enter the name of this person to save: Ashutosh
Profile saved successfully
```

## Face Recognition

The system uses the `face_recognition` library to generate a numerical facial encoding for detected faces.

When a face is detected through the webcam, its encoding is compared with the stored face encodings.

If a matching profile is found, the person's name is displayed on the screen.

If no matching profile is found, the face is classified as:

```text
Unknown
```

## Performance Optimization

To improve real-time performance, the webcam frame is resized to **25% of its original dimensions** before face detection and encoding.

```python
optimized_small_frame = cv2.resize(
    frame,
    (0, 0),
    fx=0.25,
    fy=0.25
)
```

The detected face coordinates are then scaled back to the original frame size for displaying the bounding boxes.

## Face Database

The `database.py` file provides functionality for managing stored face profiles.

It can:

- Load saved face images
- Generate facial encodings
- Store names associated with face encodings
- Save new face images
- Reload the face database

Face profiles are stored locally on the computer.

## Requirements

The project requires the following Python libraries:

```text
opencv-python
face-recognition
```

These dependencies are listed in `requirements.txt`.

## Privacy

This project stores face images locally for recognition purposes.

Actual face images should **not be uploaded to a public GitHub repository** without appropriate consent.

Face image files are excluded from Git tracking using `.gitignore`.

## Limitations

- A working webcam is required.
- Recognition performance depends on lighting and camera quality.
- Recognition may be affected by changes in appearance or camera angle.
- The project is designed as a basic local face recognition application.
- Face profiles are stored locally.
- The current system is intended primarily for educational and demonstration purposes.

## Future Improvements

Possible improvements include:

- Graphical user interface
- Improved face matching
- Multiple face samples per person
- Recognition confidence/distance display
- Attendance tracking
- Recognition history
- Database integration
- Improved error handling
- Support for multiple cameras

## Author

**Ashutosh Upreti**

## License

This project is created for educational purposes.
