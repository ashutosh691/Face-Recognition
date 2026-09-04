import face_recognition
import os
import cv2

class FaceDatabase:
    def __init__(self, storage_dir="known_faces"):
        self.storage_dir = storage_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        self.load_database()

    def load_database(self):
        """Loads all saved faces from the folder and extracts features."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for file_name in os.listdir(self.storage_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(file_name)[0]
                image_path = os.path.join(self.storage_dir, file_name)
                
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
        print(f"Database loaded: {len(self.known_face_names)} profiles found.")

    def save_new_face(self, frame, name):
        """Saves a new face snapshot to disk."""
        file_path = os.path.join(self.storage_dir, f"{name}.jpg")
        cv2.imwrite(file_path, frame)
        self.load_database()