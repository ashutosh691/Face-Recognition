import os
import sys
import time
import cv2
import face_recognition

class FaceRecognitionSystem:
    def __init__(self):
        """
        Step 1: Initialize the application workspace.
        We set up our directory structure and prepare our data tracking lists.
        """
        # Define a standard directory on the disk to save face profiles
        self.storage_directory = "saved_faces"
        
        # Automatically create the folder if it doesn't exist yet
        if not os.path.exists(self.storage_directory):
            os.makedirs(self.storage_directory)

        # Core data structures to keep track of face data in system memory
        self.known_face_encodings = []
        self.known_face_names = []

        # Step 2: Pre-load any profiles already saved on the computer
        self.load_stored_profiles()

        # Step 3: Connect directly to the MacBook's FaceTime HD Camera hardware
        # Index 0 targets the default integrated laptop webcam
        self.camera = cv2.VideoCapture(0)
        
        # Check if the camera hardware actually opened successfully
        if not self.camera.isOpened():
            print("Error: Could not access the webcam. Please verify privacy permissions.")
            sys.exit(1)

    def load_stored_profiles(self):
        """
        Step 2 Detail: Scans the storage directory, extracts 128-dimensional 
        facial features from old images, and loads them into memory.
        """
        # Clear existing memory lists to prevent duplicate processing entries
        self.known_face_encodings = []
        self.known_face_names = []

        # Look at every single file saved inside the folder
        for file_name in os.listdir(self.storage_directory):
            # Process only standard image formats
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Derive the person's name by stripping away the file extension
                person_name = os.path.splitext(file_name)[0]
                full_image_path = os.path.join(self.storage_directory, file_name)

                try:
                    # Load the image file from disk into a pixel array
                    loaded_image = face_recognition.load_image_file(full_image_path)
                    
                    # Convert the face image into a 128-dimensional numerical vector
                    face_vectors = face_recognition.face_encodings(loaded_image)

                    # Ensure at least one distinct face layout was detected in the photo
                    if len(face_vectors) > 0:
                        self.known_face_encodings.append(face_vectors[0])
                        self.known_face_names.append(person_name)
                except Exception as error:
                    print(f"Skipping unreadable profile file {file_name}: {error}")

        print(f"--- Database Initialized: Loaded {len(self.known_face_names)} profile(s) ---")

    def run(self):
        """
        Step 4: The Main Application Loop.
        Handles high-speed processing, feature extraction, and live graphics rendering.
        """
        print("\n=======================================================")
        print(" Face Recognition Desktop Application Active")
        print(" Commands: Press 's' to Save Current Face | 'q' to Quit")
        print("=======================================================\n")

        # Variables to calculate and display the actual processing frame rate (FPS)
        prev_time = 0
        current_fps = 0

        while True:
            # 1. Capture a live frame from the webcam stream feed
            frame_grabbed, frame = self.camera.read()
            if not frame_grabbed:
                print("Error: Live video feed interrupted.")
                break

            # 2. Downscale frame to 1/4 size for optimized processing speeds
            # This drastically lowers CPU overhead, keeping performance smooth and responsive
            optimized_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # 3. Convert color profiles from OpenCV default (BGR) to AI engine default (RGB)
            rgb_small_frame = cv2.cvtColor(optimized_small_frame, cv2.COLOR_BGR2RGB)

            # 4. Use the AI model to locate bounding boxes and compute facial landmarks
            live_face_locations = face_recognition.face_locations(rgb_small_frame)
            live_face_encodings = face_recognition.face_encodings(rgb_small_frame, live_face_locations)

            # Lists to store the identities determined for the current frame
            detected_names = []

            for live_encoding in live_face_encodings:
                # 5. Evaluate the live face vector against our database matrix
                # Tolerance 0.55 provides balanced precision, avoiding false positives
                matches = face_recognition.compare_faces(self.known_face_encodings, live_encoding, tolerance=0.55)
                assigned_identity = "Unknown"

                if True in matches:
                    # Find the exact list index where the match was triggered
                    matched_index = matches.index(True)
                    assigned_identity = self.known_face_names[matched_index]

                detected_names.append(assigned_identity)

            # 6. Render user interface bounding boxes and text overlays
            for face_box, name in zip(live_face_locations, detected_names):
                # Scale coordinates back up by 4x to match the original display window dimensions
                top_coord = face_box[0] * 4
                right_coord = face_box[1] * 4
                bottom_coord = face_box[2] * 4
                left_coord = face_box[3] * 4

                # Draw a clear green bounding box frame surrounding the detected face
                cv2.rectangle(frame, (left_coord, top_coord), (right_coord, bottom_coord), (0, 255, 0), 2)

                # Draw a matching solid green label bar block right below the chin line
                cv2.rectangle(frame, (left_coord, bottom_coord - 30), (right_coord, bottom_coord), (0, 255, 0), cv2.FILLED)
                
                # Render the tracked name string inside the green label bar background
                cv2.putText(frame, name, (left_coord + 6, bottom_coord - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Calculate actual performance frames per second (FPS) dynamically
            current_time = time.time()
            fps_delta = current_time - prev_time
            prev_time = current_time
            if fps_delta > 0:
                current_fps = int(1 / fps_delta)

            # Overlay performance metrics directly in the top-left corner of the window
            cv2.putText(frame, f"FPS: {current_fps}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Render the final interactive output window onto the desktop screen
            cv2.imshow("Face Recognition System Dashboard", frame)

            # Wait exactly 1 millisecond for key inputs from the hardware keyboard
            keyboard_stroke = cv2.waitKey(1) & 0xFF

            # Option A: User wants to register a new face profile profile ('s' key)
            if keyboard_stroke == ord('s'):
                if len(live_face_locations) > 0:
                    print("\n--- Registration Initiated ---")
                    new_profile_name = input("Enter the name of this person to save: ").strip()
                    
                    if new_profile_name:
                        # Construct a unique destination filename matching the input name string
                        destination_path = os.path.join(self.storage_directory, f"{new_profile_name}.jpg")
                        
                        # Save the crisp, unscaled full-frame image straight to disk as a reference
                        cv2.imwrite(destination_path, frame)
                        print(f"Profile saved successfully: {destination_path}")
                        
                        # Instantly refresh the runtime memory database to recognize the user
                        self.load_stored_profiles()
                    else:
                        print("Registration cancelled: Invalid name input entered.")
                else:
                    print("\nWarning: No faces are visible in the camera frame to save.")

            # Option B: User wants to safely shut down the system ('q' key)
            elif keyboard_stroke == ord('q'):
                print("\nShutting down system gracefully...")
                break

        # Step 5: Clean up system hooks and release hardware resources
        self.camera.release()
        cv2.destroyAllWindows()
        print("Application closed down successfully.")

if __name__ == "__main__":
    # Create the instance configuration and boot up the pipeline
    system_instance = FaceRecognitionSystem()
    system_instance.run()