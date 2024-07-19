import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import tkinter as tk
from tkinter import messagebox

from ultralytics import YOLO

model = YOLO("final_combined.pt")

cap = cv2.VideoCapture(0)
root = None  # Global variable for Tkinter window

def start():
    global root  # Use the global root variable

    if messagebox.askokcancel("Get Ready", "Please ensure you are prepared before continuing. By proceeding, you give us permission to use your camera to monitor you."):
        root.destroy()  # Destroy the main window created in main()

        names = model.names
        start_phone = None
        start_face = None
        start_multipleFace = None
        head_time = None
        elapsed_time = None
        reset_time = None
        count = 0
        reset_interval = 20
        last_transitionincrement_time = None
        last_increment_time = None

        transition_time = None
        transition_direction = None
        transition_count = 0

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        while cap.isOpened():
            try:
                success, image = cap.read()
                if not success:
                    messagebox.showwarning("Warning", "Camera is off or disconnected!")
                    break

                start = time.time()
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)
                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        if y < -12:
                            text = "Looking Left"
                            if head_time is None:
                                head_time = time.time()
                            else:
                                elapsed_time = time.time() - head_time
                                if elapsed_time >= 10:
                                    count += 1
                                    last_increment_time = time.time()
                                    head_time = None
                                    winsound.MessageBeep(0)
                                    if count == 3:
                                        messagebox.showwarning("Warning", "Please remain focused on your screen to ensure the integrity of the exam.")
                                        winsound.MessageBeep(0)
                                        count = 0
                                    
                            # Transition tracking for left
                            if transition_direction is None:
                                transition_direction = "Left"
                                transition_time = time.time()
                            elif transition_direction == "Right":
                                if time.time() - transition_time <= 5:
                                    transition_count += 1
                                    last_transitionincrement_time = time.time()
                                    transition_direction = "Left"
                                    transition_time = time.time()
                                    winsound.MessageBeep(0)
                                    if transition_count == 3:
                                        messagebox.showwarning("Warning", "Please remain focused on your screen to ensure the integrity of the exam.")
                                        winsound.MessageBeep(0)
                                        transition_count = 0

                                else:
                                    transition_direction = "Left"
                                    transition_time = time.time()

                        elif y > 12:
                            text = "Looking Right"
                            if head_time is None:
                                head_time = time.time()
                            else:
                                elapsed_time = time.time() - head_time
                                if elapsed_time >= 10:
                                    count += 1
                                    last_increment_time = time.time()
                                    head_time = None
                                    winsound.MessageBeep(0)
                                    if count == 3:
                                        messagebox.showwarning("Warning", "Please remain focused on your screen to ensure the integrity of the exam.")
                                        winsound.MessageBeep(0)
                                        count = 0
                            
                            # Transition tracking for right
                            if transition_direction is None:
                                transition_direction = "Right"
                                transition_time = time.time()
                            elif transition_direction == "Left":
                                if time.time() - transition_time <= 5:
                                    transition_count += 1
                                    last_transitionincrement_time = time.time()
                                    transition_direction = "Right"
                                    transition_time = time.time()
                                    winsound.MessageBeep(0)
                                    if transition_count == 3:
                                        messagebox.showwarning("Warning", "Please remain focused on your screen to ensure the integrity of the exam.")
                                        winsound.MessageBeep(0)
                                        transition_count = 0
                                else:
                                    transition_direction = "Right"
                                    transition_time = time.time()

                        elif x < -1:
                            text = "Looking Down"
                        elif x > 15:
                            text = "Looking Up"
                        else:
                            text = "Forward"
                            head_time = None

                        if count != 3 and last_increment_time is not None:
                            elapsed_since_last_increment = time.time() - last_increment_time
                            if elapsed_since_last_increment >= 20:
                                count = 0

                        if transition_count != 3 and last_transitionincrement_time is not None:
                            elapsed_since_last_transitionincrement = time.time() - last_transitionincrement_time
                            if elapsed_since_last_transitionincrement >= 10:
                                transition_count = 0

                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                         dist_matrix)
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        print(elapsed_time)
                        print(count)
                        print(f"Transition Count: {transition_count}")

                end = time.time()
                totalTime = end - start

                result = model.predict(image, show=True, stream=True)

                for r in result:
                    object_detected = False
                    smartphone_detected = False
                    multiple_detected = False
                    face_id = list(names)[list(names.values()).index('Face')]
                    face_count = r.boxes.cls.tolist().count(face_id)
                    print(f"Number of detected smartphones: {face_count}")

                    if face_count >= 2:
                        multiple_detected = True

                    if start_multipleFace is None:
                        start_multipleFace = time.time()
                    else:
                        multipleFace_elapsed_time = time.time() - start_multipleFace

                        if multipleFace_elapsed_time >= 10:
                            messagebox.showwarning("Warning", "More than one face has been detected!")
                            print("More than one face is detected!")
                            winsound.MessageBeep(0)
                            print("More than one Face has been detected!")
                            start_multipleFace = None

                for c in r.boxes.cls:
                    if names[int(c)] == "Smartphone":
                        smartphone_detected = True
                        if start_phone is None:
                            start_phone = time.time()

                        else:
                            phone_elapsed_time = time.time() - start_phone

                            if phone_elapsed_time >= 5:
                                messagebox.showwarning("Warning", "Smartphone Detected!")
                                winsound.MessageBeep(0)
                                print("Smartphone detected!")
                                start_phone = None

                    if names[int(c)] == "Face":
                        object_detected = True
                        start_face = None
                        print("Face Detected")

                if not object_detected:
                    print("No Face Detected")
                    if start_face is None:
                        start_face = time.time()
                    else:
                        face_elapsed_time = time.time() - start_face
                        print(face_elapsed_time)
                        if face_elapsed_time >= 5:
                            messagebox.showwarning("Warning", "No Face Detected!")
                            winsound.MessageBeep(0)

                            start_face = None
                if not smartphone_detected:
                    start_phone = None

                if not multiple_detected:
                    start_multipleFace = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    root.quit()
                    break

            except Exception as e:
                messagebox.showwarning("Warning", f"An error occurred: {str(e)}")
                break

        cap.release()
    else:
        print("Cancelled")

def exit():
    global root
    if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
        root.quit()

def main():
    global root
    root = tk.Tk()
    root.title("Start and Exit Buttons")

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=20)

    start_button = tk.Button(frame, text="Start", command=start, width=15, height=2)
    start_button.pack(side=tk.LEFT, padx=5)

    exit_button = tk.Button(frame, text="Exit", command=exit, width=15, height=2)
    exit_button.pack(side=tk.RIGHT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
