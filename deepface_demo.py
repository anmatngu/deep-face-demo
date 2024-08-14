import gradio as gr
from deepface import DeepFace
import cv2
import time

def verify_faces(img1, img2):
    try:
        img1_path = "img1.jpg"
        img2_path = "img2.jpg"
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)

        backends = [
            'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 
            'yolo', 'yunet', 'centerface', 'faster_mtcnn'
        ]
        best_backend = None
        best_time = float('inf')
        best_result = None

        for backend in backends:
            start_time = time.time()
            try:
                result = DeepFace.verify(img1_path, img2_path, enforce_detection=True, detector_backend=backend)
            except Exception as e:
                print(f"Error with backend {backend}: {e}")
                continue  # Skip to the next backend if there's an error

            elapsed_time = time.time() - start_time

            if result["verified"] and elapsed_time < best_time:
                best_time = elapsed_time
                best_backend = backend
                best_result = result

        if best_result:
            # Draw bounding boxes using the best backend
            try:
                img1_faces = DeepFace.extract_faces(img1_path, detector_backend=best_backend, enforce_detection=True)
                img2_faces = DeepFace.extract_faces(img2_path, detector_backend=best_backend, enforce_detection=True)
            except Exception as e:
                return f"Error extracting faces with {best_backend}: {e}", img1, img2

            for face in img1_faces:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]
                cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for face in img2_faces:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]
                cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Additional Features:
            recognition_result = DeepFace.find(img1_path, db_path=r"C:\Users\nguye\Documents\University\DPL302m\deep_face\my_db")
            attributes = DeepFace.analyze(img1_path, actions=['age', 'gender', 'race', 'emotion'])

            return (f"Verified: {best_result['verified']} with {best_backend} in {best_time:.2f} seconds\n"
                    f"Recognition Result: {recognition_result}\n"
                    f"Attributes: {attributes}"), img1, img2
        else:
            return "No verification was successful.", img1, img2

    except Exception as e:
        return f"Error processing: {str(e)}", img1, img2

def anti_spoofing(img1):
    try:
        img1_path = "img1.jpg"
        cv2.imwrite(img1_path, img1)

        models = ['Facenet512', 'VGG-Face', 'DeepFace', 'ArcFace']  # Add more models as needed
        best_model = None
        best_time = float('inf')
        results = {}

        for model in models:
            try:
                start_time = time.time()
                face_objs = DeepFace.extract_faces(img_path=img1_path, detector_backend='opencv', enforce_detection=True, anti_spoofing=True, model_name=model)
                elapsed_time = time.time() - start_time
                img1_is_real = all(face_obj["is_real"] for face_obj in face_objs)
                results[model] = img1_is_real

                if img1_is_real and elapsed_time < best_time:
                    best_time = elapsed_time
                    best_model = model

            except Exception as e:
                results[model] = f"Error: {str(e)}"

        if best_model:
            return f"Best Model: {best_model} with result: {results[best_model]} in {best_time:.2f} seconds", img1
        else:
            return "No model could verify the image as real.", img1

    except Exception as e:
        return f"Error during anti-spoofing check: {str(e)}", img1

iface_verification = gr.Interface(
    fn=verify_faces,
    inputs=[gr.Image(type="numpy"), gr.Image(type="numpy")],
    outputs=["text", gr.Image(type="numpy"), gr.Image(type="numpy")],
    title="Face Verification and Analysis",
    description="Upload two images to verify if they are of the same person, analyze facial attributes, and perform anti-spoofing."
)

iface_anti_spoofing = gr.Interface(
    fn=anti_spoofing,
    inputs=[gr.Image(type="numpy")],
    outputs=["text", gr.Image(type="numpy")],
    title="Face Anti-Spoofing Check",
    description="Upload an image to check for anti-spoofing."
)

iface = gr.TabbedInterface([iface_verification, iface_anti_spoofing], ["Verification & Analysis", "Anti-Spoofing"])

if __name__ == "__main__":
    iface.launch()