Face Recognition & Real-Time Proctoring System

A complete Python-based system for face detection, face recognition, and live proctoring, designed for smart schools, attendance systems, and exam-monitoring platforms.

This project uses MTCNN for face detection and InceptionResnet (FaceNet) for face embedding + recognition.
It supports face collection, dataset management, and real-time recognition via webcam or IP camera streams.
1. Project Overview
  This application allows you to:
  Collect face images for any user
  Train a recognition model on your dataset
  Recognize faces in real time
  Detect eye-movement/cheating behavior (optional module)
  Integrate with web backends (GET/POST endpoints available)
  Suitable for:
  Smart School Projects
  Attendance Systems
  Exam Proctoring
  Computer Vision Research
2. Features
  Core Features
  Real-time face detection (MTCNN)
  High-accuracy face recognition (InceptionResnet V1)
  Automatic dataset creation and management
  Live video feed processing
  Face embeddings stored in pickle files
  REST-style endpoints to integrate with any backend
  Optional / Extendable
  Eye movement tracking
  Cheating detection alerts
  Attendance logs
  API connection with .NET/Blazor frontends
  
This repository contains a modular computer-vision system implementing MTCNN-based face detection and FaceNet-based embedding generation for robust face recognition. The platform enables dataset creation, model training, and real-time recognition pipelines suitable for smart-school ecosystems, biometric authentication, and proctoring solutions. The codebase is fully extendable and designed with clean architecture principles to support future integration with enterprise systems.

5. Installation & Setup
  Prerequisites
  Python 3.8–3.11
  pip
  OpenCV
  PyTorch
  facenet-pytorch

 Install Requirements
  pip install -r requirements.txt

6. Project Structure
  project/
  │
  ├── models/                # Pretrained models (MTCNN, InceptionResnet)
  ├── dataset/               # Saved face images per user
  ├── embeddings/            # .pkl files of embeddings
  ├── face_collect.py        # Collect face images
  ├── recognize_live.py      # Live recognition
  ├── train_embeddings.py    # Train and generate embeddings
  ├── utils/                 # Helper functions
  └── README.md

7. How It Works (Architecture)
  Pipeline Architecture
    +------------------+
    |   Video Input    |
    | (Webcam / File)  |
    +--------+---------+
             |
             v
    +------------------+
    |     MTCNN        |
    | Face Detection   |
    +--------+---------+
             |
             v
    +------------------+
    | InceptionResnet  |
    | Face Embeddings  |
    +--------+---------+
             |
             v
    +----------------------+
    | Compare With Stored  |
    | Embeddings (.pkl)    |
    +--------+-------------+
             |
             v
    +----------------------+
    |    Identity Output   |
    +----------------------+

8. API-Style Endpoints

  These are not real HTTP endpoints unless you expose them.
  They explain how to integrate with your backend.
  
  Collect Faces
  GET /ai/collect_faces?id=<USER_ID>&name=<USER_NAME>
Meaning:
  You start the camera
  Create a dataset folder for the user
  Capture ~20–50 face images automatically
  Recognize Faces (Live)

  Recognize Faces (Live)
  GET /ai/recognize_live
Meaning:
  Start webcam
  Detect faces
  Compare with stored embeddings
  Return the recognized user ID + name

  9. Usage
    1. Collect Faces
    python face_collect.py --id 1 --name "----"

    2. Train Embeddings
    python train_embeddings.py

    3. Run Real-Time Recognition
    python recognize_live.py

   10. For Portfolio & Recruiters (Job-Focused Section)
    Key Technical Skills Demonstrated
    Deep Learning (MTCNN, FaceNet)
    PyTorch
    Computer Vision
    Real-time video processing
    Dataset engineering
    API integration
    Smart-school / Ed-Tech solution design
   Why This Project Matters
    This system demonstrates the ability to design and implement a full AI pipeline—from data collection to model integration—aligned with real-world use cases in education, monitoring,     and security.

  11. Contact
  For any inquiries or contributions:
  Developer: Aktham
  Email: (akthamhamza01@gmail.com)
  LinkedIn: (www.linkedin.com/in/aktham-hamza-48b750395)
