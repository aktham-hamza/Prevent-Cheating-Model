# استيراد المكتبات المطلوبة
import cv2                  # مكتبة معالجة الصور والفيديو
import pickle               # مكتبة لحفظ واسترجاع البيانات
import numpy as np          # مكتبة العمليات العددية
import os                   # مكتبة التعامل مع الملفات والمجلدات
import json                 # مكتبة التعامل مع ملفات JSON
import csv                  # مكتبة التعامل مع ملفات CSV
from datetime import datetime  # مكتبة للحصول على الوقت والتاريخ
from sklearn.neighbors import KNeighborsClassifier  # نموذج تصنيف KNN
from imutils import face_utils  # أدوات مساعدة لمعالجة الوجوه
import dlib                 # مكتبة متخصصة في كشف الوجوه والنقاط
import threading            # مكتبة تنفيذ العمليات بالتوازي
import platform             # مكتبة معرفة نوع نظام التشغيل
import time                 # مكتبة التعامل مع الزمن

# تعريف الكلاس FaceDetection الذي يحتوي على كل الوظائف المتعلقة بالوجوه
class FaceDetection:

    # دالة لتشغيل صوت تنبيهي عند إكمال جمع الصور أو عند الاكتمال
    def play_sound():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 500)  # نغمة بسيطة لمدة نصف ثانية
        else:
            os.system('play -nq -t alsa synth 0.5 sine 1000')  # لينوكس أو ماك

    # دالة لجمع صور الوجوه لتدريب النموذج
    def collect_faces(user_id, name, max_faces=20):
        # فتح الكاميرا من الموبايل لعدم وجود كاميرا في الابتوب عن طريق ربط الابتوب والموبايل على 
        # IP الشبكة ذاتها
        phone_IP = "http://192.168.217.136:8080/video"
        video=cv2.VideoCapture(phone_IP)
        
        # تحميل مصنف كشف الوجوه الجاهز من OpenCV
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        if facedetect.empty():
            raise Exception("🚨 خطأ: لم يتم تحميل مصنف الوجوه.")

        faces_data = []  # تخزين الصور
        frame_count = 0  # عداد الإطارات
        completed = False  # حالة اكتمال جمع الصور

        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (50, 50))

                if len(faces_data) < max_faces and frame_count % 10 == 0:
                    faces_data.append(resized_img)

                frame_count += 1

                # حساب النسبة المئوية للتقدم
                percent = int((len(faces_data) / max_faces) * 100)

                # رسم شريط بيضاوي حول الوجه مع تقدم النسبة
                center_x = x + w // 2
                center_y = y + h // 2
                axes_length = (w // 2, int(h * 0.6))
                cv2.ellipse(frame, (center_x, center_y), axes_length, 0, 0, 360, (80, 80, 80), 2)
                angle = int((percent / 100) * 360)
                green_color = (0, 255, 0)
                base_thickness = 8
                cv2.ellipse(frame, (center_x, center_y), axes_length, 0, 0, angle, green_color, base_thickness)

                cv2.putText(frame, f"{percent}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2)

                if percent >= 100 and not completed:
                    threading.Thread(target=FaceDetection.play_sound).start()
                    completed = True

            # عرض رسالة عند إكمال جمع الصور
            if completed:
                cv2.putText(frame, "Face collection complete!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("Capturing Faces", frame)
            if cv2.waitKey(1) == ord('q') or len(faces_data) >= max_faces:
                break

        # إنهاء الفيديو
        video.release()
        cv2.destroyAllWindows()

        # تجهيز بيانات الوجوه للحفظ
        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape(len(faces_data), -1)

        # إنشاء مجلد "models" إذا لم يكن موجود
        os.makedirs("models", exist_ok=True)

        # تخزين البيانات في الملفات المناسبة
        names_path = 'models/names.pkl'
        faces_path = 'models/faces_data.pkl'
        users_json_path = 'models/users.json'

        # تخزين معرفات المستخدمين
        if os.path.exists(names_path):
            with open(names_path, 'rb') as f:
                user_ids = pickle.load(f)
        else:
            user_ids = []

        user_ids += [user_id] * len(faces_data)
        with open(names_path, 'wb') as f:
            pickle.dump(user_ids, f)

        # تخزين بيانات الصور
        if os.path.exists(faces_path):
            with open(faces_path, 'rb') as f:
                faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)
        else:
            faces = faces_data

        with open(faces_path, 'wb') as f:
            pickle.dump(faces, f)

        # تحديث ملف أسماء المستخدمين
        if os.path.exists(users_json_path):
            with open(users_json_path, 'r') as f:
                users = json.load(f)
        else:
            users = {}

        users[str(user_id)] = name

        with open(users_json_path, 'w') as f:
            json.dump(users, f, indent=4)

        # إرجاع نتيجة النجاح
        return {
            "status": "success",
            "user_id": user_id,
            "name": name,
            "message": f"✅  collected {len(faces_data)}  photo of face '{name}' ID {user_id}"
        }

    # دالة للتعرف على الوجه وتسجيل الحضور
    def recognize_and_mark_attendance():
        # فتح الكاميرا وتحميل مصنف كشف الوجه
        phone_IP = "http://192.168.217.136:8080/video"
        video=cv2.VideoCapture(phone_IP)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # تحميل بيانات التدريب
        with open('models/names.pkl', 'rb') as w:
            LABELS = pickle.load(w)
        with open('models/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)

        # تدريب مصنف KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)

        # تحميل أسماء المستخدمين
        users_json_path = 'models/users.json'
        if os.path.exists(users_json_path):
            with open(users_json_path, 'r') as f:
                user_names = json.load(f)
        else:
            user_names = {}

        distance_threshold = 2600.0  # حد المسافة لتصنيف صحيح
        date = datetime.now().strftime("%d-%m-%Y")
        filename = f"models/Attendance_{date}.csv"
        os.makedirs("models", exist_ok=True)

        # إنشاء ملف الحضور إذا لم يكن موجوداً
        if not os.path.exists(filename):
            with open(filename, "w", newline='') as f:
                csv.writer(f).writerow(["NAME", "TIME"])

        # تحميل الأسماء المسجلة سابقاً
        logged_names = set()
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    logged_names.add(row[0])

        recognized_count = {}
        target_count = 5

        # بدء عملية التعرف وعرض النتائج
        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

                distances, _ = knn.kneighbors(resized_img, n_neighbors=1)
                mean_distance = distances[0][0]

                if mean_distance < distance_threshold:
                    output = knn.predict(resized_img)
                    user_id = str(output[0])
                    name_text = user_names.get(user_id, f"ID_{user_id}")

                    recognized_count[user_id] = recognized_count.get(user_id, 0) + 1

                    if name_text not in logged_names:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        with open(filename, "a", newline='') as f:
                            csv.writer(f).writerow([name_text, timestamp])
                        logged_names.add(name_text)

                    color = (0, 255, 0)
                else:
                    user_id = None
                    name_text = "Unknown"
                    color = (0, 0, 255)

                # رسم المستطيل على الوجه المكتشف
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == ord('q'):
                break

            # إذا تم التعرف عليه بشكل كاف
            for user_id, count in recognized_count.items():
                if count >= target_count:
                    video.release()
                    cv2.destroyAllWindows()
                    return {
                        "status": "success",
                        "recognized_id": user_id,
                        "name": user_names.get(user_id, "Unknown"),
                        "count": count,
                        "date": date,
                        "csv": filename,
                        "message": f"{user_names.get(user_id, 'Unknown')} تم التعرف عليه {count} مرات."
                    }

        video.release()
        cv2.destroyAllWindows()
        return {
            "status": "failed",
            "recognized": None,
            "message": "لم يتم التعرف على أي وجه بما فيه الكفاية."
        }

    # دالة لمسح جميع بيانات التدريب المخزنة
    def clear_training_data():
        files_to_clear = [
            'models/names.pkl',
            'models/faces_data.pkl',
            'models/users.json'
        ]

        for file_path in files_to_clear:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 تم حذف الملف: {file_path}")
            else:
                print(f"⚠️ الملف غير موجود: {file_path}")

    # دالة لتتبع النظر وتحذير من الغش أثناء الامتحان
    def face_traking(is_sheet, exam_time):
        
        # الإعدادات الأولية
        detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(os.path.dirname(__file__), "models/shape_predictor_68_face_landmarks.dat")
        predictor = dlib.shape_predictor(model_path)
        phone_IP = "http://192.168.217.136:8080/video"
        video=cv2.VideoCapture(phone_IP)

        total_frames = 0
        cheat_frames = 0
        start_time = time.time()

        # دالة لمراقبة اتجاه العين
        def get_eye_position(eye_points, shape, gray):
            eye_region = np.array([shape[i] for i in eye_points], np.int32)
            min_x, max_x = np.min(eye_region[:, 0]), np.max(eye_region[:, 0])
            min_y, max_y = np.min(eye_region[:, 1]), np.max(eye_region[:, 1])
            eye_frame = gray[min_y:max_y, min_x:max_x]
            eye_frame = cv2.resize(eye_frame, (80, 30))
            _, threshold_eye = cv2.threshold(eye_frame, 70, 255, cv2.THRESH_BINARY_INV)
            height, width = threshold_eye.shape
            left_part = threshold_eye[:, 0:width//2]
            right_part = threshold_eye[:, width//2:]

            left_intensity = np.sum(left_part)
            right_intensity = np.sum(right_part)

            if left_intensity > right_intensity:
                return "RIGHT"
            elif right_intensity > left_intensity:
                return "LEFT"
            else:
                return "CENTER"

        while True:
            ret, image = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye_pos = get_eye_position([36, 37, 38, 39, 40, 41], shape, gray)
                right_eye_pos = get_eye_position([42, 43, 44, 45, 46, 47], shape, gray)

                # تحديد اتجاه النظرة
                if left_eye_pos == "LEFT" and right_eye_pos == "LEFT":
                    gaze_direction = "Looking LEFT"
                elif left_eye_pos == "RIGHT" and right_eye_pos == "RIGHT":
                    gaze_direction = "Looking RIGHT"
                else:
                    gaze_direction = "Looking CENTER"

                cv2.putText(image, gaze_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                total_frames += 1
                if gaze_direction in ["Looking LEFT", "Looking RIGHT"]:
                    cheat_frames += 1

            elapsed_time = time.time() - start_time
            if elapsed_time >= exam_time:
                break

            cv2.imshow("Gaze Detection", image)
            if cv2.waitKey(1) == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        if total_frames == 0:
            return {"status": "error", "message": "لم يتم اكتشاف أي وجه"}

        cheat_percentage = (cheat_frames / total_frames) * 100

        if cheat_percentage > 10:
            if is_sheet:
                return {"status": "sheet detect"}

            return {"status": "cheat warning"}
        else:
            return {"status": "normal"}
