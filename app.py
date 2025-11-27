from flask import Flask, request, jsonify
from servises import FaceDetection

app = Flask(__name__)

ROUTE_PREFIX = '/ai'
# http://localhost:5000/ai/collect_faces?id=XXXX&name=XXXX
@app.route(ROUTE_PREFIX + "/collect_faces", methods=["GET"])
def collect_faces_api():
    user_id = request.args.get("id")
    name = request.args.get("name")

    if not user_id or not name:
        return jsonify({"status": "error", "message": "Both 'id' and 'name' are required"}), 400

    try:
        result = FaceDetection.collect_faces(user_id, name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# http://localhost:5000/ai/recognize_faces
@app.route(ROUTE_PREFIX + "/recognize_faces", methods=["GET"])
def recognize_faces_api():
    try:
        result = FaceDetection.recognize_and_mark_attendance()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

# http://localhost:5000/ai/face_traking?time=10
@app.route(ROUTE_PREFIX + "/face_traking", methods=["GET"])
def face_traking_api():
    is_sheet = request.args.get("is_sheet", default="false").lower() == "true"
    exam_time = request.args.get("time", default=0, type=int)
    try:
        result = FaceDetection.face_traking(is_sheet, exam_time)   
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# http://localhost:5000/ai/reset_data
@app.route(ROUTE_PREFIX + "/reset_data", methods=["GET"])
def reset_data():
    try:
        FaceDetection.clear_training_data()
        return jsonify({"status": "success", "message": "All face data cleared."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
