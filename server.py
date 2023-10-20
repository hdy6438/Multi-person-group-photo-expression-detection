import os
import threading

from dlib import get_frontal_face_detector
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model

from setting import model_path, api_port
from tool import is_ok, cv2, cv2_base64, generate_random_str

# 加载人脸检测器
detector = get_frontal_face_detector()
# 加载模型
model = load_model(model_path)

# 创建应用
app = Flask(__name__)
# 跨域
CORS(app, supports_credentials=True)

tasks = {}


def a(task_video_path, task_id, person_num):
    print(task_id, 'begin')
    cap = cv2.VideoCapture(task_video_path)  # 读取上传视频
    ok_img = []
    patience = 15
    frame_id = 0
    while True:
        success, frame = cap.read()
        if success is not True:
            cap.release()
            break

        if frame_id % patience == 0:
            ok, tot = is_ok(frame=frame, model=model, detector=detector, person_num=person_num)
            if ok:  # 获取结果
                ok_img.append({
                    "img": cv2_base64(image=frame).decode(),
                    "score": tot
                })
        frame_id += 1

    os.remove(task_video_path)
    tasks[task_id] = ok_img
    print(task_id, 'over')


@app.route('/api', methods=["POST"])
def detection_from_video():
    file = request.files.get("file")
    task_id = generate_random_str(12)
    person_num = int(request.form.get('person_num'))
    if file.content_type != "video/mp4":  # 判断文件类型
        return jsonify({
            'code': 500,
            'msg': 'error'
        })
    else:
        task_path = os.path.join('upload', task_id + '.mp4')
        file.save(task_path)  # 保存上传文件到服务器
        threading.Thread(target=a, args=(task_path, task_id, person_num,)).start()
        return jsonify({
            'code': 200,
            'task_id': task_id
        })


@app.route('/get_res', methods=["POST"])
def get_res():
    tid = request.form.get('task_id')
    if tid in tasks.keys():
        data = {
            'code': 200,
            'data': tasks[tid]
        }
        return jsonify(data)
    else:
        return jsonify({
            'code': 200,
            'data': 'no'
        })


server = WSGIServer(('0.0.0.0', api_port), app)
server.serve_forever()
