import os.path
import shutil
import tkinter as tk
from tkinter import filedialog, END
import tkinter.messagebox as msgbox
import threading

from PIL import Image
from dlib import get_frontal_face_detector
from tensorflow.keras.models import load_model
from setting import model_path
from tool import is_ok, cv2

patience = 15


def print_to_info(content):
    global info
    info.insert(END, content)
    info.insert(END, "\n")


def upload_file():
    global selectFile
    selectFile = tk.filedialog.askopenfilename()
    video_url.delete("1.0", END)
    video_url.insert("0.0", selectFile)


def progress():
    ok_img = []
    person_num = person_num_input.get("1.0", "end")

    try:
        person_num = int(person_num)
    except ValueError:
        msgbox.showerror('输入错误', '合影人数输入有误')
        return False

    if os.path.exists(selectFile) is not True:
        msgbox.showerror('文件错误', '视频文件不存在')
        msgbox.showerror('文件错误', '视频文件不存在或文件格式有误')

    cap = cv2.VideoCapture(selectFile)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0.0:  # 使用cv2模块获取fps,fps为0，即损坏
        msgbox.showerror('文件错误', '视频文件不存在或文件格式有误')
        return False

    print_to_info("正在处理......")
    print_to_info("视频打开成功,正在读取抓拍....")

    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success is not True:
            break
        if frame_id % patience == 0:
            print_to_info("正在读取第{}帧".format(frame_id).format(len(ok_img)))
            ok, tot = is_ok(frame=frame, model=model, detector=detector, person_num=person_num)
            if ok and tot > 0.85:  # 获取结果
                print_to_info("于第{}帧成功抓拍".format(frame_id).format(len(ok_img)))
                ok_img.append({
                    "img": frame,
                    "score": tot
                })
        frame_id += 1

    cap.release()
    print_to_info("视频读取完毕,成功抓拍{}张合照".format(len(ok_img)))
    print_to_info("照片已保存到原视频相同路径下")
    save_imgs(ok_img)
    show_imgs(imgs=ok_img)


def save_imgs(imgs):
    global selectFile
    save_root = selectFile + "-res"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root)
    for index, img in enumerate(imgs):
        image = Image.fromarray(cv2.cvtColor(img['img'], cv2.COLOR_BGR2RGB))
        image.save(os.path.join(save_root, str(index) + ".png"))


def show_imgs(imgs):
    for index, img in enumerate(imgs):
        cv2.imshow(str(index) + "-" + str(img['score']), img['img'])
    cv2.waitKey()


def run():
    global selectFile
    threading.Thread(target=progress).start()


if __name__ == "__main__":
    selectFile = ""

    # 加载人脸检测器
    detector = get_frontal_face_detector()
    # 加载模型
    model = load_model(model_path)

    root = tk.Tk()
    root.title('基于vgg卷积神经网络深度学习的多人合照抓拍算法')
    root.resizable(width=False, height=False)

    width = 50

    frm = tk.Frame(root)
    frm.grid(padx=8, pady=8)
    video_url = tk.Text(frm, height=2, width=width)
    video_url.grid(row=0, column=0, padx=5)
    video_url.insert("0.0", "请选择视频")

    person_num_input = tk.Text(frm, height=2, width=15)
    person_num_input.grid(row=0, column=1)
    person_num_input.insert("0.0", "请输入合影人数")

    upload_btn = tk.Button(frm, text='选择视频', command=upload_file, width=14)
    upload_btn.grid(row=1, column=1, padx=5)

    run_btn = tk.Button(frm, text='开始运行', command=run, width=14)
    run_btn.grid(row=2, column=1, padx=5)

    info = tk.Text(frm, height=25, width=width)
    info.grid(row=1, column=0, rowspan=25, pady=5)

    au_info = tk.Text(frm, height=3, width=width)
    au_info_text = "作者:\n  技术实现:何东毅\n  测试:魏嫄珂,李世洋"
    au_info.insert("0.0", au_info_text)
    au_info.grid(row=26, column=0)

    root.mainloop()
