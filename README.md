# glasses_rings
眼镜和戒指检测识别

支持更换yolo通用型检测模型，下载好的模型放在models目录下，名称修改成glasses_rings后重新启动

# 1.Download and install:

cd glasses_rings
pip install -r requirements.txt

# 2.Run the program：

# For webcam (default)
python app.py

# For specific RTSP stream
python app.py --stream "rtsp://your-camera-url"

# For video file
python app.py --stream "path/to/video.mp4"

