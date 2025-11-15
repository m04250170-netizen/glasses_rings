import cv2
from ultralytics import YOLO
import time
import argparse
import os


def multi_model_detection(video_source=0):

    try:
        person_model = YOLO('models/glasses_rings.pt')

    except Exception as e:
        print(f"模型加载错误: {e}")
        return


    source_type = "摄像头"
    if isinstance(video_source, str):
        if video_source.startswith('rtsp://') or video_source.startswith('http://'):
            source_type = "RTSP流"
        elif os.path.exists(video_source):
            source_type = "视频文件"
        else:
            print(f"错误: 视频源 '{video_source}' 不存在或格式不支持")
            return

    print(f"使用{source_type}: {video_source}")

    if source_type == "RTSP流":

        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        cap.open(video_source)
    else:
        cap = cv2.VideoCapture(video_source)


    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_source}")
        return


    if source_type in ["视频文件", "RTSP流"]:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if source_type == "视频文件":
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"视频信息: {width}x{height}, FPS: {fps:.2f}, 总帧数: {total_frames}")
        else:
            print(f"RTSP流信息: {width}x{height}, FPS: {fps:.2f}")

    print("按 'Q' 退出")
    print("按 'P' 暂停/继续")
    print("按 'R' 重新连接（RTSP流）")

    paused = False
    frame_count = 0
    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if source_type == "视频文件":

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("视频播放结束，重新开始...")
                    continue
                elif source_type == "RTSP流":

                    print("RTSP流断开，尝试重新连接...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(video_source)
                    if not cap.isOpened():
                        print("重新连接失败，继续尝试...")
                        time.sleep(5)
                        continue
                    print("重新连接成功")
                    continue
                else:
                    print("无法读取帧，退出...")
                    break

            frame_count += 1


            person_results = person_model(frame, conf=0.5)


            annotated_frame = person_results[0].plot()


            for box in person_results[0].boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = person_model.names[cls_id]


                if cls_name == 'person':
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()


                    face_y1 = int(y1)
                    face_y2 = int(y1 + (y2 - y1) * 0.4)
                    face_x1 = int(x1)
                    face_x2 = int(x2)


                    cv2.rectangle(annotated_frame, (face_x1, face_y1), (face_x2, face_y2), (255, 255, 0), 1)




            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = frame_count / elapsed_time
                fps_text = f'FPS: {current_fps:.1f}'
            else:
                fps_text = 'FPS: Calculating...'


            cv2.putText(annotated_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # cv2.putText(annotated_frame, f'Source: {source_type}', (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


            if source_type == "视频文件":
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.putText(annotated_frame, f'Frame: {current_frame}/{total_frames}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Multi-Model Detection', annotated_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord('r') and source_type == "RTSP流":

            print("手动重新连接RTSP流...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                print("重新连接成功")
            else:
                print("重新连接失败")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='多模型目标检测')
    parser.add_argument('--video', type=str, help='视频文件路径', default=None)
    parser.add_argument('--rtsp', type=str, help='RTSP流地址', default=None)
    parser.add_argument('--camera', type=int, help='摄像头索引', default=0)

    args = parser.parse_args()

    if args.rtsp:
        video_source = args.rtsp
    elif args.video:
        video_source = args.video
    else:
        video_source = args.camera

    multi_model_detection(video_source)


if __name__ == "__main__":
    main()