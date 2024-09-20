import argparse

import cv2 as cv

from vittrack import VitTrack

backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="VIT track opencv API")
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input video. Omit for using default camera.')
parser.add_argument('--model_path', type=str, default='object_tracking_vittrack_2023sep.onnx',
                    help='Usage: Set model path')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true', default=False,
                    help='Usage: Specify to save a file with results.')
parser.add_argument('--vis', '-v', action='store_true', default=True,
                    help='Usage: Specify to open a new window to show results.')
args = parser.parse_args()


def draw_plot_grid():
    frame_height, frame_width = frame.shape[:2]
    step = 47
    for i in range(9):
        cv.line(frame, (85 + step, 5), (85 + step, frame_height - 5), (255, 100, 100), 1)
        cv.line(frame, (85, 5 + step), (frame_width - 85, 5 + step), (255, 100, 100), 1)
        step += 47
    cv.rectangle(frame, (85, 5), (frame_width - 85, frame_height - 5), (255, 0, 0), 2)
    cv.putText(frame, '0', (65, frame_height - 5), cv.FONT_HERSHEY_DUPLEX,
               0.5, (255, 0, 0), 1)
    cv.putText(frame, '100', (frame_width - 78, frame_height - 5), cv.FONT_HERSHEY_DUPLEX,
               0.5, (255, 0, 0), 1)
    cv.putText(frame, '100', (45, 15), cv.FONT_HERSHEY_DUPLEX,
               0.5, (255, 0, 0), 1)

def visualize(image, bbox, score, isLocated, fps=None, box_color=(0, 255, 0), text_color=(0, 255, 0), fontScale=0.7,
              fontSize=1):
    output = image.copy()
    h, w, _ = output.shape

    height, width = output.shape[:2]

    if isLocated and score >= 0.3:
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        cv.line(output, (x, y), (x + w, y + h), box_color, 2)
        cv.line(output, (x + w, y), (x, y + h), box_color, 2)
        print(percent(x + w / 2 - 85), percent(height - (y + h / 2) - 5))
        location = (x, y - 10)
        if percent(height - (y + h / 2) - 5 + h / 2) >= 95 and percent(x + w / 2 - 85) >= 90:
            location = (x - 80, y + h + 28)
        elif percent(height - (y + h / 2) - 5 + h / 2) >= 95:
            location = (x, y + h + 28)
        elif percent(x + w / 2 - 85) >= 90:
            location = (x - 80, y - 10)

        cv.putText(output, f'{percent(x + w / 2 - 85)}cm, {percent(height - (y + h / 2) - 5)}cm', location, cv.FONT_HERSHEY_DUPLEX,
                   fontScale, text_color, fontSize)
    else:
        text_size, baseline = cv.getTextSize('Target lost!', cv.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h - text_size[1]) / 2)
        cv.putText(output, 'Target lost!', (text_x, text_y), cv.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 255), fontSize)

    return output


def percent(num):
    return round((num * 100) / 470, 1)


if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    model = VitTrack(
        model_path=args.model_path,
        backend_id=backend_id,
        target_id=target_id)

    _input = 0 if args.input is None else args.input
    video = cv.VideoCapture(_input)

    has_frame, first_frame = video.read()
    if not has_frame:
        print('No frames grabbed!')
        exit()
    first_frame_copy = first_frame.copy()
    height, width = first_frame_copy.shape[:2]
    roi = (0, 0, 640, 480)


    model.init(first_frame, roi)

    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            print('End of video')
            break
        tm.start()
        isLocated, bbox, score = model.infer(frame)
        tm.stop()

        draw_plot_grid()

        frame = visualize(frame, bbox, score, isLocated, fps=tm.getFPS())

        if args.vis:
            cv.imshow('CVLocator', frame)
        tm.reset()

    video.release()
    cv.destroyAllWindows()
