import time
import argparse
import torch
import numpy as np
import cv2
import onnxruntime

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('--model_path', default='./weights/retinaface_mobilenet25.onnx',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--engine_path', default='retinface_resnet50.trt',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--save_dir', default="./results", type=str, help='Dir to save results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    return parser.parse_args()

def load_model_ort(model_path, engine_path):
    session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            (
                'TensorrtExecutionProvider',
                {
                    'device_id': 0,
                    'trt_max_workspace_size': 2147483648,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '{}'.format(engine_path),
                }
            ),
            (
                'CUDAExecutionProvider', 
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
            )
        ]
    )
    return session

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    
    cfg = None
    args = parse_args()

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    # load onnx model
    retinaface = load_model_ort(args.model_path, args.engine_path)
    print('Finished loading model!')

    resize = 1

    source = cv2.VideoCapture(0)

    while True:
        ret, frame = source.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.float32(frame_rgb)
        orig_height, orig_width, _ = img.shape

        frame = cv2.resize(frame, (640, 640))
        img = cv2.resize(img, (640, 640))
        im_height, im_width, _ = img.shape
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        tic = time.time()
        inputs = {"input": img}
        loc, conf, landms = retinaface.run(None, inputs)
        print('net forward time: {:.4f}s'.format(time.time() - tic))

        tic = time.time()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width), format="numpy")
        priors = priorbox.forward()

        prior_data = priors

        boxes = decode(np.squeeze(loc, axis=0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        scores = np.squeeze(conf, axis=0)[:, 1]

        landms = decode_landm(np.squeeze(landms.data, axis=0), prior_data, cfg['variance'])

        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = landms * scale1 / resize

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print('post processing time: {:.4f}s'.format(time.time() - tic))

        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)

        frame = cv2.resize(frame, (orig_width, orig_height))
        if args.save_image:
            name = "test.jpg"
            cv2.imwrite(name, frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()