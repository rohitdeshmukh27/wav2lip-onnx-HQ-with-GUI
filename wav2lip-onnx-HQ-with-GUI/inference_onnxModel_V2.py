import subprocess
import platform
import numpy as np
import cv2, os, sys, argparse, audio, shutil
from os import listdir, path
from tqdm import tqdm
from PIL import Image
import onnxruntime
onnxruntime.set_default_logger_severity(3)

# face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256
detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# specific face selector
from faceID.faceID import FaceRecognition
recognition = FaceRecognition('faceID/recognition.onnx')

# arguments
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--final_audio', type=str, help='Filepath of video/audio file to use as final audio source')
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')
parser.add_argument('--hq_output', default=False, action='store_true', help='HQ output')
parser.add_argument('--static', default=False, action='store_true', help='If True, then use only first video frame for inference')
parser.add_argument('--pingpong', default=False, action='store_true', help='pingpong loop if audio is longer than video')
parser.add_argument("--cut_in", type=int, default=0, help="Frame to start inference")
parser.add_argument("--cut_out", type=int, default=0, help="Frame to end inference")
parser.add_argument("--fade", action="store_true", help="Fade in/out")
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument("--enhancer", default='none', choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=10, type=float, help='Amount of face enhancement blending 1 - 10')
parser.add_argument("--sharpen", default=False, action="store_true", help="Slightly sharpen swapped face")
parser.add_argument("--frame_enhancer", action="store_true", help="Use frame enhancer")
parser.add_argument("--face_mask", action="store_true", help="Use face mask")
parser.add_argument("--face_occluder", action="store_true", help="Use occluder face mask")
parser.add_argument('--pads', type=int, default=0, help='Padding top, bottom to adjust best mouth position, move crop up/down, between -15 to 15')
parser.add_argument('--face_mode', type=int, default=0, help='Face crop mode, 0 or 1, rect or square, affects mouth opening')
parser.add_argument('--preview', default=False, action='store_true', help='Preview during inference')

args = parser.parse_args()

    # Fix: Set img_size=256 if checkpoint filename contains 'wav2lip_256'
import os
ckpt_name = os.path.basename(args.checkpoint_path).replace('\\', '/').lower()
if 'wav2lip_256' in ckpt_name:
    args.img_size = 256
else:
    args.img_size = 96

mel_step_size = 16
padY = max(-15, min(args.pads, 15))

device = 'cpu'
if onnxruntime.get_device() == 'GPU':
    device = 'cuda'
print("Running on " + device)

if args.enhancer == 'gpen':
    from enhancers.GPEN.GPEN import GPEN
    gpen256 = GPEN(model_path="enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device)

if args.enhancer == 'codeformer':
    from enhancers.Codeformer.Codeformer import CodeFormer
    codeformer = CodeFormer(model_path="enhancers/Codeformer/codeformerfixed.onnx", device=device)

if args.enhancer == 'restoreformer':
    from enhancers.restoreformer.restoreformer16 import RestoreFormer
    restoreformer = RestoreFormer(model_path="enhancers/restoreformer/restoreformer16.onnx", device=device)

if args.enhancer == 'gfpgan':
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    gfpgan = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)

if args.frame_enhancer:
    from enhancers.RealEsrgan.esrganONNX import RealESRGAN_ONNX
    frame_enhancer = RealESRGAN_ONNX(model_path="enhancers/RealEsrgan/clear_reality_x4.onnx", device=device)

if args.face_mask:
    from blendmasker.blendmask import BLENDMASK
    masker = BLENDMASK(model_path="blendmasker/blendmasker.onnx", device=device)

if args.face_occluder:
    from xseg.xseg import MASK
    occluder = MASK(model_path="xseg/xseg.onnx", device=device)

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def load_model(device):
    model_path = args.checkpoint_path
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if device == 'cuda':
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    
    session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
    return session

def process_video(model, img, size, target_id, crop_scale=1.0):
    ori_img = img
    bboxes, kpss = model.detect(ori_img, input_size=(320,320), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    
    for kps in kpss:
        aimg, mat = get_cropped_head_256(ori_img, kps, size=size, scale=crop_scale)
    return aimg, mat

def select_specific_face(model, spec_img, size, crop_scale=1.0):
    cropped_roi = spec_img  # Use the full image, no cropping
    
    bboxes, kpss = model.detect(cropped_roi, input_size=(320,320), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    
    target_face, mat = get_cropped_head_256(cropped_roi, kpss[0], size=size, scale=crop_scale)
    target_face = cv2.resize(target_face, (112,112))
    target_id = recognition(target_face)[0].flatten()
    return target_id

def process_video_specific(model, img, size, target_id, crop_scale=1.0):
    ori_img = img
    bboxes, kpss = model.detect(ori_img, input_size=(320, 320), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    
    best_score = -float('inf')
    best_aimg = None
    best_mat = None
    
    for kps in kpss:
        aimg, mat = get_cropped_head_256(ori_img, kps, size=size, scale=crop_scale)
        face = aimg.copy()
        face = cv2.resize(face, (112, 112))
        face_id = recognition(face)[0].flatten()
        score = target_id @ face_id
        
        if score > best_score:
            best_score = score
            best_aimg = aimg
            best_mat = mat
        if best_score < 0.4:
            best_aimg = np.zeros((256,256), dtype=np.uint8)
            best_aimg = cv2.cvtColor(best_aimg, cv2.COLOR_GRAY2RGB)/255
            best_mat = np.float32([[1,2,3],[1,2,3]])
    return best_aimg, best_mat

def face_detect(images, target_id):
    os.system('cls')
    print("Detecting face and generating data...")
    
    crop_size = 256
    sub_faces = []
    crop_faces = []
    matrix = []
    face_error = []
    
    for i in tqdm(range(0, len(images))):
        try:
            crop_face, M = process_video_specific(detector, images[i], 256, target_id, crop_scale=1.0)
            if args.face_mode == 0:
                sub_face = crop_face[65-(padY):241-(padY),62:194]
            else:
                sub_face = crop_face[65-(padY):241-(padY),42:214]
            
            sub_face = cv2.resize(sub_face, (args.img_size, args.img_size))
            sub_faces.append(sub_face)
            crop_faces.append(crop_face)
            matrix.append(M)
            no_face = 0
        except:
            if i == 0:
                crop_face = np.zeros((256,256), dtype=np.uint8)
                crop_face = cv2.cvtColor(crop_face, cv2.COLOR_GRAY2RGB)/255
                sub_face = crop_face[65-(padY):241-(padY),62:194]
                sub_face = cv2.resize(sub_face, (args.img_size, args.img_size))
                M = np.float32([[1,2,3],[1,2,3]])
            
            sub_faces.append(sub_face)
            crop_faces.append(crop_face)
            matrix.append(M)
            no_face = -1
        face_error.append(no_face)
    
    return crop_faces, sub_faces, matrix, face_error

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch = [], [], []
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        frame_batch.append(frame_to_save)
        img_batch.append(frames[idx])
        mel_batch.append(m)
        
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch
        img_batch, mel_batch, frame_batch = [], [], []

def main():
    # Ensure temp directory exists
    os.makedirs('temp', exist_ok=True)
    
    if args.hq_output:
        if not os.path.exists('hq_temp'):
            os.mkdir('hq_temp')
    
    preset = 'medium'
    blend = args.blending / 10
    
    static_face_mask = np.zeros((224,224), dtype=np.uint8)
    static_face_mask = cv2.ellipse(static_face_mask, (112,162), (62,54), 0, 0, 360, (255,255,255), -1)
    static_face_mask = cv2.ellipse(static_face_mask, (112,122), (46,23), 0, 0, 360, (0,0,0), -1)
    static_face_mask = cv2.resize(static_face_mask, (256,256))
    static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB) / 255
    static_face_mask = cv2.GaussianBlur(static_face_mask, (29,29), cv2.BORDER_DEFAULT)
    
    sub_face_mask = np.zeros((256,256), dtype=np.uint8)
    sub_face_mask = cv2.rectangle(sub_face_mask, (66,69), (190,240), (255,255,255), -1)
    sub_face_mask = cv2.GaussianBlur(sub_face_mask.astype(np.uint8), (9,9), cv2.BORDER_DEFAULT)
    sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB)
    sub_face_mask = sub_face_mask / 255
    
    #im = cv2.imread(args.face)
    
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        # image branch
        orig_frame = cv2.imread(args.face)
        orig_frame = cv2.resize(orig_frame, (orig_frame.shape[1]//args.resize_factor, orig_frame.shape[0]//args.resize_factor))
        orig_frames = [orig_frame]
        fps = args.fps
        h, w = orig_frame.shape[:-1]
        # Use the full image, no cropping
        cropped_roi = orig_frame
        full_frames = [cropped_roi]
        orig_h, orig_w = cropped_roi.shape[:-1]
        target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)

    else:
        # video branch
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        video_stream.set(1, args.cut_in)
        print('Reading video frames...')
        
        if args.cut_out == 0:
            args.cut_out = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - args.cut_in
        new_duration = args.cut_out - args.cut_in
        
        if args.static:
            new_duration = 1
        
        video_stream.set(1, args.cut_in)
        
        full_frames = []
        orig_frames = []
        
        for l in range(new_duration):
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
            
            if l == 0:
                h, w = frame.shape[:-1]
                cropped_roi = frame  # Use the full frame, no cropping
                os.system('cls')
                target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
                orig_h, orig_w = cropped_roi.shape[:-1]
                print("Reading frames....")
            print(f'\r{l}', end=' ', flush=True)
            
            cropped_roi = frame  # Use the full frame, no cropping
            full_frames.append(cropped_roi)
            orig_frames.append(cropped_roi)
    
    memory_usage_bytes = sum(frame.nbytes for frame in full_frames)
    memory_usage_mb = memory_usage_bytes / (1024**2)
    print("Number of frames used for inference: " + str(len(full_frames)) + " / ~ " + str(int(memory_usage_mb)) + " mb memory usage")
    
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = f'ffmpeg -y -i "{args.audio}" -strict -2 "temp/temp.wav"'
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'
    
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    print("Length of mel chunks: {}".format(len(mel_chunks)))
    full_frames = full_frames[:len(mel_chunks)]
    
    aligned_faces, sub_faces, matrix, no_face = face_detect(full_frames, target_id)
    
    if args.pingpong:
        orig_frames = orig_frames + orig_frames[::-1]
        full_frames = full_frames + full_frames[::-1]
        aligned_faces = aligned_faces + aligned_faces[::-1]
        sub_faces = sub_faces + sub_faces[::-1]
        matrix = matrix + matrix[::-1]
        no_face = no_face + no_face[::-1]
    
    gen = datagen(sub_faces.copy(), mel_chunks)
    fc = 0
    model = load_model(device)
    
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))
    
    os.system('cls')
    print('Running on ' + onnxruntime.get_device())
    print('Checkpoint: ' + args.checkpoint_path)
    print('Resize factor: ' + str(args.resize_factor))
    if args.pingpong: print('Use pingpong')
    if args.enhancer != 'none': print('Use ' + args.enhancer)
    if args.face_mask: print('Use face mask')
    if args.face_occluder: print('Use occlusion mask')
    print('')
    
    fade_in = 11
    total_length = int(np.ceil(float(len(mel_chunks))))
    fade_out = total_length - 11
    bright_in = 0
    bright_out = 0
    
    for i, (img_batch, mel_batch, frames) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)))))):
        if fc == len(full_frames):
            fc = 0
        
        face_err = no_face[fc]
        img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
        mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)
        
        pred = model.run(None, {'mel_spectrogram': mel_batch, 'video_frames': img_batch})[0][0]
        pred = pred.transpose(1, 2, 0) * 255
        pred = pred.astype(np.uint8)
        pred = pred.reshape((1, args.img_size, args.img_size, 3))
        
        mat = matrix[fc]
        mat_rev = cv2.invertAffineTransform(mat)
        aligned_face = aligned_faces[fc]
        aligned_face_orig = aligned_face.copy()
        p_aligned = aligned_face.copy()
        full_frame = full_frames[fc]
        final = orig_frames[fc]
        
        for p, f in zip(pred, frames):
            if not args.static: fc = fc + 1
            
            if args.face_mode == 0:
                p = cv2.resize(p, (132,176))
            else:
                p = cv2.resize(p, (172,176))
            
            if args.face_mode == 0:
                p_aligned[65-(padY):241-(padY), 62:194] = p
            else:
                p_aligned[65-(padY):241-(padY), 42:214] = p
            
            aligned_face = (sub_face_mask * p_aligned + (1 - sub_face_mask) * aligned_face_orig).astype(np.uint8)
            
            if face_err != 0:
                res = full_frame
                face_err = 0
            else:
                if args.enhancer == 'gpen':
                    aligned_face = cv2.resize(aligned_face, (256,256))
                    aligned_face_enhanced = gpen256.enhance(aligned_face)
                    aligned_face_enhanced = cv2.resize(aligned_face_enhanced, (256,256))
                    aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32), blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
                
                if args.enhancer == 'codeformer':
                    aligned_face_enhanced = codeformer.enhance(aligned_face, 1.0)
                    aligned_face_enhanced = cv2.resize(aligned_face_enhanced, (256,256))
                    aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32), blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
                
                if args.enhancer == 'restoreformer':
                    aligned_face_enhanced = restoreformer.enhance(aligned_face)
                    aligned_face_enhanced = cv2.resize(aligned_face_enhanced, (256,256))
                    aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32), blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
                
                if args.enhancer == 'gfpgan':
                    aligned_face_enhanced = gfpgan.enhance(aligned_face)
                    aligned_face_enhanced = cv2.resize(aligned_face_enhanced, (256,256))
                    aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32), blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
                
                if args.face_mask:
                    seg_mask = masker.mask(aligned_face)
                    seg_mask = cv2.blur(seg_mask, (5,5))
                    seg_mask = seg_mask / 255
                    mask = cv2.warpAffine(seg_mask, mat_rev, (frame_w, frame_h))
                
                if args.face_occluder:
                    try:
                        seg_mask = occluder.mask(aligned_face_orig)
                        seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                        mask = cv2.warpAffine(seg_mask, mat_rev, (frame_w, frame_h))
                    except:
                        seg_mask = occluder.mask(aligned_face)
                        seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                        mask = cv2.warpAffine(seg_mask, mat_rev, (frame_w, frame_h))
                
                if not args.face_mask and not args.face_occluder:
                    mask = cv2.warpAffine(static_face_mask, mat_rev, (frame_w, frame_h))
                
                if args.sharpen:
                    aligned_face = cv2.detailEnhance(aligned_face, sigma_s=1.3, sigma_r=0.15)
                
                dealigned_face = cv2.warpAffine(aligned_face, mat_rev, (frame_w, frame_h))
                res = (mask * dealigned_face + (1 - mask) * full_frame).astype(np.uint8)
        
        final = res
        if args.frame_enhancer:
            final = frame_enhancer.enhance(final)
            final = cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        
        if i < 11 and args.fade:
            final = cv2.convertScaleAbs(final, alpha=0 + (0.1 * bright_in), beta=0)
            bright_in = bright_in + 1
        if i > fade_out and args.fade:
            final = cv2.convertScaleAbs(final, alpha=1 - (0.1 * bright_out), beta=0)
            bright_out = bright_out + 1
        
        if args.hq_output:
            cv2.imwrite(os.path.join('./hq_temp', '{:0>7d}.png'.format(i)), final)
        else:
            out.write(final)
        
        if args.preview:
            cv2.imshow("Result - press ESC to stop and save", final)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                out.release()
                break
            if k == ord('s'):
                args.sharpen = not args.sharpen
                print('')
                print("Sharpen = " + str(args.sharpen))
    
    out.release()
    
    if args.final_audio:
        if args.hq_output:
            command = 'ffmpeg.exe -y -i ' + '"' + args.final_audio + '"' + ' -r ' + str(fps) + ' -f image2 -i ' + '"' + './hq_temp/' + '%07d.png' + '"' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -crf 5 -preset slow -acodec aac -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'
        else:
            command = 'ffmpeg.exe -y -i ' + '"' + args.final_audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -acodec aac -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        
        if os.path.exists('temp/temp.mp4'):
            os.remove('temp/temp.mp4')
        if os.path.exists('hq_temp'):
            shutil.rmtree('hq_temp')
    else:
        if args.hq_output:
            command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -r ' + str(fps) + ' -f image2 -i ' + '"' + './hq_temp/' + '%07d.png' + '"' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -crf 5 -preset slow -acodec aac -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'
        else:
            command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -acodec aac -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'
        
        subprocess.call(command, shell=platform.system() != 'Windows')
        
        if os.path.exists('temp/temp.mp4'):
            os.remove('temp/temp.mp4')
        if os.path.exists('hq_temp'):
            shutil.rmtree('hq_temp')

if __name__ == '__main__':
    main()
