"""Two-phone sync from the audio track (CPU, .venv-spider: needs scipy), plus frame helpers for review sheets.

compare(onsets_head, onsets_second, 100) -> offset s such that second-phone time = head-phone time + offset.
"""
import json
import hashlib
import subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import correlate, correlation_lags

ROOT=Path(__file__).resolve().parents[2]

def probe(path):
    info=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)],text=True))
    video=next(s for s in info["streams"] if s["codec_type"]=="video")
    return dict(source=str(path.relative_to(ROOT)),width=video["width"],height=video["height"],
                fps=video["r_frame_rate"],duration_s=float(info["format"]["duration"]),
                audio=any(s["codec_type"]=="audio" for s in info["streams"]),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def audio(path):
    data=subprocess.check_output(["ffmpeg","-v","error","-threads","2","-i",str(path),"-vn","-ac","1","-ar","8000","-f","f32le","-"])
    x=np.frombuffer(data,np.float32)
    hop=80
    pad=(-len(x))%hop
    rms=np.sqrt(np.mean(np.pad(x,(0,pad)).reshape(-1,hop)**2,axis=1))
    onset=np.maximum(np.diff(rms,prepend=rms[0]),0)
    onset[:30]=0
    return x,onset

def compare(a,b,hz,max_lag=4):
    corr=correlate(b,a,mode="full",method="fft")
    lags=correlation_lags(len(b),len(a))
    allowed=np.abs(lags)<=max_lag*hz
    score=corr/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
    selected=np.where(allowed,score,-np.inf)
    k=int(np.argmax(selected))
    return float(lags[k]/hz),float(score[k])

def frame(cap,t):
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
    ok,img=cap.read()
    if not ok:raise RuntimeError("Cannot decode t=%.3f"%t)
    return img

def fit(img,w=320,h=180):
    scale=min(w/img.shape[1],h/img.shape[0])
    resized=cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    out=np.zeros((h,w,3),np.uint8)
    y=(h-resized.shape[0])//2;x=(w-resized.shape[1])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return out

def tile(img,label,w=320,h=180):
    out=np.zeros((h+24,w,3),np.uint8)
    out[24:]=fit(img,w,h)
    cv2.putText(out,label,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
    return out
