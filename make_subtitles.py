import collections
import contextlib
import sys
import wave
import os
import webrtcvad
import speech_recognition as sr
import moviepy.editor as mpe
import subprocess
import numpy as np




FRAME_DURATION_IN_MS = 30
MAX_DELAY_IN_MS = 300
MAX_SPEECH_IN_S = 10
MAX_SILENCE_PADDING_IN_FRAMES = 10
vad = webrtcvad.Vad(3)
max_speech_in_frames = int(MAX_SPEECH_IN_S / (FRAME_DURATION_IN_MS / 1000))
max_delay_in_frames = MAX_DELAY_IN_MS // FRAME_DURATION_IN_MS
temp_wav_name = 'speech.wav'
recognizer = sr.Recognizer()


def read_wave(path):

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):

    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp):
        self.bytes = bytes
        self.timestamp = timestamp
        self.is_speech = False


def frame_generator(frame_duration_ms, audio, sample_rate):

    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, vad, frames, max_delay_in_frames):

    speech = list()
    cur_delay_in_frames = 0
    cur_start_silence_padding_in_frames = None
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if is_speech:
            frame.is_speech = True
            if cur_start_silence_padding_in_frames is None:
                cur_start_silence_padding_in_frames = cur_delay_in_frames
            speech.append(frame)
            cur_delay_in_frames = 0
        elif not is_speech and cur_delay_in_frames == max_delay_in_frames:
            speech_len = len(speech)
            if speech_len > max_delay_in_frames:

                start_pad_diff = (cur_start_silence_padding_in_frames - MAX_SILENCE_PADDING_IN_FRAMES)
                end_pad_diff = (max_delay_in_frames - MAX_SILENCE_PADDING_IN_FRAMES)
                start_pad_in_frames = start_pad_diff * (start_pad_diff > 0)
                end_pad_diff_in_frames = speech_len - end_pad_diff 
                yield speech[start_pad_in_frames:end_pad_diff_in_frames]
            speech.clear()
            cur_delay_in_frames = 0
            cur_start_silence_padding_in_frames = None
        elif not is_speech:
            speech.append(frame)
            cur_delay_in_frames += 1
    has_speech = any([frame.is_speech for frame in speech])

    if has_speech:            
        yield speech


def generate_time_range(first_frame, last_frame):
        times = first_frame.timestamp, last_frame.timestamp
        t_hours = [f'{int(t / 3600):02}' for t in times]
        t_mins = [f'{int(t / 60):02}' for t in times]
        t_secs = [f'{int(t % 60):02}' for t in times]
        t_ms = [int((times[i] - int(times[i])) * 100) for i in range(2)]
        time_range = [f'{t_hours[i]}:{t_mins[i]}:{t_secs[i]},{t_ms[i]}' for i in range(2)]
        return time_range

def transcribe_speech(recognizer, wav_path):
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)  
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = ''
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text

def get_segment_duration(segment):
    return segment[-1].timestamp - segment[0].timestamp

def create_subtitles_for_wav(wav_name):
    audio, sample_rate = read_wave(wav_name)
    frames = frame_generator(FRAME_DURATION_IN_MS, audio, sample_rate)
    subtitles = open(f'{wav_name.split(".")[0][1:]}_subtitles.srt', 'w')

    segments = vad_collector(sample_rate, vad, frames, max_delay_in_frames)
    index = 0
    for i, segment in enumerate(segments):

        cropped_segments = list()
        if get_segment_duration(segment) > MAX_SPEECH_IN_S:
            frames_num = len(segment)
            batches_num = frames_num // max_speech_in_frames
            batches_num = batches_num + 1 if frames_num > batches_num * max_speech_in_frames else batches_num
            for i in range(batches_num):
                cropped_segments.append(segment[i*max_speech_in_frames:(i+1)*max_speech_in_frames])
        else:
            cropped_segments.append(segment)

        for j, segm in enumerate(cropped_segments):

            audio = b''.join([frame.bytes for frame in segm])
            #write_wave(f'speech_chunks/chunk_{index}.wav', audio, sample_rate)
            write_wave(temp_wav_name, audio, sample_rate)
            time_range = generate_time_range(segm[0], segm[-1])
            text = transcribe_speech(recognizer, temp_wav_name)
            if text:
                print(time_range, text)
                segm = list(filter(lambda seg: seg.is_speech, segm))

                start_time = segm[0].timestamp
                finish_time = segm[-1].timestamp

            subtitles.write(f'{index+1}\n')
            subtitles.write(f'{time_range[0]} --> {time_range[1]}\n')
            subtitles.write(f'{text}\n')
            subtitles.write('\n')
            index += 1

    subtitles.close()


def normalize(seq:np.ndarray):
    max_v = max(seq)
    return [num / max_v * 2 - 1 for num in seq]



def standardize(seq:np.ndarray):
    mean = seq.mean()
    variance = seq.var()
    return (seq - mean) / np.sqrt(variance)

def convert_audio(audio_path):
    audio = mpe.AudioFileClip(audio_path)
    audio_path = f'~{audio_path.split("/")[-1].split(".")[0]}.wav'
    audio.write_audiofile(audio_path, fps=16000, nbytes=2, ffmpeg_params=['-ac', '1'])
    return audio_path

def clean_temp_files(wav_name):
    os.remove(wav_name)
    os.remove(temp_wav_name)


def main(args):
    if len(args) != 1:
        sys.stderr.write(
            'Usage: make_subtitles.py <path to audio file>\n')
        sys.exit(1)


    audio_path = args[0]
    converted_wav_name = convert_audio(audio_path)

    create_subtitles_for_wav(converted_wav_name)

    clean_temp_files(converted_wav_name)


if __name__ == '__main__':
    main(sys.argv[1:])

