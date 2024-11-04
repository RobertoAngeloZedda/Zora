from faster_whisper import WhisperModel
import numpy as np
import soundfile
import string

#import webrtcvad
import torch

import logging

def rmv_punct(text):
        return ''.join([char for char in text if char not in string.punctuation])

class Transcriptor():

    #def __init__(self, vad_level=3):
    def __init__(self):
        #self.model = WhisperModel('small.en', device='cpu', compute_type='int8')
        self.model = WhisperModel('base.en', device='cpu', compute_type='int8')

        #self.vad = webrtcvad.Vad(vad_level)
        self.vad, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad')
        self.get_speech_timestamps, _, _, _, _ = utils

        self.status = 'waiting'
        self.sentences = ['']
        
        logging.basicConfig(
            filename='temp/logging.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.reset_queues()
    
    def reset_queues(self):
        self.audio_queue = np.array([], dtype=np.float32)
        self.temporary_queue = []
        self.confirmed_queue = []
        self.offset = 0

    def transcribe(self, file_path, prompt=''):
        segments, _ = self.model.transcribe(file_path, language='en', beam_size=3, 
                                            word_timestamps=True, initial_prompt=prompt,
                                            condition_on_previous_text=True)

        transcription = []
        for segment in segments:
            for word in segment.words:
                transcription.append((word.start, word.end, word.word))
        
        return transcription

    def update_queues(self, transcription):
        transcription = [(self.offset + start, self.offset + end, word) 
                         for start, end, word in transcription]

        new_words = []
        # as long as the new words match the ones in the temporary queue
        # they are added in the confirmed queue
        while transcription:
            start, end, word = transcription[0]

            if not self.temporary_queue:
                break

            if rmv_punct(word).lower() == rmv_punct(self.temporary_queue[0][2]).lower():
                new_words.append((start, end, word))
                self.offset = end

                transcription.pop(0)
                self.temporary_queue.pop(0)
            else:
                break

        self.confirmed_queue.extend(new_words)

        joined_new_words = ''.join([w for _, _, w in new_words])
        self.sentences[-1] += joined_new_words

        # add the remaining to the temporary queue
        self.temporary_queue = transcription

        return joined_new_words
    
    def open_file(self, file_path, data_type):
        new_audio = None

        # converting file to the format required from the whisper model
        if data_type == np.int16:
            new_audio, _ = soundfile.read(file_path, dtype=data_type)
            new_audio = new_audio.astype(np.float32) / 32768.0

        return new_audio
    
    def open_file_new(self, file_path, data_type):
        new_audio, _ = soundfile.read(file_path, dtype=data_type)

        # convirting into mono
        if len(new_audio.shape):
            new_audio = np.mean(new_audio, axis=1, dtype=data_type)
            #new_audio = new_audio[:, 0]

        # converting file to the format required from the whisper model
        new_audio = new_audio.astype(np.float32) / np.float32(np.iinfo(data_type).max+1)

        return new_audio

    def update(self, file_path, sr=16000, debug=False):
        is_speech = self.is_speech(file_path=file_path, sr=sr)

        if not is_speech:
            # If a new sentence hasnt started yet no further processing is needed
            #if self.temporary_queue == []:
            if self.status == 'waiting' or self.status == 'completed':
                self.status = 'waiting'
                if debug: logging.debug(f'File: {file_path} has no speech. (waiting for new sentence)\n')
                return ''
            # In case of "silence" at the End of a Sentence
            # the previous processing is needed 
            # to allow the words stuck in the temp_queue to be confirmed.
            else:
                self.status = 'completed'
                if debug: logging.debug(f'File: {file_path} has no speech. (the previous sentence has ended)')
        
        # The current audio is added to the queue only if needed
        else:
            self.status = 'listening'

            new_audio = self.open_file(file_path, np.int16)
            if debug: logging.debug(f'File: {file_path} has been correctly opened. (size: {len(new_audio)}, len: {len(new_audio)/sr} sec.)')

            self.audio_queue = np.concatenate((self.audio_queue, new_audio), dtype=np.float32)
        
        if debug: logging.debug(f'Audio_queue: (size: {len(self.audio_queue)}, len: {(len(self.audio_queue)/sr):.2f} sec., starts at: {self.offset:.2f} sec.)')

        # giving context for a more accurate transcription
        prompt = self.sentences[-1]
        transcription = self.transcribe(self.audio_queue, prompt=prompt)
        
        if debug: logging.debug([f'{word} ({self.offset+start:.2f}, {self.offset+end:.2f})' 
                         for start, end, word in transcription])

        # need to save offset before it changes to correctly trim the audio queue
        old_offset = self.offset
        new_words = self.update_queues(transcription)
        
        if debug: logging.debug(f'Updated transcription: \'{self.sentences[-1]}\'')

        # calculating how much audio needs to be trimmed
        cut = int((self.offset - old_offset) * sr)
        if debug: logging.debug(f'Last word ended at: {self.offset:.2f} so we must cut the first {(cut/sr):.2f} seconds.\n')
        
        self.audio_queue = self.audio_queue[cut:]

        if not is_speech:
            self.new_sentence()

        return new_words

    def is_speech(self, file_path, sr=16000):
        audio, _ = soundfile.read(file_path, dtype=np.int16)

        speech_chunks = self.get_speech_timestamps(audio, self.vad, sampling_rate=sr)
        
        return (len(speech_chunks) != 0)

    '''def is_speech(self, file_path, sr=16000):
        audio, _ = soundfile.read(file_path, dtype=np.int16)

        while len(audio) > 0:
            lenght = min(int(30*sr/1000), len(audio))
            chunk = audio[:lenght]
            chunk = chunk.tobytes()

            if self.vad.is_speech(chunk, sample_rate=sr):
                return True
            
            audio = audio[lenght:]
        
        return False'''

    def new_sentence(self):
        self.reset_queues()

        if self.sentences[-1] != '':
            self.sentences.append('')

    def get_last_complete_sentence(self):
        if len(self.sentences) >= 2:
            return self.sentences[-2] 
        return ''

if __name__ == '__main__':

    import wave
    def save_audio(audio, file_name):
        audio = audio.tobytes()

        with wave.open(file_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)
            wf.setframerate(8000)
            wf.writeframes(audio)

    def open_file_new(file_path, data_type):
        
        new_audio, _ = soundfile.read(file_path, dtype=data_type)

        # convirting into mono
        if len(new_audio.shape) > 1:
            #new_audio = np.mean(new_audio, axis=1, dtype=data_type)
            new_audio = new_audio[:, 1]

        # converting file to the format required from the whisper model
        new_audio = new_audio.astype(np.float32) / 2147483647.0

        return new_audio
    
    #print(np.iinfo(np.int32).max)
    #exit()

    audio = open_file_new('temp/stereo.wav', np.int32)
    print(len(audio)/8000)

    model = WhisperModel('base.en', device='cpu', compute_type='int8')

    segments, _ = model.transcribe(audio, language='en', beam_size=3, word_timestamps=True, condition_on_previous_text=True)

    transcription = []
    for segment in segments:
        for word in segment.words:
            transcription.append(word.word)
    
    print(''.join(transcription))
