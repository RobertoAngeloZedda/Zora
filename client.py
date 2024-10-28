from tcp_communication import connect, send_file, receive_file
from multiprocessing import Process, Manager
import time
import sounddevice as sd
import wave

SERVER_ADDRESS = '192.168.1.4'
PORT = 12345

SAMPLERATE = 16000
DURATION = 1.5
CHANNELS = 1

UPDATES_PER_SECOND = 20

def capture_audio_chunk(duration, sample_rate):
    #print(f'Recording {duration} seconds of audio..')

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='int16')
    sd.wait()
    return audio
    
def save_audio(audio, file_name):
    audio = audio.tobytes()

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes(audio)
    
    #print(f'Audio saved at: {file_name}')

def manage_audio_recording(audio_queue):
    print('\n\nStart\n\n')
    try:
        while True:
            audio = capture_audio_chunk(DURATION, SAMPLERATE)
            audio_queue.append(audio)
    except KeyboardInterrupt:
        print('\n\nEnd.\n')
        exit()

def server_communication(audio_queue):
    try:
        while True:
            if audio_queue:
                new_audio = audio_queue.pop(0)
                file_name = f'temp/temp.wav'
                save_audio(new_audio, file_name)

                socket = connect(SERVER_ADDRESS, PORT)

                send_file(socket, file_name)

                is_response_ready = socket.recv(1) == b'1'

                if is_response_ready:
                    handle_response(socket)

                    # clear audios registered during response
                    audio_queue[:] = []
                else:
                    socket.close()
            else:
                time.sleep(1/UPDATES_PER_SECOND)
    except KeyboardInterrupt:
        exit()

def handle_response(socket):
    receive_file(socket, 'temp/temp.txt')
    print('Response Received')

    socket.close()

    actions = []
    sentences = []
    with open('temp/temp.txt', 'r') as file:
        for index, line in enumerate(file.readlines()):
            if index%2:
                sentences.append(line[:-1])
            else:
                actions.append(line[:-1])

    for index in range(len(sentences)):
        print(f'\nAction: {actions[index]}\nSentence: {sentences[index]}')
    print('\n')

if __name__ == '__main__':

    manager = Manager()
    audio_queue = manager.list()
    responses_queue = manager.list()

    p1 = Process(target=manage_audio_recording, args=(audio_queue, ))
    p2 = Process(target=server_communication, args=(audio_queue, ))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()