from multiprocessing import Process, Manager
import time
from transcriptor import Transcriptor
from response_generator import Response_generator
from movement_recognition import Movement_recognition
from tcp_communication import host, receive_file, send_file

SERVER_ADDRESS = '192.168.1.4'
PORT = 12345

SAMPLERATE = 16000

MAX_COUNT = 15
UPDATES_PER_SECOND = 20

def server_listening(audio_queue, responses_queue):
    count = 0
    try:
        while(True):
            socket = host(SERVER_ADDRESS, PORT)

            file_name = f'temp/temp{count}.wav'

            receive_file(socket, file_name)

            audio_queue.append(file_name)
            count += 1 if count != MAX_COUNT else -MAX_COUNT
            
            if responses_queue:
                socket.sendall(b'1')
                
                response_file_path = responses_queue.pop(0)
                send_file(socket, response_file_path)
            else:
                socket.sendall(b'0')

            socket.close()
    except KeyboardInterrupt:
        exit()

def process_data(audio_queue, responses_queue):
    tr = Transcriptor()
    resp_gen = Response_generator()
    mov_rec = Movement_recognition()

    print('Models successfully loaded.')
    
    try:
        while(True):
            if audio_queue:
                new_audio = audio_queue.pop(0)
                transcription = tr.update(new_audio, SAMPLERATE, True)
                print(f'({tr.status}) - {transcription}')

                if tr.status == 'completed':
                    question = tr.get_last_complete_sentence()
                    print('\n\tUser\'s question:\n', question)

                    response = resp_gen.generate_response(question)
                    print('\n\tModel\'s response:\n', response)

                    sentences, actions, references, similarities = mov_rec.detect_actions(response)
                    for index in range(len(sentences)):
                        print('Sentence:', sentences[index])
                        print('Action selected:', actions[index])
                        print('Because it\'s similar to:', references[index])
                        print('Similarity:', similarities[index])
                        print()

                    response_file_path = 'temp/response.txt'
                    with open(response_file_path, 'w') as file:
                        for index in range(len(sentences)):
                            file.write(actions[index])
                            file.write('\n')
                            file.write(sentences[index])
                            file.write('\n')

                    responses_queue.append(response_file_path)

                    audio_queue[:] = []
            else:
                time.sleep(1/UPDATES_PER_SECOND)
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':

    manager = Manager()
    audio_queue = manager.list()
    responses_queue = manager.list()

    p1 = Process(target=server_listening, args=(audio_queue, responses_queue))
    p2 = Process(target=process_data, args=(audio_queue, responses_queue))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()
