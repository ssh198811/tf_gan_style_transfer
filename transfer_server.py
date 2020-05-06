import win32file
import win32pipe
import main
import queue
import _thread
import time

PIPE_WORK_NAME = r'\\.\pipe\work_pipe'
PIPE_CMU_NAME = r'\\.\pipe\cmu_pipe'

PIPE_BUFFER_SIZE = 65535
msg_queue = queue.Queue(1024)
update_flag = False

def createPipe():
    while True:
        named_pipe = win32pipe.CreateNamedPipe(PIPE_WORK_NAME,
                                               win32pipe.PIPE_ACCESS_DUPLEX,
                                               win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT | win32pipe.PIPE_READMODE_MESSAGE,
                                               win32pipe.PIPE_UNLIMITED_INSTANCES,
                                               PIPE_BUFFER_SIZE,
                                               PIPE_BUFFER_SIZE, 500, None)

        print('Create pipe !', PIPE_WORK_NAME)

        try:
            while True:
                try:
                    win32pipe.ConnectNamedPipe(named_pipe, None)
                    data = win32file.ReadFile(named_pipe, PIPE_BUFFER_SIZE, None)

                    if data is None or len(data) < 2:
                        continue

                    # print('receive msg:', data)

                    ret, msg = data
                    if ret == 0:
                        msg_byte = bytes(msg)
                        msg_str = msg_byte.decode()
                        print('receive msg:', msg_str)

                        msg_queue.put(msg_str)

                except BaseException as e:
                    print("exception:", e)
                    break
        finally:
            try:
                win32pipe.DisconnectNamedPipe(named_pipe)
                update_flag = True
            except:
                pass

def send_msg(msg=""):
    file_handle = win32file.CreateFile(PIPE_CMU_NAME,
                                       win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                       win32file.FILE_SHARE_WRITE, None,
                                       win32file.OPEN_EXISTING, 0, None)
    try:
        print('send msg:', msg)
        win32file.WriteFile(file_handle, msg.encode())
        time.sleep(1)

    finally:
        try:
            win32file.CloseHandle(file_handle)
        except:
            pass

def process_img():
    while True:
        while not msg_queue.empty():
            msg = str(msg_queue.get())
            params = msg.split(' ')
            main.process_img(params)
            send_msg(f"finish {msg}")

if __name__ == "__main__":
    main.init_tf_config()

    try:
        _thread.start_new_thread(process_img, ())
        print("启动线程")
    except BaseException as e:
        print("Error: 无法启动线程 ", e)

    createPipe()
