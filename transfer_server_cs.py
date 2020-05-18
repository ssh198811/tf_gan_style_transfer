import main
import queue
import _thread
import time
import socket
import traceback


msg_queue = queue.Queue(1024)


if __name__ == "__main__":
    main.init_tf_config()

    host = ""
    port = 12345
    address = (host, port)
    time_now = time.strftime("%Y-%m-%d %H:%S:%M", time.localtime())

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(address)
    s.listen(1)

    while True:
        print("Waiting for connections...")
        try:
            client_connection, client_address = s.accept()
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            continue

        try:

            while True:
                buf = client_connection.recv(1024)
                print("client send:", buf)
                msg_queue.put(buf)

        except (KeyboardInterrupt, SystemError):
            raise
        except:
            traceback.print_exc()

    try:
        client_connection.close()
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()


    def process_img():
        while True:
            while not msg_queue.empty():
                msg = str(msg_queue.get())
                params = msg.split(' ')
                main.process_img(params)
                s.send("finish")

    try:
        _thread.start_new_thread(process_img, ())
        print("启动线程")
    except BaseException as e:
        print("Error: 无法启动线程 ", e)

