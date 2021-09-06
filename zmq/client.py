import zmq

from wtisp.dataset.zmq import ZMQConfig


class Client(object):
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        tcp_address = 'tcp://localhost:{}'.format(ZMQConfig['port'])
        self.socket.connect('tcp://localhost:5566')

    def __call__(self, ):
        self.socket.send(b"test")
        print(self.socket.recv())


tmp = Client()
while True:
    tmp()
