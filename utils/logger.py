#!/usr/bin/env python3
# coding=utf-8
import logging
import logging.handlers

class RemoteTCPServerLogHandler(logging.StreamHandler):
    """
    """
    def __init__(self, host, port=logging.handlers.DEFAULT_TCP_LOGGING_PORT):
        logging.StreamHandler.__init__(self)
        import socket
        self.client = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
        self.tcpServerAddr = (host, port) 
        self.client.connect(self.tcpServerAddr)  
        self.stream = self.client.makefile("wr")
        selfIP = self.client.getsockname()[0]
        self.formatter = logging.Formatter("%(asctime)s | " + f"Client IP:{selfIP} | " +"%(levelname)-8s |%(filename)s:%(lineno)-4d | %(message)s")
