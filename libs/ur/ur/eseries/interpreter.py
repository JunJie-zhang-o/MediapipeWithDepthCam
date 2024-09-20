import atexit
import logging
import re
import socket

UR_INTERPRETER_SOCKET = 30020


class Interpreter:

    PORT = 30020
    DEFAULT_TIMEOUT = 5

    log = logging.getLogger("interpreter.InterpreterHelper")
    STATE_REPLY_PATTERN = re.compile(r"(\w+):\W+(\d+)?")

    def __init__(self, ip, autoConnect: bool=False):
        self.ip = ip
        if autoConnect:
            self.connect()

    def connect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.DEFAULT_TIMEOUT)
            self._sock.connect((self.ip, self.PORT))
            atexit.register(self.disconnect)
           
        except Exception as e:
            print(f"UR Interpreter Connect Failed | IP:{self.ip} Port:{self.PORT} Exception:{e}")
            exit()

    
    def disconnect(self):
        self._sock.close()
        self.connection = False


    def _get_reply(self):
        """
        read one line from the socket
        :return: text until new line
        """
        collected = b''
        while True:
            part = self.socket.recv(1)
            if part != b"\n":
                collected += part
            elif part == b"\n":
                break
        return collected.decode("utf-8")

    def sendCommand(self, command):
        """
        Send single line command to interpreter mode, and wait for reply
        :param command:
        :return: ack, or status id
        """
        self.log.debug(f"Command: '{command}'")
        if not command.endswith("\n"):
            command += "\n"

        self.socket.send(command.encode("utf-8"))
        raw_reply = self._get_reply()
        self.log.debug(f"Reply: '{raw_reply}'")
        # parse reply, raise exception if command is discarded
        reply = self.STATE_REPLY_PATTERN.match(raw_reply)
        if reply.group(1) == "discard":
            raise Exception("Interpreter discarded message", raw_reply)
        return int(reply.group(2))


    def clearInterpreter(self):
        """
            Clears all interpreted statements, objects, functions, threads, etc. generated in the current
            interpreter mode. Threads started in current interpreter session will be stopped, and deleted.
            Variables defined outside of the current interpreter mode will not be affected by a call to this
            function.
            Only statements interpreted before the clear_interpreter() function will be cleared.
            Statements sent after clear_interpreter() will be queued. When cleaning is done, any
            statements queued are interpreted and responded to. Note that commands such as abort,
            skipbuffer and state commands are executed as soon as they are received.
        """
        return self.sendCommand("clear_interpreter()")


    def skipbuffer(self):
        """
            The interpreter mode furthermore supports the opportunity to skip already sent but not executed
            statements. The interpreter thread will then (after finishing the currently executing statement) skip
            all received but not executed statements.
            After the skip, the interpreter thread will idle until new statements are received. skipbuffer will
            only skip already received statements, new statements can therefore be send right away.
            Return value should be ignored
        """
        return self.sendCommand("skipbuffer")


    def abort(self):
        """
            The interpreter mode offers a mechanism to abort limited number of script functions, even if they are
            called from the main program. Currently only movej and movel can be aborted.
            Aborting a movement will result in a controlled stop if no blend radius is defined.
            If a blend radius is defined then a blend with the next movement will be initiated right away if not
            already in an initial blend, otherwise the command is ignored.
            Return value should be ignored
        """
        return self.sendCommand("abort")


    def stateLastInterpreted(self):
        """
            Replies with the latest interpreted id, i.e. the highest number of interpreted statement so far
        """
        return self.sendCommand("statelastinterpreted")


    def stateLastExecuted(self):
        """
            Replies with the largest id of a statement that has started being executed
        """
        return self.sendCommand("statelastexecuted")


    def stateLastCleared(self):
        """
            Replies with the id for the latest statement to be cleared from the interpreter mode. This clear can
            happen when ending interpreter mode, or by calls to clear_interpreter().
        """
        return self.sendCommand("statelastcleared")


    def stateUnexecuted(self):
        """
            Replies with the num of non executed statements. i.e. the number of statements that would have
            be skipped if skipbuffer was called instead.
        """
        return self.sendCommand("stateunexecuted")


    def end_interpreter(self):
        """
            Ends the interpreter mode, and causes the interpreter_mode() function to return. This
            function can be compiled into the program by sending it to the interpreter socket(30020) as any
            other statement, or can be called from anywhere else in the program.
            By default everything interpreted will be cleared when ending, though the state of the robot, the
            modifications to local variables from the enclosing scope, and the global variables will remain
            affected by any changes made. The interpreter thread will be idle after this call.
        """
        return self.sendCommand("end_interpreter()")
