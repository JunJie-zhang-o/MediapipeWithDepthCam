#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 10:46:44
LastEditTime : 2024-08-28 21:43:44
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 
'''

# !Robot Version Must >= 5.0.0, Every Interface Can Used.

import atexit
import socket
import time
from enum import Enum, IntEnum
from typing import Union

import pysftp


class RobotMode(IntEnum)                  : 
    NO_CONTROLLER                         = 0
    DISCONNECTED                          = 1
    CONFIRM_SAFETY                        = 2
    BOOTING                               = 3
    POWER_OFF                             = 4
    POWER_ON                              = 5
    IDLE                                  = 6
    BACKDRIVE                             = 7
    RUNNING                               = 8


class ProgramState(IntEnum)               : 
    STOPPED                               = 0
    PLAYING                               = 1
    PAUSED                                = 2


class OperationalMode(Enum)            : 
    NONE                                  = "none"
    MANUAL                                = "manual"
    AUTOMATIC                             = "automatic"


class SafetyStatus(IntEnum)               : 
    NORMAL                                = 0
    REDUCED                               = 1
    PROTECTIVE_STOP                       = 2
    RECOVERY                              = 3
    SAFEGUARD_STOP                        = 4
    SYSTEM_EMERGENCY_STOP                 = 5
    ROBOT_EMERGENCY_STOP                  = 6
    VIOLATION                             = 7
    FAULT                                 = 8
    AUTOMATIC_MODE_SAFEGUARD_STOP         = 9
    SYSTEM_THREE_POSITION_ENABLING_STOP   = 10

class RobotModel(IntEnum)                 : 
    UR3                                   = 1
    UR5                                   = 2
    UR10                                  = 3
    UR16                                  = 4

class ReportType(Enum)                    : 
    CONTROLLER                            = "controller"
    SOFTWARE                              = "software"
    SYSTEM                                = "system"


class Dashboard:

    PORT = 29999
    DEFAULT_TIMEOUT = 5
    RECV_MAX_SIZE = 2048

    def __init__(self, ip: str, autoConnect: bool=False) -> None:
        
        self.ip = ip
        self._sock = None

        if autoConnect:
            self.connect()

    
    def connect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.DEFAULT_TIMEOUT)
            self._sock.connect((self.ip, self.PORT))
            self._sock.setblocking(True)
            msg = self._sock.recv(self.RECV_MAX_SIZE)
            if msg == b'Connected: Universal Robots Dashboard Server\n':
                atexit.register(self.disconnect)
                self.connection = True
                print(msg.decode().strip())
            else:
                print(f"UR Dashboard can't reply to the message normally, please check whether the device to which the link belongs is normal.")
        except Exception as e:
            print(f"UR Dashboard Connect Failed | IP:{self.ip} Port:{self.PORT} Exception:{e}")
            exit()

    
    def disconnect(self):
        self._sock.close()
        self.connection = False

    
    def sendCommand(self, cmd: str):
        try:
            if not self.connection:
                print("Please First Connect UR Dashboard Port After Execute Command!!!")
                exit()
            
            text = (cmd.encode("ascii") + b"\n")
            self._sock.sendall(text)

            ret = self._sock.recv(self.RECV_MAX_SIZE)
            print(f"Send:{str(text):<15} | Recv:{ret}")
            return ret.decode().strip()
        except Exception as e:
            print(f"UR Dashboard Connect Failed | IP:{self.ip} Exception:{e}")
            exit()


    def load(self, programName: str):
        """
            loading program. the program name need to end in .urp  .
        """
        ret = self.sendCommand(f"load {programName}")
        if ret.startswith("Loading program:"):
            return True
        else:
            print(ret)
            return False
    

    def play(self):
        """
            Starting program.
        """
        ret = self.sendCommand("play")
        return True if ret == "Starting program" else False


    def stop(self):
        """
            Stop program.
        """
        ret = self.sendCommand("stop")
        if ret == "Stopped":
            return True
        else:
            print(ret)
            return False

    
    def pause(self):
        """
            Pausing program.
        """
        ret = self.sendCommand("pause")
        if ret == "Pausing program":
            return True
        else:
            print(ret)
            return False


    def quit(self):
        """
            Close Connection
        """
        ret = self.sendCommand("quit")
        self.disconnect()
        return True


    def shutdown(self) -> bool:
        """
            shurdown and turns of robot and controller
        """
        ret = self.sendCommand("shutdown")
        if ret == "Shutting down":
            return True
        else:
            print(ret)
            return False

    
    def running(self) -> Union[bool, None]:
        """
            program running status.
        """
        ret = self.sendCommand("running")
        if ret.startswith("Program running: "):
            running = ret.split(": ")[1]
            return running == "true"
        else:
            print(ret)
            return None


    def robotmode(self) -> Union[RobotMode, None]:
        """
            enquiry robot mode.
        """
        ret = self.sendCommand("robotmode")
        if ret.startswith("Robotmode:"):
            mode = ret.split(": ")[1]
            return RobotMode[mode]
        else:
            print(ret)
            return None

    
    def getLoadedProgram(self) -> Union[str, None]:
        """
            which program is loaded.
        """
        ret = self.sendCommand("get loaded program")
        if ret.startswith("Loaded program: "):
            return ret.split(": ")[1]
        else:
            print(ret)
            return None

    
    def popup(self, msg: str) -> bool:
        """
            show the popup.
        """
        ret = self.sendCommand(f"popup {msg}")
        return True if ret == "showing popup" else False

    
    def closePopup(self) -> bool:
        """
            close the popup.
        """
        ret = self.sendCommand("close popup")
        return True if ret == "closing popup" else False


    def addToLog(self, log: str) -> bool:
        """
            add log-message to the log history.
        """
        ret = self.sendCommand(f"addToLog {log}")
        return True if ret == "Added log message" else False

    
    def isProgramSaved(self) -> tuple[bool, str]:
        """
            return the save state of the active program and path to loaded program file.
        """
        ret = self.sendCommand("isProgramSaved")
        [saved, programName] = ret.split(' ')
        return saved == "true", programName

    
    def programState(self) -> tuple[ProgramState, str]:
        """
            return the state of the active program and path to loaded program file, or Stopped if no program os loaded.
        """
        ret = self.sendCommand("programState")
        [state, programName] = ret.split(' ')
        return ProgramState[state], programName

    def polyscopeVersion(self) -> str:
        """
            return version information for the UR SW installed on the robot.
        """
        ret = self.sendCommand("PolyscopeVersion")
        return ret

    
    def version(self) -> str:
        """
            returns the version number of the UR SW installed on the robot.
        """ 
        ret = self.sendCommand("version")
        return ret

    
    def setOperationalMode(self, mode:OperationalMode) -> bool:
        """
            control the operational mode
        """
        if mode == OperationalMode.NONE:
            return False
        ret = self.sendCommand(f"set operational mode {mode.value}")
        if ret.startswith("Operational mode"):
            return True
        else:
            print(ret)
            return False

    
    def getOperationalMode(self) -> OperationalMode:
        """
            Returns the operational mode as MANUAL or AUTOMATIC if the password has been set for Mode in Settings
        """
        ret = self.sendCommand(f"get operational mode")
        return OperationalMode[ret]


    def clearOperationalMode(self) -> bool:
        """
            clear Operational Mode
        """
        ret = self.sendCommand(f"clear operational mode")
        if ret.startswith("No longer controlling the operational mode."):
            return True
        else:
            print(ret)
            return False

    
    def powerOn(self) -> bool:
        """
            powers on the robot arm.
        """
        ret = self.sendCommand(f"power on")
        return True if ret == "Powering on" else False


    def powerOff(self) -> bool:
        """
            powers off the robot arm.
        """
        ret = self.sendCommand(f"power off")
        return True if ret == "Powering off" else False


    def brakeRelease(self) -> bool:
        """
            releases the brakes.
        """
        ret = self.sendCommand(f"brake release")
        return True if ret == "Brake releasing" else False


    def safetyStatus(self) -> Union[SafetyStatus, None]:
        """
            safety status inquiry.
        """
        ret = self.sendCommand(f"safetystatus")
        if ret.startswith("Safetystatus:"):
            status = ret.split(": ")[1]  # yes, split str include space
            return SafetyStatus[status]
        else:
            print(ret)
            return None

    
    def unlockProtectiveStop(self) -> bool:
        """
            closes the current popup and unlocks protective stop.
        """
        ret = self.sendCommand(f"unlock protective stop")
        if ret == f"Protective stop releasing":
            return True
        else:
            print(ret)
            return False

    
    def closeSafetyPopup(self) -> bool:
        """
            close a safety popup.
        """
        ret = self.sendCommand(f"close safety popup")
        if ret == 'closing safety popup':
            return True
        else:
            print(ret)
            return False

    
    def loadinstallation(self, installationFileName: str):
        """
            loading installation. the installation file name need to end in .installation  .
        """
        ret = self.sendCommand(f"load installation {installationFileName}")
        return ret


    def restartSafety(self) -> bool:
        """
            used when robot gets a safety fault or violation to restart the safety .
            after safety has been rebooted the robot will be in Power off.
            You should always ensure its okey to restart the system. its highly recommended to check
            the error log before using this command.
        """
        ret = self.sendCommand(f"restart safety")
        if ret == 'Restarting safety':
            return True
        else:
            print(ret)
            return False

    
    def isInRemoteControl(self) -> bool:
        """
            returns the remote control status of the robot.
        """
        ret = self.sendCommand(f"is in remote control")
        return True if ret == "true" else False


    def getSerialNumber(self) -> Union[str, None]:
        """
            returns serial number of the robot.
        """
        ret = self.sendCommand(f"get serial number")
        return ret


    def getRobotModel(self) -> Union[str, None]:
        """
            return the robot model.
        """
        ret = self.sendCommand(f"get robot model")
        return ret


    def generateFlightReport(self, reportType:ReportType=ReportType.SYSTEM) -> Union[str, None]:
        """
            Triggers a flight report of the following type:
                Controller - report with information specific for diagnosing controller errors.For example, in case of protective stops, faults or violations
                Software   - report with information specific for polyscope software failures.
                System     - report with information about robot configuration, programs, installations etc.
            Its required to wait at least 30s between triggering software or controller reports.
        """
        ret = self.sendCommand(f"generate flight report {reportType.value}")
        if ret.startswith("Flight Report generated with id:"):
            flightReportID = ret.split(": ")[1]
            return flightReportID
        else:
            print(ret)
            return None

    
    def generateSupportFile(self, directoryPath: str) -> Union[str, None]:
        """
            Generates a flight report of the type "System" and creates a compressed collection of all the
            existing flight reports on the robot along with the generated flight report. 
            Result file ur_[robot serial number]_YYYY-MM-DD_HH-MM-SS.zip is saved inside <Directory path>
        """
        ret = self.sendCommand(f"generate support file {directoryPath}")
        if ret.startswith("Completed successfully:"):
            supportFileName = ret.split(": ")[1]
            return supportFileName
        else:
            print(ret)
            return None


    def initRobot(self):
        """
            init robot from power off -> running.
        """
        if self.isInRemoteControl():
            while True:
                robotMode = self.robotmode()
                print(robotMode)
                if robotMode == RobotMode.POWER_OFF:
                    if self.powerOn() is False:
                        return False
                if robotMode == RobotMode.POWER_ON:
                    self.brakeRelease()
                if robotMode == RobotMode.RUNNING:
                    return True
                time.sleep(0.5)
        else:
            print("You need to switch robot to remote mode!!!")
            return False


    def uploadProgram(
        self,
        localPath: str,
        remotePath: str,
        username: str = "root",
        password: str = "easybot",
    ) -> None:
        """Trasnfers a URP program from local path to Robot computer

        Args
            local_path (str): Local path to URP program
            remote_path (str): Remote path on UR computer. default program file path is /programs/
            user_name (str): User name for UR computer
            user_password(str): User password for UR compurter
        """
        if not localPath:
            print("Local file was not provided!")
            return
        try:
            # 创建连接
            # cnopts = pysftp.CnOpts()
            # cnopts.hostkeys = None
            with pysftp.Connection(self.ip, username=username, password=password ) as sftp:
                sftp.put(localpath=localPath, remotepath=remotePath)
   

        except Exception as e:
            print(f"UR uploadProgram Faild | Exception:{e}")

        else:
            print("UR program " + localPath + " is Upload to UR onboard " + remotePath)

    
    def downloadProgram(
            self,
            localPath: str,
            remotePath: str,
            username: str = "root",
            password: str = "easybot",
        ) -> None:
            """Trasnfers a URP program from local path to Robot computer

            Args
                local_path (str): Local path to URP program
                remote_path (str): Remote path on UR computer. default program file path is /programs/
                user_name (str): User name for UR computer
                user_password(str): User password for UR compurter
            """
            if not localPath:
                print("Local file was not provided!")
                return
            try:
                # 创建连接
                # cnopts = pysftp.CnOpts()
                # cnopts.hostkeys = None
                with pysftp.Connection(self.ip, username=username, password=password ) as sftp:
                    sftp.get(localpath=localPath, remotepath=remotePath)
    

            except Exception as e:
                print(f"UR downloadProgram Faild | Exception:{e}")

            else:
                print("UR program " + localPath + " is downlaod to UR onboard " + remotePath)



if __name__ == "__main__":

    db = Dashboard("192.168.1.103", autoConnect=True)
    print(db.getSerialNumber())