#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 15:37:12
LastEditTime : 2024-07-14 14:39:47
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : Define Primary And Secondary Port Data Struct
'''



from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, List


class ControlMode(IntEnum)                              : 
    CONTROL_MODE_POSITION                               = 0
    CONTROL_MODE_TEACH                                  = 1
    CONTROL_MODE_FORCE                                  = 2
    CONTROL_MODE_TORQUE                                 = 3


class RobotModes(IntEnum)                               : 
    ROBOT_MODE_NO_CONTROLLER                            = -1
    ROBOT_MODE_DISCONNECTED                             = 0
    ROBOT_MODE_CONFIRM_SAFETY                           = 1
    ROBOT_MODE_BOOTING                                  = 2
    ROBOT_MODE_POWER_OFF                                = 3
    ROBOT_MODE_POWER_ON                                 = 4
    ROBOT_MODE_IDLE                                     = 5
    ROBOT_MODE_BACKDRIVE                                = 6
    ROBOT_MODE_RUNNING                                  = 7
    ROBOT_MODE_UPDATING_FIRMWARE                        = 8

class JointModes(IntEnum)                               : 
    JOINT_MODE_RESET                                    = 235
    JOINT_MODE_SHUTTING_DOWN                            = 236
    JOINT_MODE_BACKDRIVE                                = 238
    JOINT_MODE_POWER_OFF                                = 239
    JOINT_MODE_READY_FOR_POWER_OFF                      = 240
    JOINT_MODE_NOT_RESPONDING                           = 245
    JOINT_MODE_MOTOR_INITIALISATION                     = 246
    JOINT_MODE_BOOTING                                  = 247
    JOINT_MODE_VIOLATION                                = 251
    JOINT_MODE_FAULT                                    = 252
    JOINT_MODE_RUNNING                                  = 253
    JOINT_MODE_IDLE                                     = 255


class ToolModes(IntEnum)                                : 
    JOINT_MODE_RESET                                    = 235
    JOINT_MODE_SHUTTING_DOWN                            = 236
    JOINT_MODE_POWER_OFF                                = 239
    JOINT_MODE_NOT_RESPONDING                           = 245
    JOINT_MODE_BOOTING                                  = 247
    JOINT_MODE_BOOTLOADER                               = 249
    JOINT_MODE_FAULT                                    = 252
    JOINT_MODE_RUNNING                                  = 253
    JOINT_MODE_IDLE                                     = 255




class MessagesSources(IntEnum)                          : 
    MESSAGE_SOURCE_JOINT_0_FPGA                         = 100
    MESSAGE_SOURCE_JOINT_0_A                            = 110
    MESSAGE_SOURCE_JOINT_0_B                            = 120
    MESSAGE_SOURCE_JOINT_1_FPGA                         = 101
    MESSAGE_SOURCE_JOINT_1_A                            = 111
    MESSAGE_SOURCE_JOINT_1_B                            = 121
    MESSAGE_SOURCE_JOINT_2_FPGA                         = 102
    MESSAGE_SOURCE_JOINT_2_A                            = 112
    MESSAGE_SOURCE_JOINT_2_B                            = 122
    MESSAGE_SOURCE_JOINT_3_FPGA                         = 103
    MESSAGE_SOURCE_JOINT_3_A                            = 113
    MESSAGE_SOURCE_JOINT_3_B                            = 123
    MESSAGE_SOURCE_JOINT_4_FPGA                         = 104
    MESSAGE_SOURCE_JOINT_4_A                            = 114
    MESSAGE_SOURCE_JOINT_4_B                            = 124
    MESSAGE_SOURCE_JOINT_5_FPGA                         = 105
    MESSAGE_SOURCE_JOINT_5_A                            = 115
    MESSAGE_SOURCE_JOINT_5_B                            = 125
    MESSAGE_SOURCE_TOOL_FPGA                            = 106
    MESSAGE_SOURCE_TOOL_A                               = 116
    MESSAGE_SOURCE_TOOL_B                               = 126
    MESSAGE_SOURCE_EUROMAP_FPGA                         = 107
    MESSAGE_SOURCE_EUROMAP_A                            = 117
    MESSAGE_SOURCE_EUROMAP_B                            = 127
    MESSAGE_SOURCE_TEACH_PENDANT_A                      = 108
    MESSAGE_SOURCE_TEACH_PENDANT_B                      = 118
    MESSAGE_SOURCE_SCB_FPGA                             = 40
    MESSAGE_SAFETY_PROCESSOR_UA                         = 20
    MESSAGE_SAFETY_PROCESSOR_UB                         = 30
    MESSAGE_SOURCE_ROBOTINTERFACE                       = -2
    MESSAGE_SOURCE_RTMACHINE                            = -3
    MESSAGE_SOURCE_SIMULATED_ROBOT                      = -4
    MESSAGE_SOURCE_GUI                                  = -5
    MESSAGE_SOURCE_CONTROLLER                           = 7
    MESSAGE_SOURCE_RTDE                                 = 8


class MessagesTypes(IntEnum)                            : 
    MESSAGE_TYPE_DISCONNECT                             = -1
    MESSAGE_TYPE_ROBOT_STATE                            = 16
    MESSAGE_TYPE_ROBOT_MESSAGE                          = 20
    MESSAGE_TYPE_HMC_MESSAGE                            = 22
    MESSAGE_TYPE_MODBUS_INFO_MESSAGE                    = 5
    MESSAGE_TYPE_SAFETY_SETUP_BROADCAST_MESSAGE         = 23
    MESSAGE_TYPE_SAFETY_COMPLIANCE_TOLERANCES_MESSAGE   = 24
    MESSAGE_TYPE_PROGRAM_STATE_MESSAGE                  = 25


class SafteyModeTypes(IntEnum)                          : 
    SAFETY_MODE_UNDEFINED_SAFETY_MODE                   = 11
    SAFETY_MODE_VALIDATE_JOINT_ID                       = 10
    SAFETY_MODE_FAULT                                   = 9
    SAFETY_MODE_VIOLATION                               = 8
    SAFETY_MODE_ROBOT_EMERGENCY_STOP                    = 7
    SAFETY_MODE_SYSTEM_EMERGENCY_STOP                   = 6
    SAFETY_MODE_SAFEGUARD_STOP                          = 5
    SAFETY_MODE_RECOVERY                                = 4
    SAFETY_MODE_PROTECTIVE_STOP                         = 3
    SAFETY_MODE_REDUCED                                 = 2
    SAFETY_MODE_NORMAL                                  = 1


class SafetyStatusTypes(IntEnum)                        : 
    SAFETY_STATUS_SYSTEM_THREE_POSITION_ENABLING_STOP   = 13
    SAFETY_STATUS_AUTOMATIC_MODE_SAFEGUARD_STOP         = 12
    SAFETY_STATUS_UNDEFINED_SAFETY_MODE                 = 11
    SAFETY_STATUS_VALIDATE_JOINT_ID                     = 10
    SAFETY_STATUS_FAULT                                 = 9
    SAFETY_STATUS_VIOLATION                             = 8
    SAFETY_STATUS_ROBOT_EMERGENCY_STOP                  = 7
    SAFETY_STATUS_SYSTEM_EMERGENCY_STOP                 = 6
    SAFETY_STATUS_SAFEGUARD_STOP                        = 5
    SAFETY_STATUS_RECOVERY                              = 4
    SAFETY_STATUS_PROTECTIVE_STOP                       = 3
    SAFETY_STATUS_REDUCED                               = 2
    SAFETY_STATUS_NORMAL                                = 1






class ReportLevels(IntEnum)                             : 
    REPORT_LEVEL_DEBUG                                  = 0
    REPORT_LEVEL_INFO                                   = 1
    REPORT_LEVEL_WARNING                                = 2
    REPORT_LEVEL_VIOLATION                              = 3
    REPORT_LEVEL_FAULT                                  = 4
    REPORT_LEVEL_DEVL_DEBUG                             = 128
    REPORT_LEVEL_DEVL_INFO                              = 129
    REPORT_LEVEL_DEVL_WARNING                           = 130
    REPORT_LEVEL_DEVL_VIOLATION                         = 131
    REPORT_LEVEL_DEVL_FAULT                             = 132





class RequestedTypes(IntEnum)                           : 
    REQUEST_VALUE_TYPE_BOOLEAN                          = 0
    REQUEST_VALUE_TYPE_INTEGER                          = 1
    REQUEST_VALUE_TYPE_FLOAT                            = 2
    REQUEST_VALUE_TYPE_STRING                           = 3
    REQUEST_VALUE_TYPE_POSE                             = 4
    REQUEST_VALUE_TYPE_JOINTVECTOR                      = 5
    REQUEST_VALUE_TYPE_WAYPOINT                         = 6 # (UNUSED)
    REQUEST_VALUE_TYPE_EXPRESSION                       = 7 #(UNUSED)
    REQUEST_VALUE_TYPE_NONE                             = 8 #(*) If the 'requestedType' is set to the value 8, then it is a 'PopupMessage' type.

# --------------------------------------------------------


class Joints                        : 
    pass

@dataclass
class JointsFloat(Joints)           : 
    j1                              : float
    j2                              : float
    j3                              : float
    j4                              : float
    j5                              : float
    j6                              : float


@dataclass
class JointsInt(Joints)             : 
    j1                              : int
    j2                              : int
    j3                              : int
    j4                              : int
    j5                              : int
    j6                              : int


@dataclass
class RobotModeData()               : 
    fmt                             : ClassVar[str] = "iBQ???????BBdddB"
    code                            : ClassVar[int] = 0
    bindProperty                    : ClassVar[str] = "RobotModeData"
    packageSize                     : int
    packageType                     : int
    timestamp                       : int
    isRealRobotConnected            : bool
    isRealRobotEnabled              : bool
    isRobotPowerOn                  : bool
    isEmergencyStopped              : bool
    isProtectiveStopped             : bool
    isProgramRunning                : bool
    isProgramPaused                 : bool
    robotMode                       : int
    controlMode                     : int
    targetSpeedFraction             : float
    speedScaling                    : float
    targetSpeedFractionLimit        : float
    reserved                        : int




@dataclass
class JointData()                   : 
    fmt                             : ClassVar[str] = "iBddddddddddddddddddffffffffffffffffffffffffBBBBBB"
    code                            : ClassVar[int] = 1
    bindProperty                    : ClassVar[str] = "JointData"
    packageSize                     : int
    package_type                    : int
    q_actual                        : JointsFloat
    q_target                        : JointsFloat
    qd_actual                       : JointsFloat
    I_actual                        : JointsFloat
    V_actual                        : JointsFloat
    T_motor                         : JointsFloat
    T_micro                         : JointsFloat  # Deprecated - ignore
    jointMode                       : JointsInt




@dataclass
class ToolData()                    : 
    fmt                             : ClassVar[str] = "iBBBddfBffB"
    code                            : ClassVar[int] = 2
    bindProperty                    : ClassVar[str] = "ToolData"
    packageSize                     : int
    packageType                     : int
    analogInputRange0               : int
    analogInputRange1               : int
    analogInput0                    : float
    analogInput1                    : float
    toolVoltage48V                  : float
    toolOutputVoltage               : int
    toolCurrent                     : float
    toolTemperature                 : float
    toolMode                        : int




@dataclass
class MasterboardData1()             : 
    fmt                             : ClassVar[str] = "iBiiBBddccddffffBBcIBBc"
    code                            : ClassVar[int] = 3
    bindProperty                    : ClassVar[str] = "MasterboardData1"
    packageSize                     : int
    packageType                     : int
    digitalInputBits                : int
    digitalOutputBits               : int
    analogInputRange0               : int
    analogInputRange1               : int
    analogInput0                    : float
    analogInput1                    : float
    analogOutputDomain0             : int
    analogOutputDomain1             : int
    analogOutput0                   : float
    analogOutput1                   : float
    masterBoardTemperature          : float
    robotVoltage48V                 : float
    robotCurrent                    : float
    masterIOCurrent                 : float
    safetyMode                      : int
    inReducedMode                   : int
    euromap67InterfaceInstalled     : int
    usedByURSW1                     : float
    operationalModeSelectorInput    : float
    threePositionEnablingDeviceInput: float
    usedByURSW1                     : float


@dataclass
class MasterboardData2()             : 
    fmt                             : ClassVar[str] = "iBiiBBddccddffffBBcIIffIBBc"
    code                            : ClassVar[int] = 3
    bindProperty                    : ClassVar[str] = "MasterboardData2"
    packageSize                     : int
    packageType                     : int
    digitalInputBits                : int
    digitalOutputBits               : int
    analogInputRange0               : int
    analogInputRange1               : int
    analogInput0                    : float
    analogInput1                    : float
    analogOutputDomain0             : int
    analogOutputDomain1             : int
    analogOutput0                   : float
    analogOutput1                   : float
    masterBoardTemperature          : float
    robotVoltage48V                 : float
    robotCurrent                    : float
    masterIOCurrent                 : float
    safetyMode                      : int
    inReducedMode                   : int
    euromap67InterfaceInstalled     : int
    euromapInputBits                : float
    euromapOutputBits               : float
    euromapVoltage24V               : float
    euromapCurrent                  : float
    usedByURSW1                     : float
    operationalModeSelectorInput    : float
    threePositionEnablingDeviceInput: float
    usedByURSW1                     : float



@dataclass
class CartesianInfo()               : 
    fmt                             : ClassVar[str] = "iBdddddddddddd"
    code                            : ClassVar[int] = 4
    bindProperty                    : ClassVar[str] = "CartesianInfo"
    packageSize                     : int
    packageType                     : int
    X                               : float
    Y                               : float
    Z                               : float
    Rx                              : float
    Ry                              : float
    Rz                              : float
    TCPOffsetX                      : float
    TCPOffsetY                      : float
    TCPOffsetZ                      : float
    TCPOffsetRx                     : float
    TCPOffsetRy                     : float
    TCPOffsetRz                     : float




@dataclass
class KinematicsInfo()              : 
    fmt                             : ClassVar[str] = "iBIIIIIIddddddddddddddddddddddddI"
    code                            : ClassVar[int] = 5
    bindProperty                    : ClassVar[str] = "KinematicsInfo"
    packageSize                     : int
    packageType                     : int
    checksum                        : JointsInt
    DHtheta                         : JointsFloat
    DHa                             : JointsFloat
    Dhd                             : JointsFloat
    Dhalpha                         : JointsFloat
    calibrationStatus               : int




@dataclass
class ConfigurationData()           : 
    fmt                             : ClassVar[str] = "iBdddddddddddddddddddddddddddddddddddddddddddddddddddddIIII"
    code                            : ClassVar[int] = 6
    bindProperty                    : ClassVar[str] = "ConfigurationData"
    packageSize                     : int
    packageType                     : int
    jointMinLimit                   : JointsFloat
    jointMaxLimit                   : JointsFloat
    jointMaxSpeed                   : JointsFloat
    jointMaxAcceleration            : JointsFloat
    vJointDefault                   : float
    aJointDefault                   : float
    vToolDefault                    : float
    aToolDefault                    : float
    eqRadius                        : float
    DHa                             : JointsFloat
    Dhd                             : JointsFloat
    DHalpha                         : JointsFloat
    DHtheta                         : JointsFloat
    masterboardVersion              : int
    controllerBoxType               : int
    robotType                       : int
    robotSubType                    : int




@dataclass
class ForceModeData                 : 
    fmt                             : ClassVar[str] = "iBddddddd"
    code                            : ClassVar[int] = 7
    bindProperty                    : ClassVar[str] = "ForceModeData"
    packageSize                     : int
    packageType                     : int
    Fx                              : float
    Fy                              : float
    Fz                              : float
    Frx                             : float
    Fry                             : float
    Frz                             : float
    robotDexterity                  : float


@dataclass
class AdditionInfo                  : 
    fmt                             : ClassVar[str] = "iBB??B"
    code                            : ClassVar[int] = 8
    bindProperty                    : ClassVar[str] = "AdditionInfo"
    packageSize                     : int
    packageType                     : int
    tpButtonState                   : int
    freedriveButtonEnabled          : bool
    IOEnabledFreedrive              : bool
    reservedL                       : int


@dataclass
class CalibrationData               : 
    fmt                             : ClassVar[str] = "iBdddddd"
    code                            : ClassVar[int] = 9
    bindProperty                    : ClassVar[str] = "CalibrationData"
    packageSize                     : int
    packageType                     : int
    Fx                              : float
    Fy                              : float
    Fz                              : float
    Frx                             : float
    Fry                             : float
    Frz                             : float


@dataclass
class SafetyData                    : 
    # !It is used internally bu URRobots sw only and should be skipped.
    fmt                             : ClassVar[str] = "iB"
    code                            : ClassVar[int] = 10
    bindProperty                    : ClassVar[str] = "SafetyData"
    packageSize                     : int
    packageType                     : int



@dataclass
class ToolCommunicationInfo         : 
    fmt                             : ClassVar[str] = "iB?IIIff"
    code                            : ClassVar[int] = 11
    bindProperty                    : ClassVar[str] = "ToolCommunicationInfo"
    packageSize                     : int
    packageType                     : int
    toolCommunicationIsEnabled      : bool
    baudRate                        : int
    parity                          : int
    stopBits                        : int
    RxIdleChars                     : float
    TxIdleChars                     : float


@dataclass
class ToolModeInfo                  : 
    fmt                             : ClassVar[str] = "iBBBB"
    code                            : ClassVar[int] = 12
    bindProperty                    : ClassVar[str] = "ToolModeInfo"
    packageSize                     : int
    packageType                     : int
    outputMode                      : int
    digitalOutputModeOutput0        : int
    digitalOutputModeOutput1        : int


@dataclass
class SingularityInfo               : 
    # !It is used internally bu URRobots sw only and should be skipped.
    fmt                             : ClassVar[str] = "iBBB"
    code                            : ClassVar[int] = 13
    bindProperty                    : ClassVar[str] = "SingularityInfo"
    packageSize                     : int
    packageType                     : int
    singularitySeverity             : int
    singularityType                 : int


@dataclass
class VersionMessage                : 
    # *This is the first package sent on both the primary and secondary client interfaces. This package it is not part of the robot state message
    charNum1                        : ClassVar[int] = 0
    charNum2                        : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbBB{charNum1}sBBii{charNum2}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 3
    bindProperty                    : ClassVar[str] = "VersionMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : str
    source                          : int
    robotMessageType                : int 
    projectNameSize                 : int
    projectName                     : int
    majorVersion                    : int
    minorVersion                    : int
    bugfixVersion                   : int
    buildNumber                     : int
    buildDate                       : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbBB{cls.charNum1}sBBii{cls.charNum2}s"



# Messages sent to Primary clients only
@dataclass
class SafetyModeMessage             : 
    fmt                             : ClassVar[str] = "iBQbBiiBII"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 5
    bindProperty                    : ClassVar[str] = "SafetyModeMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    robotMessageCode                : int
    robotMessageArgument            : int
    safetyModeType                  : int
    reportDataType                  : int
    reportData                      : int





@dataclass
class RobotCommMessage              : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbBiiiII{charNum}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 6
    bindProperty                    : ClassVar[str] = "RobotCommMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int 
    robotMessageCode                : int
    robotMessageArgument            : int
    robotMessageReportLevel         : int
    robotMessageDataType            : int
    robotMessageData                : int
    robotCommTextMessage            : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbBiiiII{cls.charNum}s"


@dataclass
class KeyMessage                    : 
    charNum1                        : ClassVar[int] = 0
    charNum2                        : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbBiiB{charNum1}s{charNum2}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 7
    bindProperty                    : ClassVar[str] = "KeyMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    robotMessageCode                : int
    robotMessageArgument            : int
    robotMessageTitleSize           : int
    robotMessageTitle               : str
    keyTextMessage                  : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbBiiB{cls.charNum1}s{cls.charNum2}s"



@dataclass
class ProgramThread:
    labelId                         : int
    labelNameLength                 : int
    labelName                       : str
    threadNameLength                : int
    threadName                      : str


@dataclass
class ProgramThreadsMessage         : 
    fmt                             : ClassVar[str] = "iBQbBiiBss"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 14
    bindProperty                    : ClassVar[str] = "ProgramThreadsMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    programThreads                  : List[ProgramThread]



@dataclass
class PopupMessage                  : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbBII???B{charNum}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 2
    bindProperty                    : ClassVar[str] = "PopupMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    requestId                       : int
    requestedType                   : int
    warning                         : bool
    error                           : bool
    blocking                        : bool
    popupMessageTitleSize           : int
    popupMessageTitle               : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbBII???B{cls.charNum}s"


@dataclass
class RequestValueMessage           : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbBII{charNum}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 9
    bindProperty                    : ClassVar[str] = "RequestValueMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int 
    requestId                       : int
    requestedType                   : int
    requestTextMessage              : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbBII{cls.charNum}s"


@dataclass
class TextMessage                   : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQbB{charNum}s"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 0
    bindProperty                    : ClassVar[str] = "TextMessage"
    messageSize                     : int
    messageType                     : int 
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    textTextMessage                 : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQbB{cls.charNum}s"



@dataclass
class RuntimeExceptionMessage       : 
    fmt                             : ClassVar[str] = "iBQbBii"
    code                            : ClassVar[int] = 20
    robotMsgType                    : ClassVar[int] = 10
    bindProperty                    : ClassVar[str] = "RuntimeExceptionMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    source                          : int
    robotMessageType                : int
    scriptLineNumber                : int
    scriptColumnNumber              : int


@dataclass
class GlobalVariablesSetupMessage   : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = f"iBQBH{charNum}s"
    code                            : ClassVar[int] = 25
    robotMsgType                    : ClassVar[int] = 0
    bindProperty                    : ClassVar[str] = "GlobalVariablesSetupMessage"
    messageSize                     : int
    messageType                     : int  
    timestamp                       : int
    robotMessageType                : int
    startIndex                      : int
    variableNames                   : str

    @classmethod
    def _refreshFmt(cls):
        cls.fmt =  f"iBQBH{cls.charNum}s"


@dataclass
class GlobalVariablesUpdateMessage  : 
    charNum                         : ClassVar[int] = 0
    fmt                             : ClassVar[str] = "iBQBHc"
    code                            : ClassVar[int] = 25
    robotMsgType                    : ClassVar[int] = 1
    bindProperty                    : ClassVar[str] = "GlobalVariablesUpdateMessage"
    messageSize                     : int
    messageType                     : int
    timestamp                       : int
    robotMessageType                : int 
    startIndex                      : int
    variableValues                  : int




# ----------------------------------------------------




if __name__ == "__main__":
    # a = VersionMessage(30, 40)
    # print(a)
    print(JointData())