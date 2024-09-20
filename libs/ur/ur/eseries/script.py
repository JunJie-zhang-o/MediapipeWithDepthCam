#!/usr/bin/env python3
# coding=utf-8
"""
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 11:51:56
LastEditTime : 2024-07-04 23:08:08
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 
"""


from dataclasses import InitVar, dataclass, field
from mimetypes import init
from typing import List, Optional


class CMDRepr:

    def __repr__(self) -> str:
        dic = self.__dict__
        cmd = dic["cmd"] if "cmd" in dic else "None"
        args = ""
        for arg in dic:
            if arg != "cmd":
                args += arg + "=" + str(dic[arg]) + ", "
            if arg == "cmd":  # cmd is last param
                args = args[:-2]
                break
        return f"{cmd}({args})"


@dataclass(repr=False)
class Command(CMDRepr):
    cmd: str = field(default=None, init=False)


# ----------------------------------------------------------------


@dataclass(repr=False)
class ConveyorPulseDecode(Command):
    type: int
    A: int
    B: int

    def __post_init__(self):
        self.cmd = "conveyor_pulse_decode"


@dataclass(repr=False)
class EncoderEnablePulseDecode(Command):
    encoder_index: int
    decoder_type: int
    A: int
    B: int

    def __post_init__(self):
        self.cmd = "encoder_enable_pulse_decode"


@dataclass(repr=False)
class EncoderEnableSetTickCount(Command):
    encoder_index: int
    range_id: int

    def __post_init__(self):
        self.cmd = "encoder_enable_set_tick_count"


@dataclass(repr=False)
class EncoderGetTickCount(Command):
    encoder_index: int

    def __post_init__(self):
        self.cmd = "encoder_get_tick_count"


@dataclass(repr=False)
class EncoderSetTickCount(Command):
    encoder_index: int
    count: int

    def __post_init__(self):
        self.cmd = "encoder_set_tick_count"


@dataclass(repr=False)
class EncoderUnwindDeltaTickCount(Command):
    encoder_index: int
    delta_tick_count: float

    def __post_init__(self):
        self.cmd = "encoder_unwind_delta_tick_count"


@dataclass(repr=False)
class EndForceMode(Command):

    def __post_init__(self):
        self.cmd = "end_force_mode"


@dataclass(repr=False)
class EndFreedriveMode(Command):

    def __post_init__(self):
        self.cmd = "end_freedrive_mode"


@dataclass(repr=False)
class EndScrewDriving(Command):

    def __post_init__(self):
        self.cmd = "end_screw_driving"


@dataclass(repr=False)
class EndTeachMode(Command):

    def __post_init__(self):
        self.cmd = "end_teach_mode"


@dataclass(repr=False)
class ForceMode(Command):
    task_frame: List[float]
    selection_vector: List[int]
    wrench: List[float]
    type: int
    limits: List[float]

    def __post_init__(self):
        self.cmd = "force_mode"


@dataclass(repr=False)
class ForceModeExample(Command):

    def __post_init__(self):
        self.cmd = "force_mode_example"


@dataclass(repr=False)
class ForceModeSetDamping(Command):
    damping: float

    def __post_init__(self):
        self.cmd = "force_mode_set_damping"


@dataclass(repr=False)
class ForceModeSetGainScaling(Command):
    scaling: float

    def __post_init__(self):
        self.cmd = "force_mode_set_gain_scaling"


@dataclass(repr=False)
class FreedriveMode(Command):
    freeAxes: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    feature: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        self.cmd = "freedrive_mode"


@dataclass(repr=False)
class GetConveyorTickCount(Command):

    def __post_init__(self):
        self.cmd = "get_conveyor_tick_count"


@dataclass(repr=False)
class GetFreedriveStatus(Command):

    def __post_init__(self):
        self.cmd = "get_freedrive_status"


@dataclass(repr=False)
class GetTargetTcpPoseAlongPath(Command):

    def __post_init__(self):
        self.cmd = "get_target_tcp_pose_along_path"


@dataclass(repr=False)
class GetTargetTcpSpeedAlongPath(Command):

    def __post_init__(self):
        self.cmd = "get_target_tcp_speed_along_path"


@dataclass(repr=False)
class Movec(Command):
    pose_via: List[float]
    pose_to: List[float]
    a: float = 1.2
    v: float = 0.25
    r: float = 0
    mode: int = 0

    def __post_init__(self):
        self.cmd = "movec"


@dataclass(repr=False)
class Movej(Command):
    q: List[float]
    a: float = 1.4
    v: float = 1.05
    t: float = 0
    r: float = 0

    def __post_init__(self):
        self.cmd = "movej"


@dataclass(repr=False)
class Movel(Command):
    pose: List[float]
    a: float = 1.2
    v: float = 0.25
    t: float = 0
    r: float = 0

    def __post_init__(self):
        self.cmd = "movel"


@dataclass(repr=False)
class Movep(Command):
    pose: List[float]
    a: float = 1.2
    v: float = 0.25
    r: float = 0

    def __post_init__(self):
        self.cmd = "movep"


@dataclass(repr=False)
class PathOffsetDisable(Command):
    a: float = 20

    def __post_init__(self):
        self.cmd = "path_offset_disable"


@dataclass(repr=False)
class PathOffsetEnable(Command):

    def __post_init__(self):
        self.cmd = "path_offset_enable"


@dataclass(repr=False)
class PathOffsetGet(Command):
    type: int

    def __post_init__(self):
        self.cmd = "path_offset_get"


@dataclass(repr=False)
class PathOffsetSet(Command):
    offset: List[float]
    type: int

    def __post_init__(self):
        self.cmd = "path_offset_set"


@dataclass(repr=False)
class PathOffsetSetAlphaFilter(Command):
    alpha: float

    def __post_init__(self):
        self.cmd = "path_offset_set_alpha_filter"


@dataclass(repr=False)
class PathOffsetSetMaxOffset(Command):
    transLimit: List[float]
    rotLimit: List[float]

    def __post_init__(self):
        self.cmd = "path_offset_set_max_offset"


@dataclass(repr=False)
class PauseOnErrorCode(Command):
    code: int
    argument: int = None

    def __post_init__(self):
        self.cmd = "pause_on_error_code"


@dataclass(repr=False)
class PositionDeviationWarning(Command):
    enabled: bool
    threshold: float = 0.8

    def __post_init__(self):
        self.cmd = "position_deviation_warning"


@dataclass(repr=False)
class ResetRevolutionCounter(Command):
    qNear: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        self.cmd = "reset_revolution_counter"


@dataclass(repr=False)
class ScrewDriving(Command):
    f: float
    v_limit: float

    def __post_init__(self):
        self.cmd = "screw_driving"


@dataclass(repr=False)
class ServoJ(Command):
    q: List[float]
    a: float
    v: float
    t: float = 0.008
    lookahead_time: float = 0.1
    gain: int = 300

    def __post_init__(self):
        self.cmd = "servoj"


@dataclass(repr=False)
class SetConveyorTickCount(Command):
    tick_count: int
    absolute_encoder_resolution: int = 0

    def __post_init__(self):
        self.cmd = "set_conveyor_tick_count"


@dataclass(repr=False)
class SetPos(Command):
    q: List[float]

    def __post_init__(self):
        self.cmd = "set_pos"


@dataclass(repr=False)
class SetSafetyModeTransitionHardness(Command):
    type: int

    def __post_init__(self):
        self.cmd = "set_safety_mode_transition_hardness"


@dataclass(repr=False)
class SpeedJ(Command):
    qd: List[float]
    a: float
    t: float

    def __post_init__(self):
        self.cmd = "speedj"


@dataclass(repr=False)
class SpeedL(Command):
    xd: List[float]
    a: float
    t: float
    aRot: float = 0

    def __post_init__(self):
        self.cmd = "speedl"


@dataclass(repr=False)
class StopConveyorTracking(Command):
    a: float = 20

    def __post_init__(self):
        self.cmd = "stop_conveyor_tracking"


@dataclass(repr=False)
class StopJ(Command):
    a: float

    def __post_init__(self):
        self.cmd = "stopj"


@dataclass(repr=False)
class StopL(Command):
    a: float
    aRot: float = 0

    def __post_init__(self):
        self.cmd = "stopl"


@dataclass(repr=False)
class TeachMode(Command):

    def __post_init__(self):
        self.cmd = "teach_mode"


@dataclass(repr=False)
class TrackConveyorCircular(Command):
    center: List[float]
    ticks_per_revolution: float
    rotate_tool: bool = False
    encoder_index: int = 0

    def __post_init__(self):
        self.cmd = "track_conveyor_circular"


@dataclass(repr=False)
class TrackConveyorLinear(Command):
    direction: List[float]
    ticks_per_meter: float
    encoder_index: int = 0

    def __post_init__(self):
        self.cmd = "track_conveyor_linear"


# ----------------------------------------------------------
# Module Internals


@dataclass(repr=False)
class Force(Command):

    def __post_init__(self):
        self.cmd = "force"


@dataclass(repr=False)
class GetActualJointPositions(Command):

    def __post_init__(self):
        self.cmd = "get_actual_joint_positions"


@dataclass(repr=False)
class GetActualJointPositionsHistory(Command):
    steps: int = 0

    def __post_init__(self):
        self.cmd = "get_actual_joint_positions_history"


@dataclass(repr=False)
class GetActualJointSpeeds(Command):

    def __post_init__(self):
        self.cmd = "get_actual_joint_speeds"


@dataclass(repr=False)
class GetActualTcpPose(Command):

    def __post_init__(self):
        self.cmd = "get_actual_tcp_pose"


@dataclass(repr=False)
class GetActualTcpSpeed(Command):

    def __post_init__(self):
        self.cmd = "get_actual_tcp_speed"


@dataclass(repr=False)
class GetActualToolFlangePose(Command):

    def __post_init__(self):
        self.cmd = "get_actual_tool_flange_pose"


@dataclass(repr=False)
class GetControllerTemp(Command):

    def __post_init__(self):
        self.cmd = "get_controller_temp"


@dataclass(repr=False)
class GetForwardKin(Command):
    q: Optional[list]
    tcp: Optional[list]

    def __post_init__(self):
        self.cmd = "get_forward_kin"


@dataclass(repr=False)
class GetInverseKin(Command):
    x: list
    qnear: list = field(default_factory=list)
    maxPositionError: float = 1e-10
    maxOrientationError: float = 1e-10
    tcp: Optional[list] = None

    def __post_init__(self):
        self.cmd = "get_inverse_kin"


@dataclass(repr=False)
class GetInverseKinHasSolution(Command):
    pose: list
    qnear: Optional[list] = field(default_factory=list)
    maxPositionError: Optional[float] = 1e-10
    maxOrientationError: Optional[float] = 1e-10
    tcp: Optional[list] = None

    def __post_init__(self):
        self.cmd = "get_inverse_kin_has_solution"


@dataclass(repr=False)
class GetJointTemp(Command):
    j: int

    def __post_init__(self):
        self.cmd = "get_joint_temp"


@dataclass(repr=False)
class GetJointTorques(Command):

    def __post_init__(self):
        self.cmd = "get_joint_torques"


@dataclass(repr=False)
class GetSteptime(Command):

    def __post_init__(self):
        self.cmd = "get_steptime"


@dataclass(repr=False)
class GetTargetJointPositions(Command):

    def __post_init__(self):
        self.cmd = "get_target_joint_positions"


@dataclass(repr=False)
class GetTargetJointSpeeds(Command):

    def __post_init__(self):
        self.cmd = "get_target_joint_speeds"


@dataclass(repr=False)
class GetTargetPayload(Command):

    def __post_init__(self):
        self.cmd = "get_target_payload"


@dataclass(repr=False)
class GetTargetPayloadCog(Command):

    def __post_init__(self):
        self.cmd = "get_target_payload_cog"


@dataclass(repr=False)
class GetTargetPayloadInertia(Command):

    def __post_init__(self):
        self.cmd = "get_target_payload_inertia"


@dataclass(repr=False)
class GetTargetTcpPose(Command):

    def __post_init__(self):
        self.cmd = "get_target_tcp_pose"


@dataclass(repr=False)
class GetTargetTcpSpeed(Command):

    def __post_init__(self):
        self.cmd = "get_target_tcp_speed"


@dataclass(repr=False)
class GetTargetWaypoint(Command):

    def __post_init__(self):
        self.cmd = "get_target_waypoint"


@dataclass(repr=False)
class GetTcpForce(Command):

    def __post_init__(self):
        self.cmd = "get_tcp_force"


@dataclass(repr=False)
class GetTcpOffset(Command):

    def __post_init__(self):
        self.cmd = "get_tcp_offset"


@dataclass(repr=False)
class GetToolAccelerometerReading(Command):

    def __post_init__(self):
        self.cmd = "get_tool_accelerometer_reading"


@dataclass(repr=False)
class GetToolCurrent(Command):

    def __post_init__(self):
        self.cmd = "get_tool_current"


@dataclass(repr=False)
class IsSteady(Command):

    def __post_init__(self):
        self.cmd = "is_steady"


@dataclass(repr=False)
class IsWithinSafetyLimits(Command):
    pose: list
    qnear: Optional[list]

    def __post_init__(self):
        self.cmd = "is_within_safety_limits"


@dataclass(repr=False)
class Popup(Command):
    s: str
    title: str = "Popup"
    warning: bool = False
    error: bool = False
    blocking: bool = False

    def __post_init__(self):
        self.cmd = "popup"


@dataclass(repr=False)
class Powerdown(Command):

    def __post_init__(self):
        self.cmd = "powerdown"


@dataclass(repr=False)
class SetGravity(Command):
    d: list

    def __post_init__(self):
        self.cmd = "set_gravity"


@dataclass(repr=False)
class SetPayload(Command):
    m: float
    cog: list

    def __post_init__(self):
        self.cmd = "set_payload"


@dataclass(repr=False)
class SetPayloadCog(Command):
    CoG: list

    def __post_init__(self):
        self.cmd = "set_payload_cog"


@dataclass(repr=False)
class SetPayloadMass(Command):
    m: float

    def __post_init__(self):
        self.cmd = "set_payload_mass"


@dataclass(repr=False)
class SetTargetPayload(Command):
    m: float
    cog: list
    inertia: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    def __post_init__(self):
        self.cmd = "set_target_payload"


@dataclass(repr=False)
class SetTcp(Command):
    pose: list

    def __post_init__(self):
        self.cmd = "set_tcp"


@dataclass(repr=False)
class Sleep(Command):
    t: float

    def __post_init__(self):
        self.cmd = "sleep"


@dataclass(repr=False)
class StrAt(Command):
    src: str
    index: int

    def __post_init__(self):
        self.cmd = "str_at"


@dataclass(repr=False)
class StrCat(Command):
    op1: str
    op2: str

    def __post_init__(self):
        self.cmd = "str_cat"


@dataclass(repr=False)
class StrEmpty(Command):
    str: str

    def __post_init__(self):
        self.cmd = "str_empty"


@dataclass(repr=False)
class StrLen(Command):
    str: str

    def __post_init__(self):
        self.cmd = "str_len"


@dataclass(repr=False)
class StrSub(Command):
    src: str
    index: int
    length: int

    def __post_init__(self):
        self.cmd = "str_sub"


@dataclass(repr=False)
class Sync(Command):

    def __post_init__(self):
        self.cmd = "sync"


@dataclass(repr=False)
class TextMsg(Command):
    s1: str
    s2: str = ""

    def __post_init__(self):
        self.cmd = "textmsg"


@dataclass(repr=False)
class ToNum(Command):
    str: str

    def __post_init__(self):
        self.cmd = "to_num"


@dataclass(repr=False)
class ToStr(Command):
    val: float

    def __post_init__(self):
        self.cmd = "to_str"


@dataclass(repr=False)
class ToolContact(Command):
    direction: list

    def __post_init__(self):
        self.cmd = "tool_contact"


@dataclass(repr=False)
class ToolContactExamples(Command):

    def __post_init__(self):
        self.cmd = "tool_contact_examples"


# ------------------------------------------------------------
# Module URMath


@dataclass(repr=False)
class Acos(Command):
    f: float

    def __post_init__(self):
        self.cmd = "acos"


@dataclass(repr=False)
class Asin(Command):
    f: float

    def __post_init__(self):
        self.cmd = "asin"


@dataclass(repr=False)
class Atan(Command):
    f: float

    def __post_init__(self):
        self.cmd = "atan"


@dataclass(repr=False)
class Atan2(Command):
    x: float
    y: float

    def __post_init__(self):
        self.cmd = "atan2"


@dataclass(repr=False)
class BinaryListToInteger(Command):
    l: list

    def __post_init__(self):
        self.cmd = "binary_list_to_integer"


@dataclass(repr=False)
class Ceil(Command):
    f: float

    def __post_init__(self):
        self.cmd = "ceil"


@dataclass(repr=False)
class Cos(Command):
    f: float

    def __post_init__(self):
        self.cmd = "cos"


@dataclass(repr=False)
class D2R(Command):
    d: float

    def __post_init__(self):
        self.cmd = "d2r"


@dataclass(repr=False)
class Floor(Command):
    f: float

    def __post_init__(self):
        self.cmd = "floor"


@dataclass(repr=False)
class GetListLength(Command):
    v: list

    def __post_init__(self):
        self.cmd = "get_list_length"


@dataclass(repr=False)
class IntegerToBinaryList(Command):
    x: int

    def __post_init__(self):
        self.cmd = "integer_to_binary_list"


@dataclass(repr=False)
class InterpolatePose(Command):
    p_from: list
    p_to: list
    alpha: float

    def __post_init__(self):
        self.cmd = "interpolate_pose"


@dataclass(repr=False)
class Inv(Command):
    m: list

    def __post_init__(self):
        self.cmd = "inv"


@dataclass(repr=False)
class Length(Command):
    v: list

    def __post_init__(self):
        self.cmd = "length"


@dataclass(repr=False)
class Log(Command):
    b: float
    f: float

    def __post_init__(self):
        self.cmd = "log"


@dataclass(repr=False)
class Norm(Command):
    a: list

    def __post_init__(self):
        self.cmd = "norm"


@dataclass(repr=False)
class Normalize(Command):
    v: list

    def __post_init__(self):
        self.cmd = "normalize"


@dataclass(repr=False)
class PointDist(Command):
    p_from: list
    p_to: list

    def __post_init__(self):
        self.cmd = "point_dist"


@dataclass(repr=False)
class PoseAdd(Command):
    p_1: list
    p_2: list

    def __post_init__(self):
        self.cmd = "pose_add"


@dataclass(repr=False)
class PoseDist(Command):
    p_from: list
    p_to: list

    def __post_init__(self):
        self.cmd = "pose_dist"


@dataclass(repr=False)
class PoseInv(Command):
    p_from: list

    def __post_init__(self):
        self.cmd = "pose_inv"


@dataclass(repr=False)
class PoseSub(Command):
    p_to: list
    p_from: list

    def __post_init__(self):
        self.cmd = "pose_sub"


@dataclass(repr=False)
class PoseTrans(Command):
    p_from: list
    p_from_to: list

    def __post_init__(self):
        self.cmd = "pose_trans"


@dataclass(repr=False)
class Pow(Command):
    base: float
    exponent: float

    def __post_init__(self):
        self.cmd = "pow"


@dataclass(repr=False)
class R2D(Command):
    r: float

    def __post_init__(self):
        self.cmd = "r2d"


@dataclass(repr=False)
class Random(Command):

    def __post_init__(self):
        self.cmd = "random"


@dataclass(repr=False)
class Rotvec2Rpy(Command):
    rotation_vector: list

    def __post_init__(self):
        self.cmd = "rotvec2rpy"


@dataclass(repr=False)
class Rpy2Rotvec(Command):
    rpy_vector: list

    def __post_init__(self):
        self.cmd = "rpy2rotvec"


@dataclass(repr=False)
class Sin(Command):
    f: float

    def __post_init__(self):
        self.cmd = "sin"


@dataclass(repr=False)
class Size(Command):
    v: list

    def __post_init__(self):
        self.cmd = "size"


@dataclass(repr=False)
class Sqrt(Command):
    f: float

    def __post_init__(self):
        self.cmd = "sqrt"


@dataclass(repr=False)
class Tan(Command):
    f: float

    def __post_init__(self):
        self.cmd = "tan"


@dataclass(repr=False)
class Transpose(Command):
    m: list

    def __post_init__(self):
        self.cmd = "transpose"


@dataclass(repr=False)
class WrenchTrans(Command):
    T_from_to: list
    w_from: list

    def __post_init__(self):
        self.cmd = "wrench_trans"


# ------------------------------------------------------
# Module Interfaces
@dataclass(repr=False)
class EnableExternalFtSensor(Command):
    enable: bool
    sensor_mass: float = 0.0
    sensor_measuring_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sensor_cog: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        self.cmd = "enable_external_ft_sensor"


@dataclass(repr=False)
class FtRtdeInputEnable(Command):
    enable: bool
    sensor_mass: float = 0.0
    sensor_measuring_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sensor_cog: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        self.cmd = "ft_rtde_input_enable"


@dataclass(repr=False)
class GetAnalogIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_analog_in"


@dataclass(repr=False)
class GetAnalogOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_analog_out"


@dataclass(repr=False)
class GetConfigurableDigitalIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_configurable_digital_in"


@dataclass(repr=False)
class GetConfigurableDigitalOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_configurable_digital_out"


@dataclass(repr=False)
class GetDigitalIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_digital_in"


@dataclass(repr=False)
class GetDigitalOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_digital_out"


@dataclass(repr=False)
class GetFlag(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_flag"


@dataclass(repr=False)
class GetStandardAnalogIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_standard_analog_in"


@dataclass(repr=False)
class GetStandardAnalogOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_standard_analog_out"


@dataclass(repr=False)
class GetStandardDigitalIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_standard_digital_in"


@dataclass(repr=False)
class GetStandardDigitalOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_standard_digital_out"


@dataclass(repr=False)
class GetToolAnalogIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_tool_analog_in"


@dataclass(repr=False)
class GetToolDigitalIn(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_tool_digital_in"


@dataclass(repr=False)
class GetToolDigitalOut(Command):
    n: int

    def __post_init__(self):
        self.cmd = "get_tool_digital_out"


@dataclass(repr=False)
class ModbusAddSignal(Command):
    IP: str
    slave_number: int
    signal_address: int
    signal_type: int
    signal_name: str
    sequential_mode: Optional[bool] = False

    def __post_init__(self):
        self.cmd = "modbus_add_signal"


@dataclass(repr=False)
class ModbusDeleteSignal(Command):
    signal_name: str

    def __post_init__(self):
        self.cmd = "modbus_delete_signal"


@dataclass(repr=False)
class ModbusGetSignalStatus(Command):
    signal_name: str
    is_secondary_program: bool

    def __post_init__(self):
        self.cmd = "modbus_get_signal_status"


@dataclass(repr=False)
class ModbusSendCustomCommand(Command):
    IP: str
    slave_number: int
    function_code: int
    data: str

    def __post_init__(self):
        self.cmd = "modbus_send_custom_command"


@dataclass(repr=False)
class ModbusSetDigitalInputAction(Command):
    signal_name: str
    action: str

    def __post_init__(self):
        self.cmd = "modbus_set_digital_input_action"


@dataclass(repr=False)
class ModbusSetOutputRegister(Command):
    signal_name: str
    register_value: int
    is_secondary_program: bool

    def __post_init__(self):
        self.cmd = "modbus_set_output_register"


@dataclass(repr=False)
class ModbusSetOutputSignal(Command):
    signal_name: str
    digital_value: bool
    is_secondary_program: bool

    def __post_init__(self):
        self.cmd = "modbus_set_output_signal"


@dataclass(repr=False)
class ModbusSetSignalUpdateFrequency(Command):
    signal_name: str
    update_frequency: float

    def __post_init__(self):
        self.cmd = "modbus_set_signal_update_frequency"


@dataclass(repr=False)
class ReadInputBooleanRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_input_boolean_register"


@dataclass(repr=False)
class ReadInputFloatRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_input_float_register"


@dataclass(repr=False)
class ReadInputIntegerRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_input_integer_register"


@dataclass(repr=False)
class ReadOutputBooleanRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_output_boolean_register"


@dataclass(repr=False)
class ReadOutputFloatRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_output_float_register"


@dataclass(repr=False)
class ReadOutputIntegerRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_output_integer_register"


@dataclass(repr=False)
class ReadPortBit(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_port_bit"


@dataclass(repr=False)
class ReadPortRegister(Command):
    address: int

    def __post_init__(self):
        self.cmd = "read_port_register"


@dataclass(repr=False)
class RpcFactory(Command):
    type: str
    url: str

    def __post_init__(self):
        self.cmd = "rpc_factory"


@dataclass(repr=False)
class RtdeSetWatchdog(Command):
    variable_name: str
    min_frequency: float
    action: Optional[str] = "pause"

    def __post_init__(self):
        self.cmd = "rtde_set_watchdog"


@dataclass(repr=False)
class SetAnalogInputRange(Command):
    port: int
    range: int

    def __post_init__(self):
        self.cmd = "set_analog_inputrange"


@dataclass(repr=False)
class SetAnalogOut(Command):
    n: int
    f: float

    def __post_init__(self):
        self.cmd = "set_analog_out"


@dataclass(repr=False)
class SetConfigurableDigitalOut(Command):
    n: int
    b: bool

    def __post_init__(self):
        self.cmd = "set_configurable_digital_out"


@dataclass(repr=False)
class SetDigitalOut(Command):
    n: int
    b: bool

    def __post_init__(self):
        self.cmd = "set_digital_out"


@dataclass(repr=False)
class SetFlag(Command):
    n: int
    b: bool

    def __post_init__(self):
        self.cmd = "set_flag"


@dataclass(repr=False)
class SetStandardAnalogOut(Command):
    n: int
    f: float

    def __post_init__(self):
        self.cmd = "set_standard_analog_out"


@dataclass(repr=False)
class SetStandardDigitalOut(Command):
    n: int
    b: bool

    def __post_init__(self):
        self.cmd = "set_standard_digital_out"


@dataclass(repr=False)
class SetToolDigitalOut(Command):
    n: int
    b: bool

    def __post_init__(self):
        self.cmd = "set_tool_digital_out"


@dataclass(repr=False)
class SetToolCommunication(Command):
    enabled: bool
    baud_rate: int
    parity: int
    stop_bits: int
    rx_idle_chars: float = 1.0
    tx_idle_chars: float = 3.5

    def __post_init__(self):
        self.cmd = "set_tool_communication"


@dataclass(repr=False)
class SetToolVoltage(Command):
    voltage: int

    def __post_init__(self):
        self.cmd = "set_tool_voltage"


@dataclass(repr=False)
class SocketClose(Command):
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_close"


@dataclass(repr=False)
class SocketGetVar(Command):
    name: str
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_get_var"


@dataclass(repr=False)
class SocketOpen(Command):
    address: str
    port: int
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_open"


@dataclass(repr=False)
class SocketReadAsciiFloat(Command):
    number: int
    socket_name: Optional[str] = "socket_0"
    timeout: Optional[int] = 2

    def __post_init__(self):
        self.cmd = "socket_read_ascii_float"


@dataclass(repr=False)
class SocketReadBinaryInteger(Command):
    number: int
    socket_name: Optional[str] = "socket_0"
    timeout: Optional[int] = 2

    def __post_init__(self):
        self.cmd = "socket_read_binary_integer"


@dataclass(repr=False)
class SocketReadByteList(Command):
    number: int
    socket_name: Optional[str] = "socket_0"
    timeout: Optional[int] = 2

    def __post_init__(self):
        self.cmd = "socket_read_byte_list"


@dataclass(repr=False)
class SocketReadLine(Command):
    socket_name: Optional[str] = "socket_0"
    timeout: Optional[int] = 2

    def __post_init__(self):
        self.cmd = "socket_read_line"


@dataclass(repr=False)
class SocketReadString(Command):
    socket_name: Optional[str] = "socket_0"
    prefix: Optional[str] = ""
    suffix: Optional[str] = ""
    interpret_escape: Optional[bool] = False
    timeout: Optional[int] = 2

    def __post_init__(self):
        self.cmd = "socket_read_string"



@dataclass(repr=False)
class SocketSendByte(Command):
    value: int
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_send_byte"


@dataclass(repr=False)
class SocketSendInt(Command):
    value: int
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_send_int"


@dataclass(repr=False)
class SocketSendLine(Command):
    msg: str
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_send_line"


@dataclass(repr=False)
class SocketSendString(Command):
    msg: str
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_send_string"


@dataclass(repr=False)
class SocketSetVar(Command):
    name: str
    value: int
    socket_name: Optional[str] = "socket_0"

    def __post_init__(self):
        self.cmd = "socket_set_var"


@dataclass(repr=False)
class WriteOutputBooleanRegister(Command):
    address: int
    value: bool

    def __post_init__(self):
        self.cmd = "write_output_boolean_register"


@dataclass(repr=False)
class WriteOutputFloatRegister(Command):
    address: int
    value: float

    def __post_init__(self):
        self.cmd = "write_output_float_register"


@dataclass(repr=False)
class WriteOutputIntegerRegister(Command):
    address: int
    value: int

    def __post_init__(self):
        self.cmd = "write_output_integer_register"


@dataclass(repr=False)
class WritePortBit(Command):
    address: int
    value: bool

    def __post_init__(self):
        self.cmd = "write_port_bit"


@dataclass(repr=False)
class WritePortRegister(Command):
    address: int
    value: int

    def __post_init__(self):
        self.cmd = "write_port_register"


@dataclass(repr=False)
class ZeroFtSensor(Command):

    def __post_init__(self):
        self.cmd = "zero_ftsensor"


# ---------------------------------------------------------------
# Module IOConfiguration
@dataclass(repr=False)
class ModbusSetRunstateDependentChoice(Command):
    signal_name: str
    runstate_choice: int

    def __post_init__(self):
        self.cmd = "modbus_set_runstate_dependent_choice"


@dataclass(repr=False)
class SetAnalogOutputDomain(Command):
    port: int
    domain: int

    def __post_init__(self):
        self.cmd = "set_analog_outputdomain"


@dataclass(repr=False)
class SetConfigurableDigitalInputAction(Command):
    port: int
    action: str

    def __post_init__(self):
        self.cmd = "set_configurable_digital_input_action"


@dataclass(repr=False)
class SetGpBooleanInputAction(Command):
    port: int
    action: str

    def __post_init__(self):
        self.cmd = "set_gp_boolean_input_action"


@dataclass(repr=False)
class SetInputActionsToDefault(Command):

    def __post_init__(self):
        self.cmd = "set_input_actions_to_default"


@dataclass(repr=False)
class SetRunstateConfigurableDigitalOutputToValue(Command):
    outputId: int
    state: bool

    def __post_init__(self):
        self.cmd = "set_runstate_configurable_digital_output_to_value"


@dataclass(repr=False)
class SetRunstateGpBooleanOutputToValue(Command):
    outputId: int
    state: bool

    def __post_init__(self):
        self.cmd = "set_runstate_gp_boolean_output_to_value"


@dataclass(repr=False)
class SetRunstateStandardAnalogOutputToValue(Command):
    outputId: int
    state: float

    def __post_init__(self):
        self.cmd = "set_runstate_standard_analog_output_to_value"


@dataclass(repr=False)
class SetRunstateStandardDigitalOutputToValue(Command):
    outputId: int
    state: bool

    def __post_init__(self):
        self.cmd = "set_runstate_standard_digital_output_to_value"


@dataclass(repr=False)
class SetRunstateToolDigitalOutputToValue(Command):
    outputId: int
    state: bool

    def __post_init__(self):
        self.cmd = "set_runstate_tool_digital_output_to_value"


@dataclass(repr=False)
class SetStandardAnalogInputDomain(Command):
    port: int
    domain: int

    def __post_init__(self):
        self.cmd = "set_standard_analog_input_domain"


@dataclass(repr=False)
class SetStandardDigitalInputAction(Command):
    port: int
    action: str

    def __post_init__(self):
        self.cmd = "set_standard_digital_input_action"


@dataclass(repr=False)
class SetToolAnalogInputDomain(Command):
    port: int
    domain: int

    def __post_init__(self):
        self.cmd = "set_tool_analog_input_domain"


@dataclass(repr=False)
class SetToolDigitalInputAction(Command):
    port: int
    action: str

    def __post_init__(self):
        self.cmd = "set_tool_digital_input_action"


# ---------------------------------------------------------------
# Module Processpath

@dataclass(repr=False)
class McAddCircular(Command):
    pose_via: str
    pose_to: str
    a: float
    v: float
    r: float
    mode: Optional[int] = 0

    def __post_init__(self):
        self.cmd = "mc_add_circular"


@dataclass(repr=False)
class McAddLinear(Command):
    pose: str
    a: float
    v: float
    r: float

    def __post_init__(self):
        self.cmd = "mc_add_linear"


@dataclass(repr=False)
class McAddPath(Command):
    path_id: int
    a: float
    v: float
    r: float

    def __post_init__(self):
        self.cmd = "mc_add_path"


@dataclass(repr=False)
class McGetTargetRtcpSpeed(Command):

    def __post_init__(self):
        self.cmd = "mc_get_target_rtcp_speed"


@dataclass(repr=False)
class McInitialize(Command):
    mode: int
    tcp: str
    doc: Optional[int] = 6

    def __post_init__(self):
        self.cmd = "mc_initialize"


@dataclass(repr=False)
class McLoadPath(Command):
    nc_file: str
    useFeedRate: bool

    def __post_init__(self):
        self.cmd = "mc_load_path"


@dataclass(repr=False)
class McRunMotion(Command):
    id: Optional[int] = -1

    def __post_init__(self):
        self.cmd = "mc_run_motion"


@dataclass(repr=False)
class McSetPcs(Command):
    pcs: str

    def __post_init__(self):
        self.cmd = "mc_set_pcs"


@dataclass(repr=False)
class McSetSpeedFactor(Command):
    s: float

    def __post_init__(self):
        self.cmd = "mc_set_speed_factor"




if __name__ == "__main__":

    a = Command()
    b = Movej()

    print(a)
    print(b)
