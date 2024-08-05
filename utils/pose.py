"""
Author: Elite_zhangjunjie
CreateDate: 
LastEditors: Elite_zhangjunjie
LastEditTime: 2022-11-09 17:23:32
Description: 基于list实现的Pose类,主要实现打印和调试中角度和弧度的显示,以及求逆和乘
"""

import math
from typing import Any, Union, overload
from typing_extensions import Self

try:
    from typing import SupportsIndex
except ImportError:
    from typing_extensions import SupportsIndex
from transformations import euler_matrix, euler_from_matrix
import numpy as np


class Pose(list):

    MAX_BITS = 6
    AXES = "sxyz"
    __solts__ = ("x", "y", "z", "rx", "ry", "rz")

    @overload
    def __init__(self, _list: list, *, is_rad: bool = True): ...

    @overload
    def __init__(
        self,
        _x: float,
        _y: float,
        _z: float,
        _rx: float,
        _ry: float,
        _rz: float,
        *,
        is_rad: bool = True,
    ): ...

    def __init__(self, *args, **kwargs):
        is_rad = kwargs["is_rad"] if "is_rad" in kwargs else True
        if type(args[0]) == list:
            self.__refresh(args[0], is_rad)
            super().__init__([self.x, self.y, self.z, self.rx, self.ry, self.rz])
        elif len(args) == 6:
            self.__refresh(args[:6], is_rad)
            super().__init__([self.x, self.y, self.z, self.rx, self.ry, self.rz])
        else:
            super().__init__(args[0])

    def __refresh(self, _pose: list, is_rad: bool = True):
        self.x, self.y, self.z = _pose[0], _pose[1], _pose[2]
        if is_rad:
            self.rx, self.ry, self.rz = _pose[3], _pose[4], _pose[5]
        else:
            self.rx, self.ry, self.rz = (
                math.radians(_pose[3]),
                math.radians(_pose[4]),
                math.radians(_pose[5]),
            )

    def _to_deg(self) -> list:
        return self.__round(
            [
                self.x,
                self.y,
                self.z,
                math.degrees(self.rx),
                math.degrees(self.ry),
                math.degrees(self.rz),
            ]
        )

    def __round(self, pose: list = None):
        if pose is not None:
            return [round(i, self.MAX_BITS) for i in pose]
        else:
            return [
                round(i, self.MAX_BITS)
                for i in [self.x, self.y, self.z, self.rx, self.ry, self.rz]
            ]

    @property
    def matrix(self):
        """获取欧拉角的齐次变换矩阵"""
        matrix = euler_matrix(self.rx, self.ry, self.rz, axes=self.AXES)
        matrix[0][3] = self.x
        matrix[1][3] = self.y
        matrix[2][3] = self.z
        return matrix

    def __str__(self) -> str:
        return f"Pose:{self._to_deg()} in degress"

    def __repr__(self) -> str:
        return f"Pose:{self._to_deg()} in degress"
        # return f"Pose:{self.__round()} in radian"

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        if "rz" in self.__dict__:
            super().__init__([self.x, self.y, self.z, self.rx, self.ry, self.rz])

    def __setitem__(self, __key: str, __value: Any) -> None:
        super().__setitem__(__key, __value)
        self.__refresh(self)

    def __mul__(self, __n: SupportsIndex) -> Self:
        return pose_mul(self, __n)

    def __rmul__(self, __n: SupportsIndex) -> list:
        return pose_mul(__n, self)

    def __imul__(self, __n: SupportsIndex) -> Self:
        return self.__mul__(__n)

    @property
    def inverse(self) -> list:
        return pose_inv(self)


def pose_mul(pose1: Union[Pose, list], pose2: Union[Pose, list]) -> Pose:
    """位姿的乘"""
    matrix1 = euler_to_matirx(pose1)
    matrix2 = euler_to_matirx(pose2)
    mul_matrix = matrix_mul(matrix1, matrix2)
    return matrix_to_euler(mul_matrix)


def pose_inv(pose: Union[Pose, list]) -> Pose:
    """位姿的逆"""
    matrix = euler_to_matirx(pose)
    inv_martrix = matrix_inverse(matrix)
    return matrix_to_euler(inv_martrix)


def matrix_inverse(matrix) -> list:
    """矩阵的逆"""
    return np.linalg.inv(matrix)


def matrix_mul(matrix1: list, matrix2: list) -> list:
    """矩阵的乘"""
    return np.matmul(matrix1, matrix2)


def matrix_to_euler(matrix: list) -> Pose:
    """齐次变换矩阵转欧拉角(弧度)"""
    x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]
    matrix[0][3] = 0
    matrix[1][3] = 0
    matrix[2][3] = 0
    rx, ry, rz = euler_from_matrix(matrix)
    return Pose(x, y, z, rx, ry, rz)


def euler_to_matirx(euler: Union[Pose, list]) -> list:
    """欧拉角转齐次变换矩阵"""
    [x, y, z, rx, ry, rz] = euler
    matrix = euler_matrix(rx, ry, rz, axes=Pose.AXES)
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    return matrix


def rpy2rv(x, y, z, roll, pitch, yaw):

    alpha = math.radians(yaw)
    beta = math.radians(pitch)
    gamma = math.radians(roll)

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg

    theta = math.acos((r11 + r22 + r33 - 1) / 2)
    sth = math.sin(theta)
    kx = (r32 - r23) / (2 * sth)
    ky = (r13 - r31) / (2 * sth)
    kz = (r21 - r12) / (2 * sth)

    return [x, y, z, (theta * kx), (theta * ky), (theta * kz)]


def rv2rpy(x, y, z, rx, ry, rz):

    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta
    cth = math.cos(theta)
    sth = math.sin(theta)
    vth = 1 - math.cos(theta)

    r11 = kx * kx * vth + cth
    r12 = kx * ky * vth - kz * sth
    r13 = kx * kz * vth + ky * sth
    r21 = kx * ky * vth + kz * sth
    r22 = ky * ky * vth + cth
    r23 = ky * kz * vth - kx * sth
    r31 = kx * kz * vth - ky * sth
    r32 = ky * kz * vth + kx * sth
    r33 = kz * kz * vth + cth

    beta = math.atan2(-r31, math.sqrt(r11 * r11 + r21 * r21))

    if beta > math.radians(89.99):
        beta = math.radians(89.99)
        alpha = 0
        gamma = math.atan2(r12, r22)
    elif beta < -math.radians(89.99):
        beta = -math.radians(89.99)
        alpha = 0
        gamma = -math.atan2(r12, r22)
    else:
        cb = math.cos(beta)
        alpha = math.atan2(r21 / cb, r11 / cb)
        gamma = math.atan2(r32 / cb, r33 / cb)

    return [x, y, z, math.degrees(gamma), math.degrees(beta), math.degrees(alpha)]


def get_user_frame(
    RORG: Union[Pose, list], RXX: Union[Pose, list], RXY: Union[Pose, list]
) -> Pose:
    def get1_nx(o, x):
        temp = x - o
        nx = temp / np.linalg.norm(temp)
        return nx

    def get2_ox(o, y):
        temp = y - o
        ox = temp / np.linalg.norm(temp)
        return ox

    def get3_ax(nx, ox):
        ret = np.cross(nx, ox)
        return ret

    # 返回用户坐标系为角度
    oo = np.array([k for i, k in enumerate(RORG) if i < 3])
    xx = np.array([k for i, k in enumerate(RXX) if i < 3])
    yy = np.array([k for i, k in enumerate(RXY) if i < 3])
    nx = get1_nx(oo, xx)
    ox = get2_ox(oo, yy)
    ax = get3_ax(nx, ox)

    nx = [round(i, 6) for i in nx]
    ox = [round(i, 6) for i in ox]
    ax = [round(i, 6) for i in ax]
    user_matrix = [
        [nx[0], ox[0], ax[0], oo[0]],
        [nx[1], ox[1], ax[1], oo[1]],
        [nx[2], ox[2], ax[2], oo[2]],
        [0, 0, 0, 1],
    ]
    user_euler = matrix_to_euler(user_matrix)
    return user_euler


if __name__ == "__main__":

    def test(pose: Pose):
        pose.z = pose.z + 30
        return pose

    def test2(pose: list):
        pose[2] = pose[2] + 30
        pose = [12]
        return pose

    # a = [445.303, 471.102, -271.507]
    # b = [452.033, 1017.622, -255.829]
    # c = [-280.699, 764.023, -271.870]
    a = [
        -52.72,
        -254.55,
        130,
    ]
    b = [
        -103.02,
        -306.7,
        128.8,
    ]
    c = [
        -103.18,
        -307.07,
        151.5,
    ]
    # print(rv2rpy(3.14, 1.57, 0))
    print(get_user_frame(a, b, c))
    quit()
    user_pose = Pose([100, 100, 0, 0, 0, 0])
    user2 = [100, 100, 0, 0, 0, 0]
    c = test2(user2)
    print(c)
    print(user2)

    b = test(user_pose)
    print(b)

    print(user_pose)
