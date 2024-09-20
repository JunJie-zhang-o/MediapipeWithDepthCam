#!/usr/bin/env python3
# coding=utf-8
"""
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-07 11:28:16
LastEditTime : 2024-08-07 11:47:32
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : RTDE Interface
"""

# todo:To be improved

from ur.eseries.rtde import RTDE

if __name__ == "__main__":

    r = RTDE(hostname="192.168.40.127")

    r.connect()

    print(r.get_controller_version())
    print(r.negotiate_protocol_version())

    r.send_input_setup()
    r.send_output_setup()
