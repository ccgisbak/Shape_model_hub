import numpy as np
import math


# 计算两点距离
def cal_dist(point_1, point_2):
    dist = np.sqrt(np.sum(np.power((point_1-point_2), 2)))
    return dist


# 计算两条线的夹角
def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B


# 计算线条的方位角
def azimuthAngle(point_0, point_1):
    x1, y1 = point_0
    x2, y2 = point_1

    if x1 < x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x2 - x1))
            ang = ang * 180 / math.pi
            return ang
        elif y1 > y2:
            ang = math.atan((y1 - y2) / (x2 - x1))
            ang = ang * 180 / math.pi
            return 90 + (90 - ang)
        elif y1==y2:
            return 0
    elif x1 > x2:
        if y1 < y2:
            ang = math.atan((y2-y1)/(x1-x2))
            ang = ang*180/math.pi
            return 90+(90-ang)
        elif y1 > y2:
            ang = math.atan((y1-y2)/(x1-x2))
            ang = ang * 180 / math.pi
            return ang
        elif y1==y2:
            return 0

    elif x1==x2:
        return 90
# 线生成函数
def line(p1, p2):
    A = (p1[1] - p2[1]) + 1e-5
    B = (p2[0] - p1[0]) + 1e-5
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


# 计算两条直线之间的交点
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# 计算两个平行线之间的距离
def par_line_dist(L1, L2):
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    # print(L1,L2)

    new_A1 = 1
    new_B1 = B1 / A1
    new_C1 = C1 / A1

    new_A2 = 1
    new_B2 = B2 / A2
    new_C2 = C2 / A2

    dist = (np.abs(new_C1-new_C2))/(np.sqrt(new_A2*new_A2+new_B2*new_B2))
    if A1==0  or A2==0:
        print(L1,L2,dist)
    return dist


# 计算点在直线的投影位置
def point_in_line(m, n, x1, y1, x2, y2):
    x = (m * (x2 - x1) * (x2 - x1) + n * (y2 - y1) * (x2 - x1) + (x1 * y2 - x2 * y1) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    y = (m * (x2 - x1) * (y2 - y1) + n * (y2 - y1) * (y2 - y1) + (x2 * y1 - x1 * y2) * (x2 - x1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return (x, y)



# 顺时针旋转
def Nrotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)

    dest_x = (src_x - center_x) * math.cos(radian) + (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_y - center_y) * math.cos(radian) - (src_x - center_x) * math.sin(radian) + center_y

    # return (int(dest_x), int(dest_y))
    return (dest_x, dest_y)


# 逆时针旋转
def Srotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)

    dest_x = (src_x - center_x) * math.cos(radian) - (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_x - center_x) * math.sin(radian) + (src_y - center_y) * math.cos(radian) + center_y

    # return [int(dest_x), int(dest_y)]
    return (dest_x, dest_y)



