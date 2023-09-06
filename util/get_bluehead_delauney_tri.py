import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
sys.path.append("..")
from util.utils import get_puppet_info

#Check if a point is insied a rectangle
def rect_contains(rect,point):
    if point[0] <rect[0]:
        return False
    elif point[1]<rect[1]:
        return  False
    elif point[0]>rect[2]:
        return False
    elif point[1] >rect[3]:
        return False
    return True

# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)

#Draw delaunay triangles
def draw_delaunay(img,subdiv, delaunay_color, points=None):
    if points is not None:
        tri_txt = "tri_txt.txt"
        txt_file = open(tri_txt, "w")

    trangleList = subdiv.getTriangleList()
    tri_num = len(trangleList)
    size = img.shape
    r = (0, 0, size[1], size[0])
    for index, t in enumerate(trangleList):
        """
        一个t中按照顺序存储了x1, y1, x2, y2, x3, y3
        """

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if points is not None:
            index_1 = points.index(pt1)
            index_2 = points.index(pt2)
            index_3 = points.index(pt3)
            if index == tri_num - 1:
                txt_file.write(str(index_1) + " " + str(index_2) + " " + str(index_3))
            else:
                txt_file.write(str(index_1) + " " + str(index_2) + " " + str(index_3) + "\n")

        if (rect_contains(r,pt1) and rect_contains(r,pt2) and rect_contains(r,pt3)):
            cv2.line(img,pt1,pt2,delaunay_color,1)
            cv2.line(img,pt2,pt3,delaunay_color,1)
            cv2.line(img,pt3,pt1,delaunay_color,1)

if __name__ == '__main__':

    lmk = np.loadtxt("examples_cartoon/bluehead_face_open_mouth.txt")[:, :2]
    lmk = lmk.tolist()
    img = cv2.imread("examples_cartoon/bluehead.jpg")

    #Define window names;
    win_delaunary = "Delaunay Triangulation"


    #Turn on animations while drawing triangles
    animate = True

    #Define colors for drawing
    delaunary_color = (255, 255, 255)
    points_color = (0, 0, 255)

    #Keep a copy   around
    img_orig = img.copy()

    #Rectangle to be used with Subdiv2D
    h, w = img.shape[:2]
    rect = (0, 0, w, h)

    #Create an instance of Subdiv2d
    subdiv = cv2.Subdiv2D(rect)

    points = []
    for p in lmk:
        x, y = p[0], p[1]
        points.append((round(x), round(y)))

    # ============================ 边界锚点处理 ============================
    points_numpy = np.array(points).reshape(68, -1)
    DEMO_CH = "bluehead"
    bound, scale, shift = get_puppet_info(DEMO_CH, ROOT_DIR='examples_cartoon')
    r = list(range(0, 48)) + list(range(60, 68))
    fl = points_numpy[r, :]
    fl_2d = fl[:, 0:2].reshape(1, 56 * 2)
    fl_2d = np.concatenate((fl_2d, np.tile(bound, (fl_2d.shape[0], 1))), axis=1)
    fl_2d = fl_2d.reshape(-1, 112 + bound.shape[1]).reshape(68, 2)
    fl_2d[:, 0] = np.clip(fl_2d[:, 0], 0, w - 1)
    fl_2d[:, 1] = np.clip(fl_2d[:, 1], 0, h - 1)

    # Draw points
    # for p in fl_2d:
    #     draw_point(img, p, (0, 0, 255))
    # plt.imshow(img)
    # plt.show()
    # =====================================================================

    points = []
    for p in fl_2d:
        x, y = p[0], p[1]
        points.append((round(x), round(y)))

    # Insert points into subdiv
    for p in points:
        # 每次将一个新的p加入到subdiv中之后会重新绘制分割结果
        subdiv.insert(p)
        # Show animate
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy,subdiv,(255,255,255))
            cv2.imshow(win_delaunary,img_copy)
            cv2.waitKey(100)

    #Draw delaunary triangles
    draw_delaunay(img,subdiv,(255,255,255), points)
    #Draw points

    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for p in points:
        draw_point(img_rgb,p,(0,0,255))
    plt.imshow(img_rgb)
    plt.show()

