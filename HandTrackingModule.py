import math
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        # 初始化参数
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        # 初始化导入模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

        # 指尖的坐标位置
        self.tipIds = [4, 8, 12, 16, 20]

    # 判断屏幕中是否存在手，并进行处理
    def findHands(self, img, draw=True):
        # 将图像翻转
        img = cv2.flip(img, 1)
        # 将BGR图像转换为RGB图像
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将RGB图像输入到模型当中， 获取预测结果
        self.results = self.hands.process(img_RGB)

        # 只要存在手
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # 遍历每一只手

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    # 获取某只手的关键点坐标信息
    def findPosition(self, img, handNo=0):

        # 分别保存所有关键点x， y的坐标信息
        xList = []
        yList = []

        # 保存当前手的关键点坐标信息
        self.lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            # 遍历手的每一个关键点信息（id，坐标）
            for id, lm in enumerate(myHand.landmark):
                # 获取图片的像素尺寸
                h, w, z = img.shape
                # 获取当前关键点在图像中的绝对坐标
                cx, cy = int(lm.x * w), int(lm.y * h)

                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])

        return self.lmList

    # 判断手势
    def getHandGesture(self):
        if self.lmList is None:
            return "No hand!"

        # 为每个手指设定一个flag判断当前开闭状态
        is_thumb_open = False
        is_index_open = False
        is_middle_open = False
        is_ring_open = False
        is_little_open = False

        # 当第二关节和指尖都小于固定点时，大拇指为张开状态
        if self.lmList[3][1] < self.lmList[2][1] and self.lmList[self.tipIds[0]][1] < self.lmList[2][1]:
            is_thumb_open = True

        # 其他四根手指可以直接循环
        for i in [6, 10, 14, 18]:

            if self.lmList[i][2] > self.lmList[i + 1][2] > self.lmList[i + 2][2]:
                if i == 6:
                    is_index_open = True
                elif i == 10:
                    is_middle_open = True
                elif i == 14:
                    is_ring_open = True
                elif i == 18:
                    is_little_open = True

        # 进行手势识别
        if not is_thumb_open and is_index_open and not is_middle_open and not is_ring_open and not is_little_open:
            return "一"
        elif not is_thumb_open and is_index_open and is_middle_open and not is_ring_open and not is_little_open:
            return "二"
        elif not is_thumb_open and is_index_open and is_middle_open and is_ring_open and not is_little_open:
            return "三"
        elif not is_thumb_open and is_index_open and is_middle_open and is_ring_open and is_little_open:
            return "四"
        elif is_thumb_open and is_index_open and is_middle_open and is_ring_open and is_little_open:
            return "五"
        elif is_thumb_open and not is_index_open and is_middle_open and is_ring_open and is_little_open and self.is_thumb_near_index():
            return "OK"
        elif not is_thumb_open and not is_index_open and not is_middle_open and not is_ring_open and not is_little_open:
            return "fist"

    def write(self, img, gesture):
        h, w = img.shape[0], img.shape[1]
        hand_dic = {}
        thumb_x = self.lmList[4][1]
        thumb_y = self.lmList[4][2]
        index_x = self.lmList[8][1]
        index_y = self.lmList[8][2]
        # choose_pt = (int((thumb_x + index_x) / 2), int((thumb_y + index_y) / 2))
        # dst = np.sqrt(np.square(thumb_x - index_x) + np.square(thumb_y - index_y))
        choose_pt = (index_x, index_y)
        click_state = False
        if gesture == "一":
            click_state = True
            cv2.circle(img, choose_pt, 10, (0, 0, 255), -1)  # 绘制点击坐标，为轨迹的坐标
            cv2.circle(img, choose_pt, 5, (255, 220, 30), -1)

        hand_dic['pt'] = choose_pt
        hand_dic['click'] = click_state

        return img, hand_dic

    # 判断食指与拇指是否足够接近
    def is_thumb_near_index(self):
        distance, _ = self.findDistance(4, 8)
        if distance < 30:
            return True
        else:
            return False

    # 两个手指关键点之间的距离
    def findDistance(self, p1, p2):
        # 两个关键点坐标
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 两个关键点的中心点

        # 求两点之间的欧氏距离
        length = math.hypot((x2 - x1), (y2 - y1))

        return length, [x1, y1, x2, y2, cx, cy]
