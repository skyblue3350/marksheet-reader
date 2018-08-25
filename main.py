import os
import csv
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from mainwindow import Ui_MainWindow


class AutoMarker(QtCore.QThread):
    update = QtCore.pyqtSignal(list)

    def __init__(self):
        super(AutoMarker, self).__init__()

        self.stopped = False
        self.mutex = QtCore.QMutex()

    def setFunc(self, func):
        self.func = func

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        while not self.stopped:
            self.update.emit(self.func())


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, image=None, ratio=1.0):
        super(ImageWidget, self).__init__(parent=parent)
        self.image = image
        self.ratio = ratio

    def setImage(self, image):
        self.image = image
        self.update()

    def setRatio(self, ratio):
        self.ratio = ratio
        self.update()

    def getRatio(self, ratio):
        return self.ratio

    def mousePressEvent(self, event):
        super(ImageWidget, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super(ImageWidget, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super(ImageWidget, self).mouseReleaseEvent(event)

    def wheelEvent(self, event):
        super(ImageWidget, self).wheelEvent(event)

    def paintEvent(self, event):
        # 画像がない場合は描画しない
        if self.image is None:
            return

        painter = QtGui.QPainter(self)
        # painter.begin(self)

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        itemAspectRatio  = self.image.width() / self.image.height()
        sceneAspectRatio = self.width() / self.height()

        if itemAspectRatio >= sceneAspectRatio:
            self.ratio = self.width() / self.image.width()
        else:
            self.ratio = self.height() / self.image.height()

        painter.scale(self.ratio, self.ratio)

        # Scale値依存
        # painter.translate(0, 0)
        painter.drawImage(0, 0, self.image)

        painter.end()



class Utils(object):
    @classmethod
    def getMarkerPosition(cls, target, axis):
        h, w = target.shape

        # 二極化と色反転
        res, target = cv2.threshold(target, 200, 255, cv2.THRESH_BINARY)
        target = 255 - target

        # マーカー検出
        image, contrours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # マーカー重心位置取得
        result = []
        for c in contrours:
            mu = cv2.moments(c)
            x, y= int(mu["m10"] / mu["m00"]) , int(mu["m01"] / mu["m00"])
            result.append((x, y))

        result.sort(key=lambda x: x[axis])

        return result


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(QtWidgets.QMainWindow, self).__init__(parent=parent)
        self.setupUi()
        self.setupColor()
        self.reset()

        self.thread = AutoMarker()

    def threadUpdate(self, score):
        self.result.append(score)

        if self.ui.comboBox.currentIndex() + 1 == self.ui.comboBox.count():
            self.thread.terminate()

            self.threadFinish()
        else:
            self.ui.comboBox.setCurrentIndex(self.ui.comboBox.currentIndex() + 1)

    def threadFinish(self):
        self.ui.batch_button.setText("一括処理")
        self.ui.position_button.setEnabled(True)
        self.ui.marker_button.setEnabled(True)
        self.ui.score_button.setEnabled(True)
        self.ui.output_file_button.setEnabled(True)

        self.result.sort(key=lambda x: x[0])

        f = open("result.csv", "w")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(self.result)
        f.close()

        QtWidgets.QMessageBox.information(
            self,
            "処理終了",
            "結果をCSVとして出力しました。")


    def setupUi(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # イベント定義
        self.ui.input_button.clicked.connect(self.getInputDir)
        self.ui.output_button.clicked.connect(self.getOutputDir)
        self.ui.position_button.clicked.connect(self.getMarkerPosition)
        self.ui.marker_button.clicked.connect(self.getMarker)
        self.ui.score_button.clicked.connect(self.getScore)
        self.ui.output_file_button.clicked.connect(self.outputFile)
        self.ui.batch_button.clicked.connect(self.batchMark)

        self.input_viewer = ImageWidget(self)
        layout = QtWidgets.QVBoxLayout(self.ui.input_widget)
        layout.addWidget(self.input_viewer)

        self.output_viewer = ImageWidget(self)
        layout = QtWidgets.QVBoxLayout(self.ui.output_widget)
        layout.addWidget(self.output_viewer)

    def setupColor(self):
        self.POSITION_MARKER = (255, 0, 0)
        self.POSITION_MARKER_LINE = (0, 0, 255)
        self.ANSWER_MARKER = (0, 255, 0)

    def reset(self):
        self.input_viewer.setImage(None)
        self.output_viewer.setImage(None)
        self.markers_x = None
        self.markers_y = None
        self.answer = None
        self.ui.number_lcd.display("")
        self.ui.score_lcd.display("")

    def batchMark(self):
        self.thread.setFunc(self.autoFunctions)
        self.thread.update.connect(self.threadUpdate)

        if not self.ui.input_path.text():
            QtWidgets.QMessageBox.warning(
                self,
                "入力先未定義",
                "入力先が未設定です。")
            return
        if not self.ui.output_path.text():
            QtWidgets.QMessageBox.warning(
                self,
                "出力先未定義",
                "出力先が未設定です。")
            return

        if self.thread.isRunning():
            self.ui.batch_button.setText("一括処理")
            self.thread.stop()
            self.ui.position_button.setEnabled(True)
            self.ui.marker_button.setEnabled(True)
            self.ui.score_button.setEnabled(True)
            self.ui.output_file_button.setEnabled(True)
        else:
            self.result = []

            self.ui.batch_button.setText("停止")
            self.thread.start()
            self.ui.position_button.setEnabled(False)
            self.ui.marker_button.setEnabled(False)
            self.ui.score_button.setEnabled(False)
            self.ui.output_file_button.setEnabled(False)

    def autoFunctions(self):
        self.getMarkerPosition()
        self.getMarker()
        self.getScore()
        return self.outputFile(silent=True)

    def getDir(self, title):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            title,
            "./",
            QtWidgets.QFileDialog.ShowDirsOnly
        )
        return dirname

    def getInputDir(self):
        dirname = self.getDir("入力ディレクトリを選択")
        if dirname:
            self.ui.input_path.setText(dirname)

            for p in Path(dirname).glob("*.jpg"):
                self.ui.comboBox.addItem(p.name)


    def getOutputDir(self):
        dirname = self.getDir("出力ディレクトリを選択")
        if dirname:
            self.ui.output_path.setText(dirname)

    def assertMarkerCount(self, count, markers):
        if len(markers) == count:
            return True
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "マーカー未検出",
                "マーカーの数が不足しています。\n期待値：{}\n検出数：{}".format(count, len(markers)))
            return False

    def getMarkerPosition(self):
        if not self.ui.comboBox.currentText():
            return

        path = Path(self.ui.input_path.text()) / Path(self.ui.comboBox.currentText())
        target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = target.shape
        height = int(h * 0.02)
        width = int(w * 0.02)

        self.markers_x = Utils.getMarkerPosition(target[h - height:int(h - height / 10), 0:w], 0)
        self.markers_y = Utils.getMarkerPosition(target[0:h, w - width:int(w - width / 10)], 1)

        if not self.assertMarkerCount(47, self.markers_x):
            return False
        if not self.assertMarkerCount(25, self.markers_y):
            return False

        self.marker_position_preview = cv2.imread(str(path))
        self.marker_position_preview = cv2.cvtColor(self.marker_position_preview, cv2.COLOR_BGR2RGB)
        h, w, c = self.marker_position_preview.shape

        # マーカー検出位置描画
        for x in self.markers_x:
            cv2.circle(self.marker_position_preview, x, 15, self.POSITION_MARKER, -1)
            cv2.line(self.marker_position_preview, x, (x[0], h), self.POSITION_MARKER_LINE, 5)
        for y in self.markers_y:
            cv2.circle(self.marker_position_preview, y, 15, self.POSITION_MARKER, -1)
            cv2.line(self.marker_position_preview, y, (w, y[1]), self.POSITION_MARKER_LINE, 5)

        qimage = QtGui.QImage(
            self.marker_position_preview.data,
            w,
            h,
            (c * w),
            QtGui.QImage.Format_RGB888)

        self.input_viewer.setImage(qimage)

    def getMarker(self):
        if not self.ui.comboBox.currentText():
            return

        path = Path(self.ui.input_path.text()) / Path(self.ui.comboBox.currentText())
        value = self.ui.spinBox.value()
        target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        res, target = cv2.threshold(target, value, 255, cv2.THRESH_BINARY)
        self.answer = 255 - target

        self.marker_preview = self.marker_position_preview.copy()
        self.marker_preview[np.where(self.answer == 255)] = self.ANSWER_MARKER
        h, w, c = self.marker_preview.shape

        qimage = QtGui.QImage(
            self.marker_preview.data,
            w,
            h,
            (c * w),
            QtGui.QImage.Format_RGB888)

        self.input_viewer.setImage(qimage)

    def getScore(self):
        _ = [
            self.markers_x is None,
            self.markers_y is None,
            self.answer is None
        ]
        if any(_):
            return

        areas = {
            "number_x": self.markers_x[:7],
            "answer_x": list(zip(*[iter(self.markers_x[7:])]*10)),
            "number_y": self.markers_y[:10],
            "answer_y": self.markers_y,
        }

        path = Path(self.ui.input_path.text()) / Path(self.ui.comboBox.currentText())
        self.answer_preview = cv2.imread(str(path))
        h, w, c = self.answer_preview.shape

        # 学籍番号の処理
        result = ""
        for x in areas["number_x"]:
            for number, y in enumerate(areas["number_y"]):
                if self.answer[y[1]][x[0]] == 255:
                    result += str(number)
                    # 出力確認用の画像
                    radius = int(w * 0.01)
                    border = int(radius / 3)
                    cv2.circle(self.answer_preview, (x[0], y[1]), radius, self.POSITION_MARKER, border)
                    cv2.putText(
                        self.answer_preview,
                        str(number),
                        (x[0] - int(radius/2), y[1] + int(radius/2)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=int(border / 5),
                        color=self.POSITION_MARKER,
                        thickness=border)
                    break
        else:
            self.ui.number_lcd.display(result)

        # 各問題の処理
        result = []
        for i, v in enumerate(areas["answer_x"]):
            for y in areas["answer_y"]:
                ans = []
                for number, x in enumerate(areas["answer_x"][i]):
                    if self.answer[y[1]][x[0]] == 255:
                        ans.append(1)

                        # 出力確認用の画像
                        radius = int(w * 0.01)
                        border = int(radius / 3)
                        cv2.circle(self.answer_preview, (x[0], y[1]), radius, self.POSITION_MARKER, border)
                        cv2.putText(
                            self.answer_preview,
                            str(number + 1),
                            (x[0] - int(radius/2), y[1] + int(radius/2)),
                            cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(border / 5),
                            color=self.POSITION_MARKER,
                            thickness=border)
                    else:
                        ans.append(0)
                else:
                    result.append(ans)

        qimage = QtGui.QImage(
            self.answer_preview.data,
            w,
            h,
            (c * w),
            QtGui.QImage.Format_RGB888)
        self.output_viewer.setImage(qimage)

        # 採点処理
        f = csv.reader(open("answer.csv"))
        header = next(f)

        score = 0
        for i, row in enumerate(f):
            # print(i+1, "問目", end="")
            
            if [ 1 if r == "x" else 0 for r in row[1:]] == result[i]:
                # print("正解")
                score += 1
            else:
                pass
                # print("不正解")
        else:
            self.ui.score_lcd.display(str(score))

    def outputFile(self, silent=False):
        if not self.ui.output_path.text():
            QtWidgets.QMessageBox.warning(
                self,
                "出力先未定義",
                "出力先が未設定です。")
            return
        if not self.ui.comboBox.currentText():
            return
        if self.answer_preview is None:
            return

        path = Path(self.ui.output_path.text()) / Path(
            str(int(self.ui.number_lcd.value()))
            + "_" + str(int(self.ui.score_lcd.value())) + "_"
            + str(Path(self.ui.comboBox.currentText())))
        cv2.imwrite(str(path), self.answer_preview)

        if silent:
            pass
        else:
            QtWidgets.QMessageBox.information(
                self,
                "出力終了",
                "出力が完了しました")
        return [
            str(int(self.ui.number_lcd.value())),
            int(self.ui.score_lcd.value())
        ]

if __name__ == "__main__":
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal("")
    sys.excepthook = excepthook

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
