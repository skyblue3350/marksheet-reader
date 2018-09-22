import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def open_dir(path):
    p = Path(path)

    if not p.exists():
        raise argparse.ArgumentTypeError("not exists : {}".format(p))

    if not p.is_dir():
        raise argparse.ArgumentTypeError("not dir : {}".format(p))

    return p


class MarkSheetResult(object):
    def __init__(self, path, number, question, score):
        self.path = path
        self.number = number
        self.question = question
        self.score = score

    def __repr__(self):
        return "{} student: {} score: {}".format(self.__class__.__name__, self.number, self.score)


class MarkSheetParser(object):
    def __init__(self, path: Path, thresh: int):
        self.path = path
        self.thresh = thresh
        self.color_image = np.array(Image.open(self.path.open("rb")).convert("L"))
        _, self.image = cv2.threshold(self.color_image, self.thresh, 255, cv2.THRESH_BINARY)
        self.image = 255 - self.image
        self.h, self.w = self.image.shape

    def trackPoisiton(self) -> (list, list):
        width = int(self.w * 0.03)
        height = int(self.h * 0.03)

        padding = 10
        for w in range(self.w // padding):
            start_h = self.h - padding * (w + 1)
            end_h = self.h - padding * w

            markers_x = self.__trackPosition(self.image[start_h:end_h, 0:self.w], 0)
            if len(markers_x) == 47:
                break
        else:
            raise IndexError("cant find width marker")

        for h in range(self.h // padding):
            start_w = self.w - padding * (h + 1)
            end_w = self.w - padding * h

            markers_y = self.__trackPosition(self.image[0:self.h, start_w:end_w], 1)
            if len(markers_y) == 25:
                break
        else:
            raise IndexError("cant find height marker")

        return markers_x, markers_y

    def __trackPosition(self, image: np.ndarray, axis: int) -> list:
        # マーカー検出
        image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # マーカー重心位置取得
        result = []
        for c in contours:
            mu = cv2.moments(c)
            try:
                x, y = int(mu["m10"] / mu["m00"]) , int(mu["m01"] / mu["m00"])
            except ZeroDivisionError:
                pass
            else:
                result.append((x, y))

        result.sort(key=lambda x: x[axis])
        return result

    def getNumber(self, markers_x: list, markers_y: list) -> str:
        student_number = ""
        for x in markers_x[:7]:
            for number, y in enumerate(markers_y[:10]):
                # 上から順に探して見つけたらその行は終了
                if self.image[y[1]][x[0]]:
                    student_number += str(number)
                    break
        else:
            return student_number

    def getQuestion(self, markers_x: list, markers_y: list) -> list:
        result = []
        for i, v in enumerate(list(zip(*[iter(markers_x[7:])]*10))):
            for y in markers_y:
                ans = []
                # 交点位置を参照して色が塗られてるかチェック
                for number, x in enumerate(v):
                    # マークされていたら1
                    if self.image[y[1]][x[0]]:
                        ans.append(1)
                    else:
                        ans.append(0)
                else:
                    # 1行終わったら1問追加
                    result.append(ans)
        else:
            return np.asarray(result)


class MarkSheetReader(object):
    def __init__(self, args):
        if args.config:
            self.config = self.load_config(args.config)
        else:
            self.config = args

        self.load_answer()

    def load_config(self, path: Path) -> list:
        pass

    def load_answer(self):
        f = csv.reader(self.config.answer)
        header = next(f)

        self.answer = []
        for row in f:
            row = row[1:]
            row = np.asarray(row, dtype=bool).astype(int)

            if not row.shape[0] == 10:
                raise SyntaxError

            self.answer.append(row)

        if not len(self.answer) == 100:
            raise EOFError

    def __iter__(self):
        for ext in self.config.ext:
            for p in self.config.input.glob("*." + ext):
                parser = MarkSheetParser(p, self.config.thresh)
                x, y = parser.trackPoisiton()

                number = parser.getNumber(x, y)
                question = parser.getQuestion(x, y)

                score = 0
                for i, q in enumerate(question):
                    if np.allclose(q, self.answer[i]):
                        score += 1

                yield MarkSheetResult(p, number, question, score)

        raise StopIteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Mode Marksheet parser")
    parser.add_argument("-i", "--input", type=open_dir, required=True, help="input directory")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    parser.add_argument("-t", "--thresh", type=int, required=False, default=240, help="threshold value")
    parser.add_argument("-e", "--ext", type=str, required=False, default=["jpg", "png", "gif"], nargs="+",
                        help="target file extension")
    parser.add_argument("-a", "--answer", type=argparse.FileType("r"), required=True, help="answer csv file")
    parser.add_argument("-c", "--config", type=argparse.FileType("r"), required=False, help="config file path TBD")

    args = parser.parse_args()

    reader = MarkSheetReader(args)

    result = []
    for sheet in reader:
        print(sheet.path, sheet)
        result.append({
            "number": sheet.number,
            "score": sheet.score,
        })

    writer = csv.DictWriter(args.output, lineterminator="\n", fieldnames=result[0].keys())
    writer.writeheader()
    writer.writerows(result)
