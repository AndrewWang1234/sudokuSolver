import cv2
import numpy as np
import imutils
from google.cloud import vision
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# CONFIG
# ---------------------------
SIDE = 450
CELL_SIZE = SIDE // 9

# Set this to your Google credentials JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


client = vision.ImageAnnotatorClient()

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def reorder(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4,2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    new_points[0] = points[np.argmin(s)]
    new_points[2] = points[np.argmax(s)]
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]
    return new_points

def find_sudoku_contour(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = gray.shape
    min_area = 0.15 * w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return reorder(approx)
    return None

def warp_sudoku(gray, pts):
    dst = np.array([[0,0],[SIDE-1,0],[SIDE-1,SIDE-1],[0,SIDE-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(gray, M, (SIDE, SIDE))
    return warped

def preprocess_cell(cell):
    h, w = cell.shape
    margin = int(0.05 * w)
    cell = cell[margin:h-margin, margin:w-margin]
    _, cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV)
    return cell

def ocr_cell_google(cell_image):
    """Send the cell image to Google Vision and return detected digit"""
    _, encoded_image = cv2.imencode('.png', cell_image)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.text_annotations:
        text = response.text_annotations[0].description.strip()
        text = ''.join(filter(str.isdigit, text))
        return text if text else "."
    return "."

def extract_grid_google(warped_gray):
    grid = []
    for y in range(9):
        row = []
        for x in range(9):
            y1, y2 = y*CELL_SIZE, (y+1)*CELL_SIZE
            x1, x2 = x*CELL_SIZE, (x+1)*CELL_SIZE
            cell = warped_gray[y1:y2, x1:x2]
            cell_proc = preprocess_cell(cell)
            digit = ocr_cell_google(cell_proc)
            row.append(digit)
        grid.append(row)
    return grid

def draw_grid_overlay(warped_gray, grid):
    vis = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    for y in range(9):
        for x in range(9):
            digit = grid[y][x]
            if digit != ".":
                cv2.putText(vis, digit, (x*CELL_SIZE+10,(y+1)*CELL_SIZE-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
    return vis

# ---------------------------
# MAIN LOOP
# ---------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)

MODE = "live"
snapshot_grid = None
snapshot_vis = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        pts = find_sudoku_contour(gray)
        if pts is not None:
            cv2.polylines(display, [pts.astype(int)], True, (0,255,0), 2)

        if MODE == "live":
            cv2.imshow("Webcam Feed", display)
        elif MODE in ["snapshot", "done"]:
            cv2.imshow("Sudoku Snapshot", snapshot_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Take snapshot
        if MODE == "live" and pts is not None and key == ord('s'):
            warped = warp_sudoku(gray, pts)
            snapshot_grid = extract_grid_google(warped)
            snapshot_vis = draw_grid_overlay(warped, snapshot_grid)
            MODE = "snapshot"

        # Confirm / retry
        elif MODE == "snapshot":
            if key == ord('c'):
                print("✅ Confirmed grid:")
                snapshot_grid = [[d if d != "." else "0" for d in row] for row in snapshot_grid]

                with open("sudoku)grid.txt", "w") as f:
                    for row in snapshot_grid:
                        print(row)
                        line = ' '.join(row)
                        f.write(line + '\n')
                MODE = "done"
            elif key == ord('r'):
                snapshot_grid = None
                snapshot_vis = None
                MODE = "live"

        elif MODE == "done" and key == ord('r'):
            snapshot_grid = None
            snapshot_vis = None
            MODE = "live"

finally:
    cap.release()
    cv2.destroyAllWindows()
