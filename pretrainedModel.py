import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# -------------------------------
# CONFIG
# -------------------------------
TEMPLATE_FOLDER = "betterTemplates"  # folder with 1-9 digits
CELL_SIZE = (50,50)
BLANK_PIXEL_THRESHOLD = 100  # increase threshold for faint digits
SHOW_DEBUG = False  # set True for visualization
MEMORY_FILE = "correction_memory.json"
GRID_FILE = "sudoku_grid.txt"

# -------------------------------
# UTILS
# -------------------------------
def preprocess_for_match(img):
    """Resize, equalize, and threshold an image for template matching."""
    img_resized = cv2.resize(img, CELL_SIZE)
    # img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    # img_eq = cv2.equalizeHist(img_blur)
    # img_bin = cv2.adaptiveThreshold(
    #     img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV, 21, 7
    # )
    # return img_bin
    return img_resized

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    new_points[0] = points[np.argmin(s)]
    new_points[2] = points[np.argmax(s)]
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]
    return new_points

def extract_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours_data = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sudoku_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            sudoku_contour = approx
            break
    if sudoku_contour is None:
        return None, None, None

    pts = reorder(sudoku_contour)
    side = 450
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(gray, matrix, (side, side))

    cell_size = side // 9
    cells = []
    for y in range(9):
        row = []
        for x in range(9):
            x1, y1 = x * cell_size, y * cell_size
            row.append(warped[y1:y1+cell_size, x1:x1+cell_size])
        cells.append(row)
    return warped, cells, pts

def save_grid_to_file(grid, filename=GRID_FILE):
    """Save the final corrected grid to a text file."""
    with open(filename, "w") as f:
        for row in grid:
            f.write(" ".join(map(str, row)) + "\n")
# -------------------------------
# LOAD TEMPLATES (1-9)
# -------------------------------
templates = {}
for filename in os.listdir(TEMPLATE_FOLDER):
    if filename.lower().endswith((".png", ".jpg")):
        digit = int(os.path.splitext(filename)[0])
        # if digit == 0:
        #     continue
        tmpl_img = cv2.imread(os.path.join(TEMPLATE_FOLDER, filename), cv2.IMREAD_GRAYSCALE)
        templates[digit] = preprocess_for_match(tmpl_img)

print(f"Loaded templates: {list(templates.keys())}")

# -------------------------------
# MATCH DIGIT FUNCTION
# -------------------------------
def match_digit(cell, top_n=3, show_debug=False):
    """Return top N candidate digits for a cell."""
    cell_proc = preprocess_for_match(cell)
    # If necessary, handle empty or blank cells
    # if cv2.countNonZero(cell_proc) < BLANK_PIXEL_THRESHOLD:
    #     return [(0, 1.0)]

    scores = {}
    for digit, tmpl in templates.items():
        # Try using a different matching method (e.g., TM_CCORR_NORMED)
        res = cv2.matchTemplate(cell_proc, tmpl, cv2.TM_CCOEFF_NORMED)  # Or TM_CCORR_NORMED, TM_SQDIFF_NORMED
        _, val, _, _ = cv2.minMaxLoc(res)
        scores[digit] = val

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if show_debug:
        # Show debugging output
        for digit, score in sorted_scores:
            print(f"Digit {digit}: Match score = {score:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        axes[0].imshow(cell_proc, cmap='gray')
        axes[0].set_title("Processed Cell")
        axes[0].axis('off')
        
        best_digit = sorted_scores[0][0]
        best_template = templates[best_digit]

        axes[1].imshow(best_template, cmap='gray')
        axes[1].set_title(f"Template {best_digit}")
        axes[1].axis('off')
        
        plt.suptitle(f"Match score: {sorted_scores[0][1]:.3f}")
        plt.show()

    # Adjust threshold for matching (e.g., 0.5)
    if sorted_scores[0][1] < 0.5:
        return [(0, 1.0)]  # Return 0 as the most probable match if it's below threshold

    return sorted_scores[:top_n]


# -------------------------------
# UPDATE GRID WITH CORRECTIONS
# -------------------------------
def build_grid_with_corrections(grid_candidates):
    """Build a grid, applying corrections where needed."""
    final_grid = np.zeros((9, 9), dtype=int)
    
    for y in range(9):
        for x in range(9):
            top_digit, top_conf = grid_candidates[y][x][0]
            corrected_digit = apply_corrections(top_digit, top_conf)
            final_grid[y, x] = corrected_digit
    return final_grid

def load_correction_memory():
    """Load correction memory from disk if it exists."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_correction_memory():
    """Save correction memory to disk."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(correction_memory, f, indent=4)


# -------------------------------
# CORRECTION MEMORY
# -------------------------------
correction_memory = load_correction_memory()  # {wrong_digit: {correct_digit: count}}

def apply_corrections(detected_digit, confidence):
    """
    Use correction history to decide if we should override the detected digit.
    - Only override if the misread count greatly outweighs correct count
    - AND if the confidence is not very high
    """
    detected_str = str(detected_digit)
    if detected_str not in correction_memory:
        return detected_digit

    data = correction_memory[detected_str]
    correct_count = data["correct"]
    misreads = data["misread"]

    if not misreads:
        return detected_digit

    # Find the most common misread target
    likely_digit = max(misreads, key=misreads.get)
    misread_count = misreads[likely_digit]

    # Decision rule
    if misread_count > correct_count / 3 and confidence < 0.67:
        print(f"[AUTO-CORRECTION] {detected_digit} → {likely_digit} "
        f"(confidence={confidence:.2f}, history={misread_count} vs {correct_count})")
        # Only correct if it's MUCH more likely to be misread, AND confidence is not high
        return int(likely_digit)

    return detected_digit

def add_correction(wrong_digit, correct_digit):
    if str(wrong_digit) not in correction_memory:
        correction_memory[str(wrong_digit)] = {"correct": 0, "misread": {}}
    if str(correct_digit) not in correction_memory[str(wrong_digit)]["misread"]:
        correction_memory[str(wrong_digit)]["misread"][str(correct_digit)] = 0
    correction_memory[str(wrong_digit)]["misread"][str(correct_digit)] += 1
    save_correction_memory()



def add_confirmed_correct(digit):
    if str(digit) not in correction_memory:
        correction_memory[str(digit)] = {"correct": 0, "misread": {}}
    correction_memory[str(digit)]["correct"] += 1
    save_correction_memory()



# -------------------------------
# MAIN LOOP
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

snapshot_grid = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    warped, cells, pts = extract_grid(frame)
    if pts is not None:
        pts_int = pts.reshape(-1,2).astype(int)
        cv2.polylines(display, [pts_int], True, (0,255,0), 3)

    cv2.putText(display, "s: snapshot | c: confirm | r: retry | q: quit", 
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Webcam", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and warped is not None and cells is not None:
        snapshot_grid = []
        for y in range(9):
            row_candidates = []
            for x in range(9):
                candidates = match_digit(cells[y][x], top_n=3, show_debug=SHOW_DEBUG)
                row_candidates.append(candidates)
                if SHOW_DEBUG:
                    print(f"Cell ({y},{x}) top candidates: {candidates}")
            snapshot_grid.append(row_candidates)
        print("Snapshot taken. Press 'c' to confirm or 'r' to retry")
        cv2.imshow("Warped Grid", warped)

    elif key == ord('c') and snapshot_grid is not None:
        best_grid = build_grid_with_corrections(snapshot_grid)
        print("Detected Grid (top candidate per cell, duplicates resolved):")
        print(best_grid)

        manually_corrected = set()
        while True:
            user_input = input("Entre correctionas 'row,col,correct_digit' (or 'done'): ").strip()
            if user_input.lower() == "done":
                break
            try:
                row, col, correct_digit = map(int, user_input.split(","))
                wrong_digit = best_grid[row, col]
                best_grid[row, col] = correct_digit
                add_correction(wrong_digit, correct_digit)
                manually_corrected.add((row, col))
                print(f"Corrected cell ({row},{col}) from {wrong_digit} -> {correct_digit}")
                print("Updated Grid:")
                print(best_grid)
            except Exception as e:
                print("Invalid input. Format: row,col,digit (example: 2,3,8)")

        for y in range(9):
            for x in range(9):
                if(y, x) not in manually_corrected:
                    add_confirmed_correct(best_grid[y, x])
        print("Final corrected grid:")
        print(best_grid)
        print("Corrections memory so far", correction_memory)
        save_correction_memory()
        save_grid_to_file(best_grid)

    elif key == ord('r'):
        snapshot_grid = None
        print("Retry grid detection")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
