# Sudoku Scanner with Google OCR

## Description
This app uses OpenCV and Google Cloud Vision API to detect Sudoku puzzles from a webcam feed, extract the grid, and recognize digits with OCR. There is also a version that uses local template matching and normalized cross-correlation that is less accurate.

## Features
- Real-time Sudoku grid detection
- OCR digit extraction with Google Vision API or localized template matching algorithm
- Saves recognized grid to text file
- Interactive snapshot and confirmation conrtols
- Correction memory for improving recognition over time
- Sudoku solver with Dancing Links Algorithm

## Controls
- s: Take a snapshot of the detected Sudoku grid
- c: Confirm the detected grid and stat correction mode
- r: Rety the scan (discard the current snapshot)
- q: Quit the application


## Installation
### 1. Clone the repository  
Open your terminal or command prompt and run:  
git clone https://github.com/AndrewWang1234/sudokuSolver.git  
cd sudokuSolver

### 2. Install Dependencies
Open your terminal or command prompt and run: 
pip install -r requirements.txt

### 3. Contact
Please let me know if there are any bugs or issues with my project and the instructions. This is my first time posting a project to github so I am still very new to this stuff.  
gmail: w.andrew9504@gmail.com  
phone: 512 774 0641
