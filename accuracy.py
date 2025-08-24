import numpy as np

# Correct Sudoku board (provided by user)
correct_board = np.array([
    [0, 0, 0, 0, 3, 0, 0, 0, 2],
    [0, 0, 0, 4, 0, 9, 0, 0, 8],
    [0, 0, 3, 0, 1, 5, 0, 0, 0],
    [1, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 6, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 9],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 6, 0, 4, 0, 0, 3, 1],
    [0, 1, 0, 0, 0, 0, 5, 6, 0]
])

# Detected Sudoku board (output from the detection function)
detected_board = np.array([
    [0, 0, 0, 0, 3, 0, 0, 0, 2],
    [0, 0, 0, 4, 0, 9, 0, 0, 3],
    [0, 0, 3, 0, 1, 5, 0, 0, 0],
    [1, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 6, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 9],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 6, 0, 4, 0, 0, 3, 1],
    [0, 1, 0, 0, 0, 0, 5, 5, 0]
])
# Calculate the number of non-zero values in the correct board
non_zero_correct = np.count_nonzero(correct_board)

# Calculate the number of matching non-zero values between the two boards
matching_values = np.sum((correct_board == detected_board) & (correct_board != 0))

# Calculate the accuracy percentage
accuracy_percentage = (matching_values / non_zero_correct) * 100

print(f"Accuracy: {accuracy_percentage}%")
