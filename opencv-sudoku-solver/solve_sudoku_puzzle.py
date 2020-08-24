# -----------------------------
#   USAGE
# -----------------------------
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image sudoku_puzzle.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.sudoku.puzzle import extract_digit
from pyimagesearch.sudoku.puzzle import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to trained digit classifier")
ap.add_argument("-i", "--image", required=True, help="Path to input sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="Whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# Load the digit classifier from disk
print("[INFO] Loading digit classifier...")
model = load_model(args["model"])

# Load the input image from disk and resize it
print("[INFO] Processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# Find the puzzle in the image
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

# Initialize the 9x9 sudoku board
board = np.zeros((9, 9), dtype="int")

# A sudoku puzzle is a 9x9 grid (81 individual cells), so we can infer the location of each cell
# by dividing the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# Initialize the list to store the (x, y) coordinates of each cell location
cellLocs = []

# Loop over the grid locations
for y in range(0, 9):
    # Initialize the current list of cell locations
    row = []
    for x in range(0, 9):
        # Compute the starting and ending (x, y) coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY
        # Add the (x, y) coordinates to the cell locations list
        row.append((startX, startY, endX, endY))
        # Crop the cell from the wrapped transform image and then extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=args["debug"] > 0)
        # Verify that the digit is not empty
        if digit is not None:
            foo = np.hstack([cell, digit])
            cv2.imshow("Cell/Digit", foo)
            # Resize the cell to 28x28 pixels and then prepare the cell for classification
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # Classify the digit and update the sudoku board with the prediction
            pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = pred
        # Add the row to the cell locations
        cellLocs.append(row)

# Construct a sudoku puzzle from the board
print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

# Solve the sudoku puzzle
print("[INFO] solving sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()

# Loop over the cell locations and board
for (cellRow, boardRow) in zip(cellLocs, solution.board):
    # Loop over individual cell in the row
    for (box, digit) in zip(cellRow, boardRow):
        # Unpack the cell coordinates
        startX, startY, endX, endY = box
        # Compute the coordinates of where the digit will be drawn on the output puzzle image
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY
        # Draw the result digit on the sudoku puzzle image
        cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# Show the output image
cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)