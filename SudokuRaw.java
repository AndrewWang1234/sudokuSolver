import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class SudokuRaw {
    public static void main(String[] args) {
        int[][] board = new int[9][9];

        try {
            Scanner scanner = new Scanner(new File("C:/Users/Andrew/SudokuProject/src/sudoku_grid.txt"));
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    if (scanner.hasNextInt()) {
                        board[i][j] = scanner.nextInt();
                    }
                }
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error: sudoku_grid.txt not found!");
            return;
        }

        // Debug print (to check it worked)
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }

        // Pass the board into your solver
        new Sudoku(board);
    }
}