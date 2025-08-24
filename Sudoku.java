import java.util.ArrayList;
import java.util.List;

public class Sudoku {

    static class Node {
        Node L, R, U, D;
        Column C;
        int r, c, d;

        Node() {}

        Node(Column col, int r, int c, int d) {
            this.C = col;
            this.r = r;
            this.c = c;
            this.d = d;
            U = D = this;
            L = R = this;
        }
    }

    static final class Column extends Node {
        int size;

        Column(int name) {
            super();
            this.C = this;
            this.size = 0;
            U = D = this;
            L = R = this;
        }
    }

    private Column ROOT;
    private Column[] cols;

    private void buildColumns() {
        ROOT = new Column(-1);
        cols = new Column[numCols];
         Column prev = ROOT;
         for (int i = 0; i < numCols; i++) {
            Column col = new Column(i);

            col.L = prev;
            col.R = prev.R;
            prev.R.L = col;
            prev.R = col;

            cols[i] = col;
            prev = col;
         }
    }

    private void insertCandidate(Candidate candidate) {
        Node first = null;
        Node prev = null;

        for (int colID : candidate.colIDs) {
            Column col = cols[colID];
            Node node = new Node(col, candidate.r, candidate.c, candidate.d);

            node.D = col;
            node.U = col.U;
            col.U.D = node;
            col.U = node;
            col.size++;

            if (first == null) {
                first = node;
                prev = node;
            } else {
                node.L = prev;
                node.R = prev.R;
                prev.R.L = node;
                prev.R = node;
                prev = node;
            }
        }
    }

    private void buildDLXRows() {
        for (Candidate candidate : candidates) {
            insertCandidate(candidate);
        }
    }

    private void cover(Column c) {
        c.R.L = c.L;
        c.L.R = c.R;
        for (Node i = c.D; i != c; i = i.D) {
            for (Node j = i.R; j != i; j = j.R) {
                j.D.U = j.U;
                j.U.D = j.D;
                j.C.size--;
            }
        }
    }

    private void uncover(Column c) {
        for (Node i = c.U; i != c; i = i.U) {
            for (Node j = i.L; j != i; j = j.L) {
                j.C.size++;
                j.D.U = j;
                j.U.D = j;
            }
        }
        c.R.L = c;
        c.L.R = c;
    }

    private Column chooseColumn() {
        Column best = null;
        int minSize = Integer.MAX_VALUE;

        for (Column c = (Column) ROOT.R; c != ROOT; c = (Column) c.R) {
            if (c.size < minSize) {
                minSize = c.size;
                best = c;
            }
        }
        return best;
    }

    private void printBoard() {
        for (int r = 0; r < n; r++) {
            if (r % 3 == 0 && r != 0) {
                System.out.println("------+-------+------"); // horizontal box separator
            }
            for (int c = 0; c < n; c++) {
                if (c % 3 == 0 && c != 0) {
                    System.out.print("| "); // vertical box separator
                }
                System.out.print(board[r][c] + " ");
            }
            System.out.println();
        }
    }


    private List<Node> solution = new ArrayList<>();

    private boolean solved = false;

    private void search(int k) {
        if (solved) return;
        if (ROOT.R == ROOT) {
            System.out.println("Solution Found!");
            for (Node n : solution) {
                board[n.r][n.c] = n.d;
            }
            printBoard();
            solved = true;
            //All columns covered -> solution found
            return;
        }

        Column c = chooseColumn(); //picks column with least nodes
        cover(c);

        for (Node r = c.D; r != c; r = r.D) {
            solution.add(r);
            for (Node j = r.R; j != r; j = j.R) {
                cover(j.C);
            }

            search(k + 1);

            solution.remove(solution.size() - 1);
            for (Node j = r.L; j != r; j = j.L) {
                uncover(j.C); // backtrack
            }
        }

        uncover(c); //restore column chosen

    }

    int n = 9;
    //int box = 3;
    int numCols = 324;
    List<Candidate> candidates;
    int cellRule, colRule, boxRule, rowRule;
    private int[][] board;

    public Sudoku(int[][] board) {
        candidates = new ArrayList<>();
        this.board = board;
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                for (int i = 1; i < 10; i++) {
                    cellRule = r * 9 + c;
                    rowRule = 81 + r * 9 + (i - 1);
                    colRule = 162 + c * 9 + (i - 1);
                    boxRule = 243 + ((r /3) * 3 + (c / 3)) * 9 + (i - 1);
                    int[] colIDs = new int[]{cellRule, rowRule, colRule, boxRule};
                    candidates.add(new Candidate(r, c, i, colIDs));
                }

            }
        }
        buildColumns();
        buildDLXRows();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] != 0) {
                    int digit = board[i][j];

                    for (Candidate candidate : candidates) {
                        if (candidate.r == i && candidate.c == j && candidate.d == digit) {
                            Node rowNode = null;
                            for (int colID : candidate.colIDs) {
                                Column col = cols[colID];
                                for (Node n = col.D; n != col; n =n.D) {
                                    if (n.r == i && n.c == j && n.d == digit) {
                                        rowNode = n; 
                                        break;
                                    }
                                }
                                if (rowNode != null) break;
                            }
                            if (rowNode != null) {
                                solution.add(rowNode);

                                for (Node n = rowNode; ; n = n.R) {
                                    cover(n.C);
                                    if (n.R == rowNode) break;
                                }
                            }
                        }
                    }
                }
            }
        }
        System.out.println("Starting search...");
        search(0);
        System.out.println("Search finished.");
    }

}


