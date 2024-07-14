class BoggleGame:
    def __init__(self, board, words):
        self.board = board
        self.words = words
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.result = set()
        self.visited = [[False] * self.num_cols for _ in range(self.num_rows)]

    def is_valid(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and not self.visited[row][col]

    def dfs(self, row, col, node, path):
        if node in self.words:
            self.result.add(node)
        if not self.is_valid(row, col):
            return

        self.visited[row][col] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid(new_row, new_col):
                self.dfs(new_row, new_col, node + self.board[new_row][new_col], path + [(new_row, new_col)])

        self.visited[row][col] = False

    def find_words(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.dfs(row, col, self.board[row][col], [(row, col)])
        return list(self.result)


# Example usage:
if __name__ == "__main__":
    # Input Boggle board from user
    board = []
    num_rows = int(input("Enter number of rows: "))
    num_cols = int(input("Enter number of columns: "))

    print("Enter the board row by row:")
    for _ in range(num_rows):
        board.append(input().strip().split())

    # Input list of words from user
    words = input("Enter words separated by space: ").strip().split()

    # Initialize and solve Boggle game
    game = BoggleGame(board, words)
    found_words = game.find_words()

    # Print the result
    print("Words found in the Boggle board:", found_words)
