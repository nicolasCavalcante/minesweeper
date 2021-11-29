import itertools
import random
import numpy as np
import heapq
from test_sample_inference import call_testsample



class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count
        self.safes = set()
        self.mines = set()

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        return self.mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        return self.safes

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.mines.add(cell)

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.safes.add(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial height and width
        self.height = height
        self.width = width
        self.tot_mines = mines
        self.board = - np.ones((height, width))

        # Keep track of which cells have been clicked on
        self.moves_made = set()
        self.sequence=[]

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []
        pos = list(itertools.product(range(height), range(width)))
        prob_prior = mines / len(pos)
        self.all_cells = dict([(p, prob_prior) for p in pos])

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)
        self.sequence.append(cell)
        self.mark_safe(cell)
        self.board[cell[0], cell[1]] = count
        
        neig_cells = set()
        for x, y in itertools.product([-1, 0, 1], [-1, 0, 1]):
            new_cell = cell[0] + x, cell[1] + y
            cond = new_cell[0] >= 0 and new_cell[0] < self.height
            cond = cond and (new_cell[1] >= 0 and new_cell[1] < self.width)
            if cond:
                neig_cells.add(new_cell)
        self.add_sentence(neig_cells, count)
        # sentence = Sentence(neig_cells, count)
        # [sentence.mark_mine(cell) for cell in self.mines]
        # [sentence.mark_safe(cell) for cell in self.safes]
        # self.knowledge.append(sentence)
        # for sentence in self.knowledge:
        #     safe_maybe = sentence.cells - sentence.mines
        #     if sentence.count == len(sentence.mines):
        #         for cell in safe_maybe:
        #             self.mark_safe(cell)
        #     mine_maybe = sentence.cells - sentence.safes
        #     if len(mine_maybe) == sentence.count:
        #         for cell in mine_maybe:
        #             self.mark_mine(cell)
        # TODO: step 5
        
        # print('\n------knowledge-------')
        # for sentence in self.knowledge:
        #     print(sentence)
        # print('\n------safe-------')
        # print(self.safes)
        # print('\n------mines-------')
        # print(self.mines)
        # print('\n------moves_made-------')
        # print(self.sequence)
        # print('\n')

    def add_sentence(self, neig_cells: set, count):
        neig_cells.difference_update(self.safes)
        sentence = Sentence(neig_cells, count)
        self.knowledge.append(sentence)
        safe_maybe = sentence.cells - sentence.mines
        mine_maybe = sentence.cells - sentence.safes
        update_rem = True
        eval_knowloge = self.knowledge.copy()
        while update_rem:
            pop_id = set()
            update_rem = False
            for i, sentence in enumerate(eval_knowloge):
                safe_maybe = sentence.cells - sentence.mines
                mine_maybe = sentence.cells - sentence.safes
                if sentence.count == len(sentence.mines):
                    pop_id.add(i)
                    update_rem = True
                    for cell in safe_maybe:
                        self.mark_safe(cell)
                if len(mine_maybe) == sentence.count:
                    pop_id.add(i)
                    update_rem = True
                    for cell in mine_maybe:
                        self.mark_mine(cell)
            for i in reversed(sorted(list(pop_id))):
                eval_knowloge.pop(i)
        self.update_prob()
            

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # return None
        to_make = self.safes - self.moves_made
        choice = random.choice(list(to_make)) if to_make else None
        if choice is not None:
            print(f'I choose: {choice}')
        return choice

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # TODO: make the least likely to have bomb move
        to_make = self.all_cells.keys() - self.mines - self.moves_made
        if not to_make:
            return None
        choice = self.update_prob()
        
        # choice = random.choice(list(to_make)) if to_make else None
        # if choice is not None:
        print(f'I choose: {choice}')
        return choice

    def update_prob(self):
        grid_probs = call_testsample(self)
        for r, c in self.all_cells:
            self.all_cells[(r, c)] = 1 - grid_probs[r, c]
        # p_prior = self.tot_mines / len(self.all_cells)
        # probs = dict([(c, [p_prior]) for c in self.all_cells.keys()])
        # for sentence in self.knowledge:
        #     unknown = sentence.cells - sentence.mines - sentence.safes
        #     prob = (sentence.count - len(sentence.mines)) / max(len(unknown),1)
        #     for cell in unknown:
        #         probs[cell].append(prob)
        #     for cell in sentence.mines:
        #         probs[cell].append(1)
        #     for cell in sentence.safes:
        #         probs[cell].append(0)
        best_pick = []
        for key in self.all_cells.keys():
            if key not in self.moves_made:
                heapq.heappush(best_pick, (-self.all_cells[key], key))
        for cell in self.mines:
            self.all_cells[cell] = 0
        for cell in self.safes:
            self.all_cells[cell] = 1
        return best_pick[0][1]
        # print(1)


def joint_prob(x: list):
    x = np.array(x)
    y = np.log(x / (1 - x)).sum()
    prob = 1 / (np.exp(-y) + 1)
    return prob