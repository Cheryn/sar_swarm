# From https://stackoverflow.com/questions/30023763/how-to-make-an-interactive-2d-grid-in-a-window-in-python

from tkinter import *
import numpy as np

# todo: matrix saving does not save the last cell clicked, and does not reset cells that are erased
class Cell():
    FILLED_COLOR_BG = "green"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "green"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size = size
        self.fill = False
        self.value = 0

    def _switch(self):
        """ Switch if the cell is filled or not. """
        self.fill = not self.fill

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master is not None :
            fill = Cell.FILLED_COLOR_BG
            outline = Cell.FILLED_COLOR_BORDER
            self.value = 1

            if not self.fill:
                fill = Cell.EMPTY_COLOR_BG
                outline = Cell.EMPTY_COLOR_BORDER
                self.value = 0

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = fill, outline = outline)

class CellGrid(Canvas):
    def __init__(self,master, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize
        self.grid = []
        for row in range(rowNumber):
            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        #memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        self.matrix = np.matrix(np.zeros((rowNumber, columnNumber)))

        #bind click action
        self.bind("<Button-1>", self.handleMouseClick)
        #bind moving while clicking
        self.bind("<B1-Motion>", self.handleMouseMotion)
        #bind release button action - clear the memory of midified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        self.bind("<Double-Button-1>", self.save)

        self.draw()

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()
                self.matrix[cell.ord, cell.abs] = cell.value

        with open("map3.txt", "w") as file:
            for line in self.matrix:
                np.savetxt(file, line, fmt='%.0f')
                file.write("\n")

    def _eventCoords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        cell._switch()
        cell.draw()
        #add the cell to the list of cell switched during the click
        self.switched.append(cell)
        #self.draw()

    def handleMouseMotion(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]

        if cell not in self.switched:
            cell._switch()
            cell.draw()
            self.switched.append(cell)
        #self.draw()

    def save(self, event):
        print(event)
        self.draw()

if __name__ == "__main__" :
    app = Tk()

    grid = CellGrid(app, 50, 50, 10)
    grid.pack()

    app.mainloop()