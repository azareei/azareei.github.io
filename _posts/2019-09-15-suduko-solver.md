---
layout: page
title: Sudoku solver with Pyhton
date: 2019-09-15 15:09:00-0400
description: How to solve Sudoku with Python?
---

<br>


<p>
The following code finds a solution to the Sudoku puzzle using recursive methods. If multiple solution exists, this code only finds one of the solutions that needs less computation. You can tweak the code to find all of the solutions. You can find this on my <a href="https://github.com/ahmadzareei/Sudoku">github</a>.
</p>

<br>
<br>

{% highlight python linenos %}

import numpy as np
import copy


def checkRules(puzzle):
    """ this function receives a sudoku puzzle as a 9x9 list.
    and checks if it satisfies the rules of Sudoku, specifically
    (i): if all the numbers in rows are unique.
    (ii): if all the numbers in columns are unique
    (iii): if all the numbers in cells are unique"""
    

    # Checking condition (i)
    # Checking numbers to be unique in rows    
    rowCheck = True;
    for i in range(9):
        for j in range(9):
            if not puzzle[i][j]==0:
               if puzzle[i][:].count(puzzle[i][j]) != 1:
                   rowCheck = False;

    # Checking condition (ii)
    # checking to be unique in columns
    colCheck = True;
    for i in range(9):
        col = [row[i] for row in puzzle]
        for j in range(9):
            if not col[j]==0:
               if col.count(col[j]) != 1:
                   colCheck = False;


    # Checking condition (iii)
    # Checking numbers to be unique in each cell
    cellCheck = True;
    for i in range(3):
        for j in range(3):
            cell = [];
            cell = [row[3*i:3*(i+1)] for row in puzzle[3*i:3*(i+1)]];
            cell_flat = [];
            for row in cell:
                cell_flat = cell_flat + row;
            for k in range(9):
                if not cell_flat[k]==0:
                    if cell_flat.count(cell_flat[k])!=1:
                        cellCheck=False;
                        
    return rowCheck and colCheck and cellCheck


def possibilities(puzzle, index1, index2):
    """ This function receives a puzzle, and two indices
    for the row (index1), and column (index2), and retuns all the possibilites 
    of filling the tile at puzzle[index1][index2]
    """
    # The row corresponding to the tile puzzle[index1][index2]
    row = puzzle[index1][:];
    # The column corresponding to the tile puzzle[index1][index2]
    col = [ROW[index2] for ROW in puzzle];
    # The cell containing the tile puzzle[index1][index2]
    cell = [];
    cell = [row[3*(index2//3):3*(index2//3+1)] for row in puzzle[3*(index1//3):3*(index1//3+1)]];
    cell_flat = [];
    for r in cell:
        cell_flat = cell_flat + r;

    # finding numbers that are not in row, col, or cell

    out = []; # will contain all the possibilites
    for i in range(9):
        integer = i + 1;
        if not((integer  in row) or (integer in col) or (integer in cell_flat)):
            out.append(integer) # appending the number to out if its not in row, col, and cell
    return out

def solve_sodoku(puzzle):
    """ This function solves the sudoku puzzle
    and returns the solution
    and a conditional which is true if the puzzle is solvable and
    False if not solvable.
    """
    # Checking if the puzzle satisfy the rules of the puzzle
    # and is already a solved puzzle
    if (checkRules(puzzle)):
        if np.sum(puzzle) == 405:
            return puzzle, True;
    else:
        return [], False # if the puzzle doesn't satisfy the conditions of the puzzle

    cond = True;

    # This while loop fills in all the tiles
    # that only one number can fill them
    # it also stores the tile with the smallest number of possibilities
    while cond:
        sol=copy.deepcopy(puzzle); # making a copy of the puzzle
        num = 0; # number of possible changes
        iBest = 0;
        jBest= 0 ;
        numPos= 9; # the maximum number of possiblities 
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    sol[i][j] = possibilities(puzzle, i, j)
                    if len(sol[i][j]) ==1:
                        num = num + 1;
                    else : # the length is greater than 1; we try to find the minimum
                        if len(sol[i][j])<numPos:
                            numPos = len(sol[i][j]);
                            iBest = i;
                            jBest = j;
        if num == 0: # there is no element with 1 possiblity to change
            cond=False;
        else:
            for i in range(9):
                for j in range(9):
                    if puzzle[i][j] == 0:
                        if len(sol[i][j]) == 1:
                            puzzle[i][j] = sol[i][j][0];
                                

    if np.sum(puzzle) == 405: # if the puzzle is solved, we return
        return puzzle, True

    # If the puzzle is not solved yet, we need to search for all
    # possibilities in a tile with smallest number of possibilites
    # to save some time!
    for i in range(len(sol[iBest][jBest])):
            puz_help = copy.deepcopy(puzzle)
            puz_help[iBest][jBest] = sol[iBest][jBest][i];
            puz, con = solve_sodoku(puz_help);
            if con:
                return puz, con;
            else:
                    continue
    return [],False
                                
    


if __name__=='__main__':
    
    # Easy puzzle
    puzzle1 = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
              [6, 0, 0, 1, 9, 5, 0, 0, 0],
              [0, 9, 8, 0, 0, 0, 0, 6, 0],
              [8, 0, 0, 0, 6, 0, 0, 0, 3],
              [4, 0, 0, 8, 0, 3, 0, 0, 1],
              [7, 0, 0, 0, 2, 0, 0, 0, 6],
              [0, 6, 0, 0, 0, 0, 2, 8, 0],
              [2, 0, 0, 4, 1, 9, 0, 0, 5],
              [3, 0, 0, 0, 8, 0, 0, 7, 9]];    
    # Hard puzzle
    puzzle2 = [[7, 0, 1, 4, 0, 6, 3, 0, 2],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [3, 0, 0, 9, 0, 1, 0, 0, 6],
               [4, 0, 6, 0, 0, 0, 2, 0, 7],
               [0, 0, 0, 0, 4, 0, 0, 0, 0],
               [9, 0, 2, 0, 0, 0, 8, 0, 4],
               [1, 0, 0, 3, 0, 7, 0, 0, 8],
               [0, 0, 0, 0 ,0, 0, 0, 0, 0],
               [6, 0, 4, 5, 0, 2, 1, 0, 9]];


    print('---------------Solving an easy puzzle ------------')
    print("The puzzle:")
    for r in puzzle1:
        print(r);
    sol,con = solve_sodoku(puzzle1)
    print("The solution is:")
    for r in sol:
        print(r)
    print('---------------Solving a hard puzzle ------------')        
    print("The puzzle:")
    for r in puzzle2:
        print(r);
    sol,con = solve_sodoku(puzzle2)
    print("The solution is:")
    for r in sol:
        print(r)    

{% endhighlight %}
