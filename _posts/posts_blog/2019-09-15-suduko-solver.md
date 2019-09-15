---
layout: posts-article
title: Sudoku solver with Pyhton
category: blog
---

<br>


<h2> Sudoku solver with Python </h2>


<p>
The following code finds a solution to the Sudoku puzzle.
</p>



<div class="org-src-container">
<pre class="src src-python">
<span style="color: #b4fa70; font-weight: bold;">import</span> numpy <span style="color: #b4fa70; font-weight: bold;">as</span> np
<span style="color: #b4fa70; font-weight: bold;">import</span> copy


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">checkRules</span>(puzzle):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #9FC59F;">""" this function receives a sudoku puzzle as a 9x9 list.</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   and checks if it satisfies the rules of Sudoku, specifically</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   (i): if all the numbers in rows are unique.</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   (ii): if all the numbers in columns are unique</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   (iii): if all the numbers in cells are unique"""</span>


<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking condition (i)</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking numbers to be unique in rows    </span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">rowCheck</span> = <span style="color: #e9b2e3;">True</span>;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> j <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #b4fa70; font-weight: bold;">not</span> puzzle[i][j]==0:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  <span style="color: #b4fa70; font-weight: bold;">if</span> puzzle[i][:].count(puzzle[i][j]) != 1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  <span style="color: #fcaf3e;">rowCheck</span> = <span style="color: #e9b2e3;">False</span>;

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking condition (ii)</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">checking to be unique in columns</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">colCheck</span> = <span style="color: #e9b2e3;">True</span>;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">col</span> = [row[i] <span style="color: #b4fa70; font-weight: bold;">for</span> row <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle]
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> j <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #b4fa70; font-weight: bold;">not</span> col[j]==0:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  <span style="color: #b4fa70; font-weight: bold;">if</span> col.count(col[j]) != 1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  <span style="color: #fcaf3e;">colCheck</span> = <span style="color: #e9b2e3;">False</span>;


<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking condition (iii)</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking numbers to be unique in each cell</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cellCheck</span> = <span style="color: #e9b2e3;">True</span>;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(3):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> j <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(3):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell</span> = [];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell</span> = [row[3*i:3*(i+1)] <span style="color: #b4fa70; font-weight: bold;">for</span> row <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle[3*i:3*(i+1)]];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell_flat</span> = [];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> row <span style="color: #b4fa70; font-weight: bold;">in</span> cell:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell_flat</span> = cell_flat + row;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> k <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #b4fa70; font-weight: bold;">not</span> cell_flat[k]==0:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> cell_flat.count(cell_flat[k])!=1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cellCheck</span>=<span style="color: #e9b2e3;">False</span>;

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> rowCheck <span style="color: #b4fa70; font-weight: bold;">and</span> colCheck <span style="color: #b4fa70; font-weight: bold;">and</span> cellCheck


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">possibilities</span>(puzzle, index1, index2):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #9FC59F;">""" This function receives a puzzle, and two indices</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   for the row (index1), and column (index2), and retuns all the possibilites </span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   of filling the tile at puzzle[index1][index2]</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   """</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">The row corresponding to the tile puzzle[index1][index2]</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">row</span> = puzzle[index1][:];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">The column corresponding to the tile puzzle[index1][index2]</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">col</span> = [ROW[index2] <span style="color: #b4fa70; font-weight: bold;">for</span> ROW <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">The cell containing the tile puzzle[index1][index2]</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell</span> = [];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell</span> = [row[3*(index2//3):3*(index2//3+1)] <span style="color: #b4fa70; font-weight: bold;">for</span> row <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle[3*(index1//3):3*(index1//3+1)]];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell_flat</span> = [];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> r <span style="color: #b4fa70; font-weight: bold;">in</span> cell:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cell_flat</span> = cell_flat + r;

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">finding numbers that are not in row, col, or cell</span>

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">out</span> = []; <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">will contain all the possibilites</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">integer</span> = i + 1;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #b4fa70; font-weight: bold;">not</span>((integer  <span style="color: #b4fa70; font-weight: bold;">in</span> row) <span style="color: #b4fa70; font-weight: bold;">or</span> (integer <span style="color: #b4fa70; font-weight: bold;">in</span> col) <span style="color: #b4fa70; font-weight: bold;">or</span> (integer <span style="color: #b4fa70; font-weight: bold;">in</span> cell_flat)):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   out.append(integer) <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">appending the number to out if its not in row, col, and cell</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> out

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">solve_sodoku</span>(puzzle):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #9FC59F;">""" This function solves the sudoku puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   and returns the solution</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   and a conditional which is true if the puzzle is solvable and</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   False if not solvable.</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span><span style="color: #9FC59F;">   """</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Checking if the puzzle satisfy the rules of the puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">and is already a solved puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> (checkRules(puzzle)):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> np.<span style="color: #e090d7; font-weight: bold;">sum</span>(puzzle) == 405:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> puzzle, <span style="color: #e9b2e3;">True</span>;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">else</span>:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> [], <span style="color: #e9b2e3;">False</span> <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">if the puzzle doesn't satisfy the conditions of the puzzle</span>

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cond</span> = <span style="color: #e9b2e3;">True</span>;

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">This while loop fills in all the tiles</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">that only one number can fill them</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">it also stores the tile with the smallest number of possibilities</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">while</span> cond:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">sol</span>=copy.deepcopy(puzzle); <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">making a copy of the puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">num</span> = 0; <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">number of possible changes</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">iBest</span> = 0;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">jBest</span>= 0 ;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">numPos</span>= 9; <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">the maximum number of possiblities </span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> j <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> puzzle[i][j] == 0:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   sol[i][j] = possibilities(puzzle, i, j)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #e090d7; font-weight: bold;">len</span>(sol[i][j]) ==1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">num</span> = num + 1;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">else</span> : <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">the length is greater than 1; we try to find the minimum</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #e090d7; font-weight: bold;">len</span>(sol[i][j])&lt;numPos:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">numPos</span> = <span style="color: #e090d7; font-weight: bold;">len</span>(sol[i][j]);
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">iBest</span> = i;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">jBest</span> = j;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> num == 0: <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">there is no element with 1 possiblity to change</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">cond</span>=<span style="color: #e9b2e3;">False</span>;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">else</span>:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> j <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(9):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> puzzle[i][j] == 0:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #e090d7; font-weight: bold;">len</span>(sol[i][j]) == 1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   puzzle[i][j] = sol[i][j][0];


<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> np.<span style="color: #e090d7; font-weight: bold;">sum</span>(puzzle) == 405: <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">if the puzzle is solved, we return</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> puzzle, <span style="color: #e9b2e3;">True</span>

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">If the puzzle is not solved yet, we need to search for all</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">possibilities in a tile with smallest number of possibilites</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">to save some time!</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> i <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">range</span>(<span style="color: #e090d7; font-weight: bold;">len</span>(sol[iBest][jBest])):
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">puz_help</span> = copy.deepcopy(puzzle)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   puz_help[iBest][jBest] = sol[iBest][jBest][i];
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">puz</span>, <span style="color: #fcaf3e;">con</span> = solve_sodoku(puz_help);
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">if</span> con:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> puz, con;
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">else</span>:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">continue</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">return</span> [],<span style="color: #e9b2e3;">False</span>




<span style="color: #b4fa70; font-weight: bold;">if</span> <span style="color: #e090d7; font-weight: bold;">__name__</span>==<span style="color: #e9b96e;">'__main__'</span>:

<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Easy puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">puzzle1</span> = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [6, 0, 0, 1, 9, 5, 0, 0, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [0, 9, 8, 0, 0, 0, 0, 6, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [8, 0, 0, 0, 6, 0, 0, 0, 3],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [4, 0, 0, 8, 0, 3, 0, 0, 1],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [7, 0, 0, 0, 2, 0, 0, 0, 6],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [0, 6, 0, 0, 0, 0, 2, 8, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [2, 0, 0, 4, 1, 9, 0, 0, 5],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span> [3, 0, 0, 0, 8, 0, 0, 7, 9]];    
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">Hard puzzle</span>
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">puzzle2</span> = [[7, 0, 1, 4, 0, 6, 3, 0, 2],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [0, 0, 0, 0, 0, 0, 0, 0, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [3, 0, 0, 9, 0, 1, 0, 0, 6],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [4, 0, 6, 0, 0, 0, 2, 0, 7],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [0, 0, 0, 0, 4, 0, 0, 0, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [9, 0, 2, 0, 0, 0, 8, 0, 4],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [1, 0, 0, 3, 0, 7, 0, 0, 8],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [0, 0, 0, 0 ,0, 0, 0, 0, 0],
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>  [6, 0, 4, 5, 0, 2, 1, 0, 9]];


<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">'---------------Solving an easy puzzle ------------'</span>)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"The puzzle:"</span>)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> r <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle1:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(r);
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">sol</span>,<span style="color: #fcaf3e;">con</span> = solve_sodoku(puzzle1)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"The solution is:"</span>)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> r <span style="color: #b4fa70; font-weight: bold;">in</span> sol:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(r)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">'---------------Solving a hard puzzle ------------'</span>)        
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"The puzzle:"</span>)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> r <span style="color: #b4fa70; font-weight: bold;">in</span> puzzle2:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(r);
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #fcaf3e;">sol</span>,<span style="color: #fcaf3e;">con</span> = solve_sodoku(puzzle2)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"The solution is:"</span>)
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">for</span> r <span style="color: #b4fa70; font-weight: bold;">in</span> sol:
<span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #DCDCCC; background-color: #212526;"> </span>   <span style="color: #b4fa70; font-weight: bold;">print</span>(r)    

</pre>
</div>


