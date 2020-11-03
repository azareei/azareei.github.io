---
layout: posts-article
title: How to install Lapack & Blas on linux? 
category: linux
---

<br>
<h2> Goal </h2>
LAPack and BLAS are powerfull linear algebra libraries written in fortran. Here I go over installing and using them in linux. First we need to install BLAS, and then we can install LAPack. 

<h2> How to do it? </h2>

<ul>
<li> Download the latest version of <a href="http://www.netlib.org/blas/"> BLAS </a></li>
<li> Open a terminal and go to the directory where you have saved it. </li>
<ul> 
<li> $ tar -xvf blas-3.6.0.tgz # create BLAS subdirectory</li> 
<li> $ cd BLAS-3.6.0 </li>
<li> $ gfortran -O3 -c *.f # compiling </li>
<li> $ ar cr libblas.a *.o  # creates libblas.a </li>
<li> $ sudo cp ./libblas.a /usr/local/lib/ </li>
</ul>
<li> So far we have installed BLAS. Now download the latest version of <a href="http://www.netlib.org/lapack/">LAPack</a>.</li>
<li> Open a terminal and go to the directory where you have saved it. </li>
<ul> 
<li> $ tar -xvf ./lapack-3.6.0.tgz # create LAPack subdirectory</li> 
<li> $ cd lapack-3.6.0 </li>
</ul>
<li> Now you need to change the directory of BLAS in the file "make.inc.example". Open this file and find the line that reads</li>
<ul>
<li> BLASLIB      = ../../librefblas.a </li>
</ul>
<li> and change it to :</li>
<ul>
<li> BLASLIB      = /usr/local/libblas.a </li>
</ul>
<li> Save this file as "make.inc" and then run make:  
<ul>
<li> $ make </li>
</ul>
<li> $ sudo cp ./liblapack.a /usr/local/lib/ </li>
<li> compile your code using "-lblas -llapack" flags. </li>
<li>Voila!! </li>



-------------------------------------------
<li> If you get the error "recipe for target 'znep.out' failed" during installation of LAPack, run the command "$ ulimit -s unlimited". Thanks to <a href="https://www.imprs-astro.mpg.de/content/prof-dr-werner-becker"> Prof. Becker</a> for bringing this up to my attention. </li>
