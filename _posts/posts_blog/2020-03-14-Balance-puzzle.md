---
layout: posts-article
title: Brain Teaser - Weighing puzzle 
category: blog
---



<br>




<p>
You have a balance and 12 marbles: 11 identical, one secretly
lighter/heavier. Balance lets you see whether LHS is &gt;, =, or &lt;
RHS. Determine which marble is weird and whether it is lighter or
heavier in 3 weighings.
</p>


<p>
We first see if such solution is feasible. There are 3 outcomes per each balance test (LHS&gt;RHS, LHS=RHS, or LHS&lt;RHS). So if we run 3 weighings, then we will have 3x3x3 different distinct outcomes. Considering the fact that each marble can be lighter or heavier or all equal, then we have 2x12+1=25 different cases. Since 25&lt;27, then we can definitely design the weighings so that each outcome corresponds to one case. An example of such case is shown below: 
</p>


<div class="figure">
<p><img src="{{site.baseurl}}/img/posts/marbles.png" alt="marbles.png" width="800px" />
</p>
</div>


<p>
To generalize this question, if we have \( n \) marbles with one being lighter/heavier or maybe equal, we will have \( 2n+1\) cases, for which we need at least \(m \) where \( 3^m \geq 2n+2\). 
</p>

