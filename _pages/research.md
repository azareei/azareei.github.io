---
layout: research
title: research
seo_title: "Research | Ahmad Zareei: Physical AI, World Models & 3D Vision"
permalink: /projects/
description: 
meta_description: "Ahmad Zareei, Senior Research Scientist at Meta, works on physical AI, world models, 3D computer vision, graph neural network simulators, and LLM agents."
nav: true
---


I came to AI through physics. I am a Senior Research Scientist at Meta, and my research centers on physical AI: building AI systems that perceive, model, and reason about the physical world. My work spans 3D computer vision, graph neural networks, world models and learned simulators, and LLM-powered agents, and I take ideas from research all the way to production: from real-time 3D face reconstruction from a single camera, to the virtual try-on system for Ray-Ban Meta smart glasses, to agentic AI pipelines that design research studies, simulate users with LLM synthetic personas, and synthesize findings.

My background is in applied mathematics and mechanics. During my Ph.D. at UC Berkeley, I developed transformation-based methods to control waves, including cloaks for water and flexural waves. As a postdoctoral fellow at Harvard SEAS, I built deep learning surrogates for the inverse design of mechanical metamaterials and graph neural network simulators that learn to solve PDEs. That path shapes how I approach AI today: I look for the structure underneath a problem, such as the governing equations, the dynamics, and the symmetries, and build it into models so they generalize beyond the data they were trained on. This physics-informed machine learning perspective drives my interest in world models, embodied AI and robotics, and AI for science. My research has appeared in Physical Review Letters, PNAS, Advanced Materials, and Scientific Reports.

*Research interests:* physical AI · world models · learned physical simulators · 3D computer vision · graph neural networks · LLM agents · physics-informed ML · AI for science

<br>

<h1> mentorship: </h1>
<ul>
<li><h3>PhD:</h3>
  <ul>
      <li> Rini Gladstone (Meta Reality Labs), Project: Graph Neural Networks (GNN) for physics simulation <br>
          Output: Coauthor in a publication (in process) </li>  
      <li> Eder Medina (Harvard University), Project: Large deformation in soft lens <br>
          Output: Coauthor in a PRL publication </li>
      <li> Bolei Deng (Harvard University), Project:  Transition waves in multistable links <br>
          Output: Coauthor in a PNAS publication </li>
      <li> Pan Deng (Harvard University), Project: Random resistor networks and porous media <br>
          Output: Coauthor in a PRL publication  </li>                  
  </ul>
</li>

<li> <h3>Master:</h3>
<ul>
    <li> Pourya Pilva (Aachen, Germany), Project: Graph neural networks in solid mechanics computation <br>
         Output: Coauthor in a conference publication </li>
    <li> Sven Borden (EPFL, Switzerland), Project:  Bistable units in robotics locomotion <br>
         Output: M.Sc. thesis</li>
</ul>
</li>

<li> 
<h3>Undergrad:</h3>
<ul>
    <li> Peter Grenfel, Project: Swimming in low-reynolds <br>
         Ouput: Coauthor in a PRA publication </li>
</ul>
</li>
</ul>

<br>

<h1> teaching: </h1>
<ul>
  <li> <b> Harvard University</b> </li><br>
  <ul>
    <li><b>Computational Methods (Finite Elements)</b> (ESC 228), w/ Professor <a href="https://www.seas.harvard.edu/about-us/directory?search=%22Katia%20Bertoldi%22">Katia Bertoldi</a>, School of Engineering and Applied Sciences, Harvard University, Spring 2021 </li>
  </ul>
</ul>
<ul>
  <li> <b> University of California Berkeley</b> </li><br>
  <ul>
    <li><b>Programming</b>, w/ Professor <a href="https://www.me.berkeley.edu/people/faculty/panayiotis-papadopoulos"> Panayiotis Papadopoulos</a>, Department of Mechanical Engineering, UC Berkeley, Fall 2018 </li>
    <li><b>Numerical Methods in Partial Differential Equations</b> (Math 228B), w/ Professor <a href="https://math.berkeley.edu/~sethian/">James Sethian</a>, Department of Mathematics, UC Berkeley, Spring 2018 </li>
    <li><b>Multivariable Calculus</b>, w/ Professor <a href="https://math.berkeley.edu/~auroux/">Denis Auroux</a>, Department of Mathematics, UC Berkeley, Fall 2017 </li>
    <li><b>Physics: Heat, Electricity, and Magnetism</b>, w/ Professor <a href="http://physics.berkeley.edu/people/faculty/alex-zettl">Alex Zettl</a>, Department of Physics, UC Berkeley, Spring 2016 </li>
    <li><b>Linear Algebra</b>, w/ Professor <a href="https://math.berkeley.edu/~serganov/">Vera Serganova</a>, Department of Mathematics, UC Berkeley, Fall 2015 </li>
  </ul>
  <br>
  <li> <b> Sharif University of Technology</b> </li><br>
  <ul>
    <li><b> Analytical Mechanics I </b>, w/ Professor <a href="https://scholar.google.com/citations?user=Jkmd00gAAAAJ&hl=en">Akhavan</a>, Department of Physics, SUT, Fall 2012 </li>
    <li><b> Mechanics of Materials III (Advanced) </b>, w/ Professor <a href="https://scholar.google.com/citations?user=hJ81dGgAAAAJ&hl=en">Noseir</a>, Department of Mechanical Engineering, SUT, Fall 2012 </li>
    <li><b> Electromagnetism II </b>, w/ Professor <a href="http://sharif.edu/~bahmanabadi/">Bahmanabadi</a>, Department of Physics, SUT, Spring 2012 </li>
    <li><b> Electromagnetism I </b>, w/ Professor <a href="http://sharif.edu/~bahmanabadi/">Bahmanabadi</a>, Department of Physics, Fall 2011 </li>
    <li><b> Numerical Computations </b>, w/ Dr. <a href="https://scholar.google.com/citations?user=05OU5R4AAAAJ">Sadeghian</a>, Department of Mechanical Engineering, SUT, Spring 2011 </li>
    <li> Instructor, <b> Preparation for National Physics Olympiad </b>, <a href="https://en.wikipedia.org/wiki/Alborz_High_School"> Alborz High School</a>, 2012, Tehran, Iran </li>
    <!-- <li> Teaching <b> Intro. to Physics and Laboratory </b>, Shahid Soltani High School (NODET), summer 2011, Karaj, Iran </li> -->
</ul>
</ul>
<br>


## Projects:
<div class="projects grid">
  {% assign sorted_projects = site.projects | sort: "importance" %}
  {% for project in sorted_projects %}
  <div class="grid-item">
    {% if project.redirect %}
    <a href="{{ project.redirect }}" target="_blank">
    {% else %}
    <a href="{{ project.url | relative_url }}">
    {% endif %}
      <div class="card hoverable">
        {% if project.img %}
        <img src="{{ project.img | relative_url }}" alt="project thumbnail">
        {% endif %}
        <div class="card-body">
          <h2 class="card-title text-lowercase">{{ project.title }}</h2>
          <p class="card-text">{{ project.description }}</p>
          <div class="row ml-1 mr-1 p-0">
            {% if project.github %}
            <div class="github-icon">
              <div class="icon" data-toggle="tooltip" title="Code Repository">
                <a href="{{ project.github }}" target="_blank"><i class="fab fa-github gh-icon"></i></a>
              </div>
              {% if project.github_stars %}
              <span class="stars" data-toggle="tooltip" title="GitHub Stars">
                <i class="fas fa-star"></i>
                <span id="{{ project.github_stars }}-stars"></span>
              </span>
              {% endif %}
            </div>
            {% endif %}
          </div>
        </div>
      </div>
    </a>
  </div>
{% endfor %}

</div>




## Older Projects:

<div class="projects grid">
  {% assign sorted_projects = site.projects_old | sort: "importance" %}
  {% for project in sorted_projects %}
  <div class="grid-item">
    {% if project.redirect %}
    <a href="{{ project.redirect }}" target="_blank">
    {% else %}
    <a href="{{ project.url | relative_url }}">
    {% endif %}
      <div class="card hoverable">
        {% if project.img %}
        <img src="{{ project.img | relative_url }}" alt="project thumbnail">
        {% endif %}
        <div class="card-body">
          <h2 class="card-title text-lowercase">{{ project.title }}</h2>
          <p class="card-text">{{ project.description }}</p>
          <div class="row ml-1 mr-1 p-0">
            {% if project.github %}
            <div class="github-icon">
              <div class="icon" data-toggle="tooltip" title="Code Repository">
                <a href="{{ project.github }}" target="_blank"><i class="fab fa-github gh-icon"></i></a>
              </div>
              {% if project.github_stars %}
              <span class="stars" data-toggle="tooltip" title="GitHub Stars">
                <i class="fas fa-star"></i>
                <span id="{{ project.github_stars }}-stars"></span>
              </span>
              {% endif %}
            </div>
            {% endif %}
          </div>
        </div>
      </div>
    </a>
  </div>
{% endfor %}

</div>
