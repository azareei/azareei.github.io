---
layout: page
title: research
permalink: /projects/
description: 
nav: true
---


<br>
<div class="row mt-3" style="text-align:center;">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" width="600" src="{{site.baseurl}}/assets/research/summary.png">
    </div>
</div>
<br>

Historically, scientists and engineers have avoided nonlinearities,
instabilities, and disorder due to their complexities; however,
recently, it has been shown that by harnessing geometric/material
nonlinearities or exploiting instabilities and disorder in a system,
novel functionalities and exceptional properties can be created. My
vision is to address the challenge of dealing with nonlinear
behaviors, instability, and disorder in the context of mechanics and
particularly waves; and to unlock the huge potentials gained through
using these features. For this purpose, I'm focusing on three
different areas: (i) Active materials where material properties
actively change in space and time, resulting in peculiar instabil-
ities and nonlinear behaviors; (ii) Passive metamaterials where an
ordered network of unit cells results in unique properties in the
nonlinear regime; and (iii) Disordered medium which occurs naturally
in many physical and bi- ological settings, has rich physics and is
challenging to deal with. 

<br>
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
