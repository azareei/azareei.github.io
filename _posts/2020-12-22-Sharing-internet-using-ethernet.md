---
layout: post
title:  Sharing internet using Ethernet connection
date: 2020-12-22 22:30:00-0400
description: Sharing internet using Ethernet
---

Say you have a Jetson nano, and you want to connect it to the internet using an Ethernet connection with your PC/Laptop. This is straightforward with just a few clicks. I have Ubuntu as my OS and this is what I did

-  Open network manager (you can run your network manager by typing `nm-connection-editor` in your terminal)
-  Click on creating a new connection
-  Select Ethernet, name is as you wish, e.g., Jetson
- Go to "IPv4 Settings", and select "Shared to the other computers" method
- Now plug in the Ethernet cable to your pc and Jetson. Select jetson connection on your PC and your PC will act as a router for your Jetson.
