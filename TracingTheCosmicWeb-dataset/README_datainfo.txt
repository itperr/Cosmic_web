############################################################
#
#
#
#
# Data used in "Tracing the Cosmic Web" comparisson project
#   by Libeskind et al (c) 2017
#
#   For question please contact: 
#   
#   Noam I Libeskind
#   email: nlibeskind@aip.de   
#   
#
#
#
############################################################

This directory contains two types of files: CUBE and FOF
The simulation that the methods were run on was:

Planck cosmology
Lbox = 200Mpc/h 
Nparticles = 512^3


Terminology/convention for web classification

void 	   = 0
sheet 	   = 1
filament   = 2
knot 	   = 3
n/a 	   = -1



Each method was asked to return two files:

CUBE_200_512_<method>.txt
FOF_200_512_<method>.txt


CUBE file format: The cosmic web classification of each point in space is returned on a regular grid of 2Mpc spacing. The first three columns give the (x,y,z) coordinate of the cell's center. The last column returns the "web_id", described above.

FOF file format: Each method was asked to add a final column to the FOF catalog with the "web_id".

The meaning of the other columns may be ascertained from the header in the FOF catalog: fof_catalog_200Mpc512.txt (b=0.2 likning length)

We include the gadget z=0 snapshot directory as well ./snapshots

