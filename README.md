# sleap_tools
A repository to regroup useful code to manipulate sleap objects and h5 exports, as well as help and method to setup tracking routines.

# Working with H5 files

sleap tracking can be exported as h5 format, which is standard for data storage but not super intuitive. In sleap_utils.py, a SleapTracks class is available. It is a wrapper for .h5 sleap files that helps manipulate tracking data.

## SleapTracks structure

SleapTracks objects have attributes based on the data available in .h5 files. Among these:

- Information on tracked nodes, n° of animals, frames
- direct access to xy values of each node for each animal
