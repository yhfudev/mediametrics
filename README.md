Media Metrics
============

# Introduction

Media Metrics is a tool to compare the target video with the original source vidoe frame by frame and generate various the media quality metric values, such as MSE, PSNR, and SSIM etc.

The tool support various file formats, such as mp4, avi etc, or even a sequence of seperated image files (*.png, *.jpg etc).

# Installation

To compile the software,

```
./configure
make
```

# Usage

To show the help message, use the program argument option --help:

```
mediametrics --help

Media Matrics v0.2.0 yhfudev@gmail.com

  generate the media metrics for meida file(s),
  the supported metrics include MSE/PSNR/MS-SSIM etc.

Usage
    mediametrics [options] <source> <compared>

Options:
	-g	Disable GPU if available. default enabled
	-m	show images
	-o <output>	output file name
	-r <res>	resolutions
	-b <#>	the output frame # of the first frame
	-s <#>	the frame # of the original input video
	-d <#>	the frame # of the compared input video
	-h	show this message

<source>    the png files, use format string
<compared>  the video file name

the source format examples:
 -> media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_%05d.png
 -> media.xiph.org/tearsofsteel/tearsofsteel-1080-png/graded_edit_final_%05d.png
 -> 1080/sintel_trailer_2k_%04d.png
 -> media.xiph.org/sintel/sintel-4k-png/%08d.png
 -> ED-1080-png/%05d.png

the resolutions examples:
 -> 1920x1080
 -> 1920*1080,320x180
```




