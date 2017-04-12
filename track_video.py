track_video.py

import skimage.io
import skvideo.io
import micp
import cPickle
import sys

def main(vd_fname,model_fname,out_fname):
	vprint("Loading model '{}'".format(model_fname))
	model_img = skimage.io.imread(model_fname)
	vprint("Loading video '{}'".format(vd_fname))
	vd = skvideo.io.vread(vd_fname)
	vprint("Creating foreground detector")
	fg_det = micp.gen_bgsub_pixel_detector(vd[:100])
	vprint("Starting tracker")
	tracks = micp.track_video(model_img,vd,fg_det)
	vprint("Saving tracks to '{}'".format(out_fname))
	cPickle.dump(tracks,open(out_fname,'w'))
	vprint("done!")

if __name__ == '__main__':
	if(len(sys.argv)!=4):
		print sys.argv
		print "Usage: python {} <input video> <model image> <output pickle>".format(sys.argv[0])
	else:
		main(sys.argv[1],sys.argv[2],sys.argv[3])