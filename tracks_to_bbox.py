# tracks_to_bbox.py

import numpy
import skimage.draw, skimage.io
import skvideo.io
import cPickle
import sys
import micp

def main(vd_fname,model_fname,tracks_fname,outvid_fname):
	print "Loading model '{}'".format(model_fname)
	model_img = skimage.io.imread(model_fname)
	bbox_vtx = numpy.array([[0.,0.],[0.,model_img.shape[1]],[model_img.shape[0],model_img.shape[1]],[model_img.shape[0],0.]])
	print "Loading video '{}'".format(vd_fname)
	vd = skvideo.io.vread(vd_fname)
	print "Loading tracks '{}'".format(tracks_fname)
	tracks = cPickle.load(open(tracks_fname))
	print "Applying bounding boxes to tracks"
	bbox_vd = micp.bbox_from_tracks(bbox_vtx,vd,tracks)
	print "Writing output to '{}'".format(outvid_fname)
	skvideo.io.vwrite(outvid_fname,bbox_vd,outputdict={'-vcodec':'libx264','-crf':'18'})
	print "done!"

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print sys.argv
		print "Usage: python {} <input video> <model image> <tracks pickle> <output video>".format(sys.argv[0])
	else:
		main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])