# tracks_to_bbox.py

import numpy
import skimage.draw, skimage.io
import skvideo.io
import cPickle
import sys
import micp

def main(vd_fname,tracks_fname,outvid_fname,model_fname=None):
	print "Loading video '{}'".format(vd_fname)
	vd = skvideo.io.vread(vd_fname)
	print "Loading tracks '{}'".format(tracks_fname)
	result = cPickle.load(open(tracks_fname))
	model_img = None
	if not(model_fname is None):
		print "Loading model '{}'".format(model_fname)
		model_img = skimage.io.imread(model_fname)
	elif 'model_img' in result.keys():
		print "Loading stored model '{}'".format(result['model_fname'] if 'model_fname' in result.keys() else '<unlabeled>')
		model_img = result['model_img']
	else:
		print "No stored model and no model file provided, exiting!"
		return
	bbox_vtx = numpy.array([[0.,0.],[0.,model_img.shape[1]],[model_img.shape[0],model_img.shape[1]],[model_img.shape[0],0.]])
	print "Applying bounding boxes to tracks"
	bbox_vd = micp.bbox_from_tracks(bbox_vtx,vd,tracks)
	print "Writing output to '{}'".format(outvid_fname)
	skvideo.io.vwrite(outvid_fname,bbox_vd,outputdict={'-vcodec':'libx264','-crf':'18'})
	print "done!"

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print sys.argv
		print "Usage: python {} <input video> <tracks pickle> <output video> [model image]".format(sys.argv[0])
	else:
		main(*(sys.argv[1:]))