# track_video.py

import skimage.io
import skvideo.io
import micp
import cPickle
import sys

def main(vd_fname,model_fname,out_fname,model_offsets=None):
	micp.vprint("Loading model '{}'".format(model_fname))
	model_img = skimage.io.imread(model_fname)
	micp.vprint("Loading video '{}'".format(vd_fname))
	vd = skvideo.io.vread(vd_fname)
	metadata = skvideo.io.ffprobe(vd_fname)
	micp.vprint("Creating foreground detector")
	fg_det = micp.gen_bgsub_pixel_detector(vd[:100])
	micp.vprint("Starting tracker")
	tracks = micp.track_video(model_img,vd,fg_det)
	result = dict()
	result['tracks'] = tracks
	result['vd_fname'] = vd_fname
	result['model_fname'] = model_fname
	result['model_img'] = model_img
	# Note: can't pickle function objects (boo)
	# result['fg_det'] = fg_det 
	if model_offsets is None:
		result['CoM_x'] = model_img.shape[1]/2
		result['CoM_y'] = model_img.shape[0]/2
		result['CoM_theta'] = 0.0
	else:
		result.update(model_offsets)
	result['metadata'] = metadata['video']
	micp.vprint("Saving tracks to '{}'".format(out_fname))
	cPickle.dump(result,open(out_fname,'w'))
	micp.vprint("done!")

if __name__ == '__main__':
	if(len(sys.argv)!=4):
		print sys.argv
		print "Usage: python {} <input video> <model image> <output pickle>".format(sys.argv[0])
	else:
		main(sys.argv[1],sys.argv[2],sys.argv[3])