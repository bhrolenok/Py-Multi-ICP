# tracks_to_bbox.py

import cPickle
import sys
import micp
import btfutil

def main(tracks_fname,outbtf_dirname,start_time=None, stop_time=None):
	print "Loading tracks '{}'".format(tracks_fname)
	tracks = None
	CoM_y,CoM_x,CoM_theta = None,None,None
	framerate = 30.0
	result = cPickle.load(open(tracks_fname))
	if type(result) is dict:
		tracks = result['tracks']
		CoM_x = result['CoM_x'] if 'CoM_x' in result.keys() else 0.0
		CoM_y = result['CoM_y'] if 'CoM_y' in result.keys() else 0.0
		CoM_theta = result['CoM_theta'] if 'CoM_theta' in result.keys() else 0.0
		if 'metadata' in result.keys():
			ffmpeg_probe_str_fr = result['metadata']['@avg_frame_rate']
			f_str,s_str = ffmpeg_probe_str_fr.split('/')
			framerate = float(f_str)/float(s_str)
	else:
		print "Warning! old, list-only tracks detected. Using defaults."
		tracks = result
	if not((start_time is None) or (stop_time is None)):
		print "Snippet from {} to {}".format(float(start_time),float(stop_time))
		tracks = micp.snip(tracks,float(start_time),float(stop_time))
	print "Converting to BTF"
	btf = btfutil.BTF()
	micp.btf_from_tracks(tracks,btf_obj=btf,framerate=framerate,CoM_x=CoM_x,CoM_y=CoM_y,CoM_theta=CoM_theta)
	print "Writing output to '{}'".format(outbtf_dirname)
	btf.save_to_dir(outbtf_dirname)
	print "done!"

if __name__ == '__main__':
	if len(sys.argv) <= 3 or len(sys.argv)>5:
		print sys.argv
		print "Usage: python {} <tracks pickle> <output dir> [start time (seconds)] [stop time (seconds)]".format(sys.argv[0])
	else:
		main(*sys.argv[1:])
