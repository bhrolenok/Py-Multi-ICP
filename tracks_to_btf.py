# tracks_to_bbox.py

import cPickle
import sys
import micp
import btfutil

def main(tracks_fname,outbtf_dirname,start_time=None, stop_time=None):
	print "Loading tracks '{}'".format(tracks_fname)
	tracks = cPickle.load(open(tracks_fname))
	if not((start_time is None) or (stop_time is None)):
		print "Snippet from {} to {}".format(float(start_time),float(stop_time))
		tracks = micp.snip(tracks,float(start_time),float(stop_time))
	print "Converting to BTF"
	btf = btfutil.BTF()
	micp.btf_from_tracks(tracks,btf)
	print "Writing output to '{}'".format(outbtf_dirname)
	btf.save_to_dir(outbtf_dirname)
	print "done!"

if __name__ == '__main__':
	if len(sys.argv) <= 3 or len(sys.argv)>5:
		print sys.argv
		print "Usage: python {} <tracks pickle> <output dir> [start time (seconds)] [stop time (seconds)]".format(sys.argv[0])
	else:
		main(*sys.argv[1:])
