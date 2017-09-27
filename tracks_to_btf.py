# tracks_to_bbox.py

import cPickle
import sys
import micp
import btfutil

def main(tracks_fname,outbtf_dirname):
	print "Loading tracks '{}'".format(tracks_fname)
	tracks = cPickle.load(open(tracks_fname))
	print "Converting to BTF"
	btf = btfutil.BTF()
	micp.btf_from_tracks(tracks,btf)
	print "Writing output to '{}'".format(outbtf_dirname)
	btf.save_to_dir(outbtf_dirname)
	print "done!"

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print sys.argv
		print "Usage: python {} <tracks pickle> <output dir>".format(sys.argv[0])
	else:
		main(sys.argv[1],sys.argv[2])
