# micp.py
import numpy, scipy.optimize
import pyqtgraph
import skvideo.io
import skimage.io, skimage.draw, skimage.transform
import pyflann
import time
import sys

DEBUG_VERBOSE=True

def vprint(thing):
	if(DEBUG_VERBOSE):
		print thing

def make_tform_mat(tform):
	return numpy.matrix([[numpy.cos(tform[2]), -numpy.sin(tform[2]), tform[0]],\
		                 [numpy.sin(tform[2]),  numpy.cos(tform[2]), tform[1]],\
		                 [                  0,                    0,        1]])

def tform_pts(pts,tform):
	tform_mat = make_tform_mat(tform)
	aug_pts = numpy.column_stack([pts,numpy.ones(shape=(len(pts),1))])
	return tform_mat.dot(aug_pts.T).T[:,:-1]

def tform_error(tform,src,dst):
	tform_mat = make_tform_mat(tform)
	augmented_src = numpy.column_stack([src,numpy.ones(shape=(len(src),1))])
	tformed_src = tform_mat.dot(augmented_src.T).T[:,:-1]
	return numpy.linalg.norm(tformed_src-dst,axis=1).sum()

def est_tform(detection_pts,model_pts,max_iters=100,initial_tform=None,use_spo_min=False):
	aligned_model_pts = model_pts+detection_pts[0]-model_pts[0]
	if not(initial_tform is None):
		aligned_model_pts = tform_pts(model_pts,initial_tform)
	nn_structure = pyflann.FLANN()
	nn_structure.build_index(detection_pts.astype('float'))
	closest_detection_idx,residuals = nn_structure.nn_index(aligned_model_pts.astype('float'))
	last_idx = None
	itr_ctr = 0
	last_tform = None
	while (itr_ctr<max_iters) and not(numpy.array_equal(closest_detection_idx,last_idx)):
		itr_ctr = itr_ctr+1
		last_idx = closest_detection_idx
		closest_detection_pts = detection_pts[closest_detection_idx]
		if use_spo_min:
			opt_res = scipy.optimize.minimize(fun=tform_error,\
			                                  x0=numpy.array([0,0,0]),\
			                                  args=(model_pts,closest_detection_pts))
			last_tform = opt_res.x
		else:
			et_res = skimage.transform.estimate_transform('euclidean',model_pts,closest_detection_pts)
			last_tform = numpy.append(et_res.translation,et_res.rotation)
		aligned_model_pts = tform_pts(model_pts,last_tform)
		closest_detection_idx,residuals = nn_structure.nn_index(aligned_model_pts.astype('float'))
	return last_tform,nn_structure,aligned_model_pts

def mask_to_pts(mask):
	row_idxs, col_idxs = numpy.mgrid[0:mask.shape[0],0:mask.shape[1]]
	return numpy.array(zip(row_idxs[mask],col_idxs[mask]))

def detect_new_tracks(model_pts,frame_pts,max_detections=50,cover_radius=10,cover_threshold=0.1):
	multi_ctr = 0
	tform_list = list()
	while multi_ctr<max_detections and len(frame_pts) > cover_threshold*len(model_pts):
		multi_ctr = multi_ctr+1
		tform,nn_structure,aligned_model_pts = est_tform(frame_pts,model_pts)
		# remove detections within cover_radius from transformed model pixels
		uncovered = numpy.ones(len(frame_pts))
		for tmp_pt in numpy.array(aligned_model_pts):
			covered_idx,dists = nn_structure.nn_radius(tmp_pt.flatten(),cover_radius)
			uncovered[covered_idx] = 0
		frame_pts = frame_pts[uncovered.astype('bool')]
		# add new detection if the number of covered points is greater than cover_threshold*model_pts
		# num_covered = len(uncovered)-numpy.count_nonzero(uncovered)
		# print num_covered, cover_threshold*len(model_pts), num_covered/float(len(model_pts)),cover_threshold
		if len(uncovered)-numpy.count_nonzero(uncovered) > cover_threshold*len(model_pts):
			tform_list.append(tform)
			vprint("New track #{}".format(multi_ctr+1))
	return tform_list

def track_next_frame(model_pts,frame_pts,prev_det_tforms,cover_radius=10,cover_threshold=0.01):
	tform_list = list()
	for tform_idx in range(len(prev_det_tforms)):
		tform = prev_det_tforms[tform_idx]
		new_tform,nn_structure,aligned_model_pts = est_tform(frame_pts,model_pts,initial_tform=tform)
		uncovered = numpy.ones(len(frame_pts))
		for tmp_pt in numpy.array(aligned_model_pts):
			covered_idx, dists = nn_structure.nn_radius(tmp_pt.flatten(),cover_radius)
			uncovered[covered_idx] = 0
		frame_pts = frame_pts[uncovered.astype('bool')]
		if len(uncovered)-numpy.count_nonzero(uncovered) > cover_threshold*len(model_pts):
			tform_list.append(new_tform)
		else:
			vprint("Lost track #{}".format(tform_idx))
	return tform_list, frame_pts

def gen_bgsub_pixel_detector(vd,bg_threshold=0.5):
	bg_image = numpy.median(vd,axis=0)
	def pd(frame):
		bg_subbed = numpy.abs(frame-bg_image)
		if len(bg_subbed.shape)==3:
			bg_subbed = bg_subbed.sum(axis=-1)
		bg_mask = bg_subbed>(bg_subbed.max()*bg_threshold)
		return mask_to_pts(bg_mask)
	return pd

def track_video(model,vd,foreground_detector):
	model_mask = model[:,:,3]>200
	model_pts = mask_to_pts(model_mask)
	tracks = list()
	last_frame_tracks = None
	frame_ctr = 0
	for frame in vd:
		tbefore = time.time()
		frame_ctr = frame_ctr+1
		vprint("Frame #{}".format(frame_ctr))
		this_frame_tracks = list()
		frame_pts = foreground_detector(frame)
		if not(last_frame_tracks is None):
			this_frame_tracks, frame_pts = track_next_frame(model_pts,frame_pts,last_frame_tracks)
		this_frame_tracks += detect_new_tracks(model_pts,frame_pts)
		tracks.append(this_frame_tracks)
		last_frame_tracks = this_frame_tracks
		tafter = time.time()
		vprint("{} tracks at {} fps".format(len(this_frame_tracks),1.0/(tafter-tbefore)))
	return tracks

def bbox_from_tracks(model,vd,tracks,copy_frame=True):
	if copy_frame:
		vd = vd.copy()
	bbox_pts = numpy.array([[0,0],[0,model.shape[1]],[model.shape[0],model.shape[1]],[model.shape[0],0]])
	for f_idx in range(len(vd)):
		frame = vd[f_idx]
		f_tracks = tracks[f_idx]
		for tform in f_tracks:
			tformed_bbox_pts = tform_pts(bbox_pts,tform).round().astype('int')
			rr,cc = skimage.draw.polygon_perimeter(tformed_bbox_pts[:,0].flatten(),tformed_bbox_pts[:,1].flatten(),shape=frame.shape,clip=True)
			frame[rr,cc,:]=[255,0,0]
	return vd

def main(vd_fname,model_fname,out_fname):
	vprint("Loading model '{}'".format(model_fname))
	model_img = skimage.io.imread(model_fname)
	vprint("Loading video '{}'".format(vd_fname))
	vd = skvideo.io.vread(vd_fname)
	vprint("Creating foreground detector")
	fg_det = gen_bgsub_pixel_detector(vd[:100])
	vprint("Starting tracker")
	tracks = track_video(model_img,vd,fg_det)
	vprint("Applying bounding boxes to tracks")
	bbox_vd = bbox_from_tracks(model_img,vd,tracks)
	vprint("Writing output to '{}'".format(out_fname))
	skvideo.io.vwrite(out_fname,bbox_vd)
	vprint("done!")

if __name__ == '__main__':
	if(len(sys.argv)!=4):
		print sys.argv
		print "Usage: python {} <input video> <model image> <output video>".format(sys.argv[0])
	else:
		main(sys.argv[1],sys.argv[2],sys.argv[3])