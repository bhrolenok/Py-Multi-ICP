# micp.py
import numpy, scipy.optimize
import skvideo.io
import skimage.io, skimage.draw, skimage.transform
import pyflann
import time

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

def detect_new_tracks(model_pts,frame_pts,next_track_id,max_detections=50,cover_radius=10,cover_threshold=0.1):
	multi_ctr = 0
	tform_dict = dict()
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
			tform_dict[next_track_id] = tform #tform_dict.append(tform)
			vprint("New track #{}".format(next_track_id))
			next_track_id = next_track_id+1
	return tform_dict

def track_next_frame(model_pts,frame_pts,prev_det_tforms,cover_radius=10,cover_threshold=0.01):
	tform_dict = dict()
	for tform_id in prev_det_tforms:
		tform = prev_det_tforms[tform_id]
		new_tform,nn_structure,aligned_model_pts = est_tform(frame_pts,model_pts,initial_tform=tform)
		uncovered = numpy.ones(len(frame_pts))
		for tmp_pt in numpy.array(aligned_model_pts):
			covered_idx, dists = nn_structure.nn_radius(tmp_pt.flatten(),cover_radius)
			uncovered[covered_idx] = 0
		frame_pts = frame_pts[uncovered.astype('bool')]
		if len(uncovered)-numpy.count_nonzero(uncovered) > cover_threshold*len(model_pts):
			tform_dict[tform_id] = new_tform #tform_dict.append(new_tform)
		else:
			vprint("Lost track #{}".format(tform_id))
	return tform_dict, frame_pts

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
	next_track_id = 0
	for frame in vd:
		tbefore = time.time()
		frame_ctr = frame_ctr+1
		vprint("Frame #{}".format(frame_ctr))
		this_frame_tracks = dict()
		frame_pts = foreground_detector(frame)
		if not(last_frame_tracks is None):
			this_frame_tracks, frame_pts = track_next_frame(model_pts,frame_pts,last_frame_tracks)
			next_track_id = max(next_track_id,max(this_frame_tracks)+1)
		new_tracks = detect_new_tracks(model_pts,frame_pts,next_track_id)
		for key in new_tracks:
			if key in this_frame_tracks:
				vprint("Track ID overlap, probably an error!")
				vprint("new_tracks: {}".format(new_tracks))
				vprint("this_frame_tracks: {}".format(this_frame_tracks))
				assert(not(key in this_frame_tracks))
		this_frame_tracks.update(new_tracks) # this_frame_tracks += detect_new_tracks(model_pts,frame_pts)
		tracks.append(this_frame_tracks)
		last_frame_tracks = this_frame_tracks
		tafter = time.time()
		vprint("{} tracks at {} fps".format(len(this_frame_tracks),1.0/(tafter-tbefore)))
	return tracks

def bbox_from_tracks(bbox_vtx,vd,tracks,copy_frame=True):
	if copy_frame:
		vd = vd.copy()
	for f_idx in range(len(vd)):
		frame = vd[f_idx]
		f_tracks = tracks[f_idx]
		for tform_id in f_tracks:
			tform = f_tracks[tform_id]
			tformed_bbox_vtx = tform_pts(bbox_vtx,tform).round().astype('int')
			rr,cc = skimage.draw.polygon_perimeter(tformed_bbox_vtx[:,0].flatten(),tformed_bbox_vtx[:,1].flatten(),shape=frame.shape,clip=True)
			frame[rr,cc,:]=[255,0,0]
	return vd

def btf_from_tracks(tracks,btf_obj=None,framerate=30.0):
	col_data = dict()
	col_data['id'] = list()
	col_data['ximage'] = list()
	col_data['yimage'] = list()
	col_data['timage'] = list()
	col_data['timestamp'] = list()
	for f_idx in range(len(tracks)):
		timestamp = int(f_idx)
		for t_id in tracks[f_idx].keys():
			col_data['id'].append('{}'.format(t_id))
			col_data['ximage'].append('{}'.format(tracks[f_idx][t_id][0]))
			col_data['yimage'].append('{}'.format(tracks[f_idx][t_id][1]))
			col_data['timage'].append('{}'.format(tracks[f_idx][t_id][2]))
			col_data['timestamp'].append('{}'.format(timestamp))
	if not(btf_obj is None):
		btf_obj.column_data = col_data
	return col_data

def snip(tracks,time1,time2,framerate=30.0):
	return tracks[int(time1*framerate):int(time2*framerate)]
