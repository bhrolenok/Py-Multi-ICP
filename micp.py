# micp.py
import numpy, scipy.optimize
import pyqtgraph
import skvideo.io
import skimage.io, skimage.draw
import pyflann

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

def est_tform(detection_pts,model_pts,max_iters=100):
	aligned_model_pts = model_pts+detection_pts[0]-model_pts[0]
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
		opt_res = scipy.optimize.minimize(fun=tform_error,\
		                                  x0=numpy.array([0,0,0]),\
		                                  args=(model_pts,closest_detection_pts))
		last_tform = opt_res.x
		aligned_model_pts = tform_pts(model_pts,last_tform)
		closest_detection_idx,residuals = nn_structure.nn_index(aligned_model_pts.astype('float'))
	return last_tform,closest_detection_idx

def mask_to_pts(mask):
	row_idxs, col_idxs = numpy.mgrid[0:mask.shape[0],0:mask.shape[1]]
	return numpy.array(zip(row_idxs[mask],col_idxs[mask]))

def bbox_detect(model,frame,bg_img,copy_frame=True):
	if copy_frame:
		frame = frame.copy()
	model_mask = model[:,:,3]>200
	model_pts = mask_to_pts(model_mask)
	bg_subbed = numpy.abs(frame-bg_img)
	if len(bg_subbed.shape)==3:
		bg_subbed = bg_subbed.sum(axis=-1)
	bg_mask = bg_subbed>(bg_subbed.max()/2.0)
	detection_pts = mask_to_pts(bg_mask)
	tform,covered_pts_idx = est_tform(detection_pts,model_pts)
	bbox_pts = numpy.array([[0,0],[0,model.shape[1]], [model.shape[0],model.shape[1]], [model.shape[0],0]])
	tformed_bbox_pts = tform_pts(bbox_pts,tform).round().astype('int')
	rr,cc = skimage.draw.polygon_perimeter(tformed_bbox_pts[:,0].flatten(),tformed_bbox_pts[:,1].flatten(),shape=frame.shape,clip=True)
	frame[rr,cc,:]=[255,0,0]
	return frame

def bgsub_snippets():
	vd = skvideo.io.vread("Downloads/2011MARS_Aphaenogaster_Fruitfly_PredPrey_mar28/MVI_9582.MOV",num_frames=100)
	img_med = numpy.median(vd,axis=0)
	bg_subbed = numpy.abs(vd-img_med)
	pyqtgraph.image(bg_subbed)
	# skvideo.io.vwrite("Downloads/2011MARS_Aphaenogaster_Fruitfly_PredPrey_mar28/detections.mp4",bg_subbed)
