# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick	
# --------------------------------------------------------
"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
from fast_rcnn.test import im_detect 
import cv2  
import copy  
import utils.cython_bbox  
from utils.cython_bbox import bbox_overlaps 
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
import google.protobuf as pb2
# from fast_rcnn.test import vis_detections
import google.protobuf.text_format
from utils.cython_nms import nms
import random
try:
   import cPickle as pickle
except:
   import pickle
import shutil
CLASSES = ( '__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb,roidb_w, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)            
            
           
        self.start_iters= 15000
        self.step_iter= 5000
        self.object_num= 1
        self._n_classes= 21

        check_roidb(roidb, True)
        check_roidb(roidb_w, True)
        self._General_roidb= roidb
        self._Weakly_roidb= roidb_w
        self._Present_roidb=[]
 
        curr_roidb= self.get_curr_roidb()
        self.solver.net.layers[0].set_roidb(curr_roidb) 
    def vis_detections(self,image_name, im, class_name, dets,score, thresh=0.5):
        """Draw detected bounding boxes."""
        # inds = np.where(dets[:, -1] >= thresh)[0]
        # if len(inds) == 0:
        #     return

        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in xrange(1):
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        # save the image
        fig = plt.gcf()
        fig.savefig("images/output_"+image_name)
    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        net = self.solver.net
             
        if cfg.TRAIN.BBOX_REG:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()
            
            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis]) 
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
 
    def get_curr_roidb(self):
        if self.solver.iter >=self.start_iters: 
            self.start_iters += self.step_iter
            self.object_num += 2
            sorted_boxes, sorted_img_inds, sorted_img_labels,sorted_scores= self.Roidb_detect_and_sort()             
            curr_roidb= self.Roidb_selection(sorted_boxes, sorted_img_inds, sorted_img_labels,sorted_scores) 			  
            curr_roidb= self.update_roidb(curr_roidb)
        else:
            curr_roidb= self._General_roidb
        print 'Computing bounding-box regression targets...'
       # print "curr_roidb:",curr_roidb
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(curr_roidb,self._n_classes)
        print 'done'
        
        # replace zero elements with EPSILON
        self.bbox_stds[np.where(self.bbox_stds == 0)[0]] = cfg.EPS    		

        return curr_roidb
    # def Roidb_vote(self):


            
    def Roidb_detect_and_sort(self):
            
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
            '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        caffemodel = os.path.join(self.output_dir, filename)
        prototxt = os.path.join('models/CaffeNet', 'test.prototxt')
            
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, str(caffemodel), caffe.TEST)
        
        self._Present_roidb=[]
        roidb= self._General_roidb + self._Weakly_roidb
        

        for j in xrange(len(roidb)):
            # if roidb[j]['islabel'] <= self.object_num  :
            self._Present_roidb.append(roidb[j])


        roidb = self._Present_roidb 
        n_imgs = len(roidb)             

         # initializing local data structures:
        detected_dets = np.array([0,0,0,0])
        detected_scores = np.array([-1]) 
        detected_img_inds = np.array([-1])        
        detected_labels = np.array([0])
        shutil.rmtree("cache/")
        os.mkdir("cache/")
        
        for i in xrange(n_imgs):   
            random.seed()  
            im_name = roidb[i]['image']
            image_name=im_name.split("/")[-1]
            im = cv2.imread(im_name)

            if roidb[i]['flipped']:
                im = im[:, ::-1, :]

            mask_im = np.random.random(size = im.shape)
            mask_im[mask_im>=0.7] = 0
            mask_im[mask_im<0.7] = 1

            curr_shape=im.shape

            bb_boxes = roidb[i]['boxes'] 
            scores, pred_boxes = im_detect(net, im, bb_boxes)	
 
            curr_img_labels= roidb[i]['img_labels'] 
            
            scores[:,0]= 0 # background scores deleted 			    
#----------------------------------begin yanxp 10.15-----------------------
            # labels_length=len(curr_img_labels)
            
            # bb_list=[]
            # cls_list=[]
            # score_list=[]
            # per_class_proposals_num = 3
            
            # for k in xrange(labels_length):
            #     for num in xrange(per_class_proposals_num):
            #         bb_i=scores[:,curr_img_labels[k]].argmax()
            #         cls_j=curr_img_labels[k]
            #         bb_list.append(bb_i)
            #         cls_list.append(cls_j)
            #         score_list.append(scores[bb_i,cls_j])
            #         scores[bb_i,cls_j]=0

            # curre_select_num = 0
            # total_select_num = 2

            # total_score_per_class_proposals=np.zeros([labels_length,per_class_proposals_num])
            # for k in xrange(n_imgs):
            #     select_img_name = roidb[k]['image']
            #     seletc_im = cv2.imread(select_img_name)
            #     seletc_im_shape = seletc_im.shape
            #     seletc_img_labels= roidb[k]['img_labels']
            #     if curr_shape[0]>seletc_im_shape[0] or curr_shape[1]>seletc_im_shape[1] or k == i or curr_img_labels in seletc_img_labels:
            #         continue
            #     curre_select_num += 1
            #     if curre_select_num > total_select_num :
            #         break

            #     per_class_proposals_score=np.zeros([labels_length,per_class_proposals_num])
            #     for cls_id in xrange(labels_length):
            #         detected_per_class_proposal_scores = np.array([-1]) 
            #         detected_per_class_proposal_id = np.array([-1])  
            #         for j in xrange(per_class_proposals_num):
            #             index_num = cls_id*per_class_proposals_num+j
            #             latent_box = pred_boxes[bb_list[index_num], 4 * cls_list[index_num] : 4 * (cls_list[index_num] + 1)]
            #             latent_box_score = score_list[index_num]  
            
            #             proposal_height = int(latent_box[3]-latent_box[1])
            #             proposal_width = int(latent_box[2]-latent_box[0])
            #             x_min = int(latent_box[0])
            #             y_min = int(latent_box[1])
            #             score_orignal = latent_box_score
            #             combination_im=seletc_im.copy()
            #             start_y=random.randint(0,seletc_im.shape[0]-proposal_height)
            #             start_x=random.randint(0,seletc_im.shape[1]-proposal_width)
            #             combination_im[start_y:start_y+proposal_height, start_x:start_x+proposal_width, :] = im[y_min:y_min+proposal_height, x_min:x_min+proposal_width, :]

            #             if roidb[i]['flipped']:
            #                 seletc_im = seletc_im[:, ::-1, :]
            #                 combination_im = combination_im[:, ::-1, :]

            #             sample_boxes = roidb[k]['boxes'] 
            #             combination_boxes=sample_boxes[0:per_class_proposals_num]
            #             for d in xrange(len(combination_boxes)):
            #                 combination_boxes[d][0]=0
            #                 combination_boxes[d][1]=0
            #                 combination_boxes[d][2]=seletc_im.shape[1]
            #                 combination_boxes[d][3]=seletc_im.shape[0]


            #             combination_scores, combination_pred_boxes = im_detect(net, combination_im, combination_boxes)   

            #             bb_i_combination = combination_scores[:,cls_list[index_num]].argmax()
            #             cls_j_combination = cls_list[index_num]
            #             score_combination = combination_scores[bb_i_combination,cls_j_combination]
            #             # # cv2.imshow('image_orig',seletc_im)
            #             # combination_box = combination_pred_boxes[bb_i_combination, 4 * cls_j_combination : 4 * (cls_j_combination + 1)]
            #             # print "score of specific class in sample image before:",score_sample
            #             # print "score of specific class in sample image after :",score_combination

            #             # cv2.rectangle(combination_im,(int(combination_box[0]),int(combination_box[1])),
            #             #                  (int(combination_box[2]),int(combination_box[3])),(255,0,0),5)
            #             # cv2.imshow('image_comb',combination_im)
            #             # cv2.waitKey(0)
            #             # cv2.destroyAllWindows()
            #             detected_per_class_proposal_scores = np.hstack((detected_per_class_proposal_scores, score_combination))
            #             detected_per_class_proposal_id = np.hstack((detected_per_class_proposal_id, j))

            #         score_sortedIndex = np.argsort(detected_per_class_proposal_scores,axis=0)[::-1] 
            #         detected_per_class_proposal_id= detected_per_class_proposal_id[score_sortedIndex][0:per_class_proposals_num].astype(int) 
            #         for per_id in xrange(per_class_proposals_num/2):
            #             per_class_proposals_score[cls_id,detected_per_class_proposal_id[per_id]]  = per_class_proposals_score[cls_id,detected_per_class_proposal_id[per_id]]+ int(per_class_proposals_num/2) - per_id +5
            #     total_score_per_class_proposals = per_class_proposals_score + total_score_per_class_proposals

            # if curre_select_num == 0:
            #     for cls_id in xrange(labels_length):
            #         index_num =cls_id*per_class_proposals_num
            #         latent_box = pred_boxes[bb_list[index_num], 4 * cls_list[index_num] : 4 * (cls_list[index_num] + 1)]
            #         latent_box_score = score_list[index_num]
            #         proposal_height = int(latent_box[3]-latent_box[1])
            #         proposal_width = int(latent_box[2]-latent_box[0])
            #         cv2.rectangle(im,(int(latent_box[0]),int(latent_box[1])),
            #                              (int(latent_box[2]),int(latent_box[3])),(255,0,0),5)
            #         cv2.imshow('orig_im',im)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            #         x=np.floor(latent_box[0])
            #         y=np.floor(latent_box[1])
            #         mask_im[y:y+proposal_height,x:x+proposal_width,:] = 1 
            #         detected_dets = np.vstack((detected_dets, latent_box))
            #         detected_scores = np.hstack((detected_scores, latent_box_score))
            #         detected_img_inds = np.hstack((detected_img_inds, i))
            #         detected_labels = np.hstack((detected_labels, cls_list[index_num]))
            # else:
            #     for cls_id in xrange(labels_length):
            #         index_cls = np.where(total_score_per_class_proposals[cls_id,:] == max(total_score_per_class_proposals[cls_id,:]))
            #         index_num = (cls_id*per_class_proposals_num+index_cls[0])[0]
            #         latent_box = pred_boxes[bb_list[index_num], 4 * cls_list[index_num] : 4 * (cls_list[index_num] + 1)]
            #         latent_box_score = score_list[index_num]
            #         proposal_height = int(latent_box[3]-latent_box[1])
            #         proposal_width = int(latent_box[2]-latent_box[0])
            #         x=np.floor(latent_box[0])
            #         y=np.floor(latent_box[1])
            #         cv2.rectangle(im,(int(latent_box[0]),int(latent_box[1])),
            #                              (int(latent_box[2]),int(latent_box[3])),(255,0,0),5)
            #         cv2.imshow('orig_im',im)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            #         mask_im[y:y+proposal_height,x:x+proposal_width,:] = 1 
            #         detected_dets = np.vstack((detected_dets, latent_box))
            #         detected_scores = np.hstack((detected_scores, latent_box_score))
            #         detected_img_inds = np.hstack((detected_img_inds, i))
            #         detected_labels = np.hstack((detected_labels, cls_list[index_num]))
            labels_length=len(curr_img_labels)
            bb_list=[]
            cls_list=[]
            score_list=[]
            for k in xrange(labels_length):
                bb_i=scores[:,curr_img_labels[k]].argmax()
                cls_j=curr_img_labels[k]
                bb_list.append(bb_i)
                cls_list.append(cls_j)
                score_list.append(scores[bb_i,cls_j])
                scores[bb_i,cls_j]=0

            for j in xrange(labels_length):
                if score_list[j]>0:
                    latent_box = pred_boxes[bb_list[j], 4 * cls_list[j] : 4 * (cls_list[j] + 1)]
                    latent_box_score = score_list[j]
                    proposal_height = int(latent_box[3]-latent_box[1])
                    proposal_width = int(latent_box[2]-latent_box[0])
                    x=np.floor(latent_box[0])
                    y=np.floor(latent_box[1])
                    # cv2.rectangle(im,(int(latent_box[0]),int(latent_box[1])),
                    #                      (int(latent_box[2]),int(latent_box[3])),(255,0,0),2)
                    # cv2.imshow('orig_im',im)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    mask_im[y:y+proposal_height,x:x+proposal_width,:] = 1            
                    detected_dets = np.vstack((detected_dets, latent_box))
                    detected_scores = np.hstack((detected_scores, latent_box_score))
                    detected_img_inds = np.hstack((detected_img_inds, i))
                    detected_labels = np.hstack((detected_labels, cls_list[j]))
            noise_im = np.multiply(im , mask_im)
            noise_im = np.uint8( noise_im )
            im_data=open('cache/'+image_name+'.pkl','wb')
            pickle.dump(noise_im,im_data)
            im_data.close()
            # cv2.imshow('noise_im',noise_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
       

#-------------------------------------- end ------------------------------------ 


#--------------------------------------begin 10.23-------------------------------
  #           CONF_THRESH = 0.5
  #           NMS_THRESH = 0.3
  #           for cls_ind, cls in enumerate(CLASSES[1:]):
  #               cls_ind += 1 # because we skipped background
  #               if cls_ind not in curr_img_labels:
    #     continue
    # cls_boxes = pred_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
  #               cls_scores = scores[:, cls_ind]
  #               dets = np.hstack((cls_boxes,
  #               cls_scores[:, np.newaxis])).astype(np.float32)
  #               keep = nms(dets, NMS_THRESH)
  #               dets = dets[keep, :]
  #               inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
  #               if len(inds) == 0 :
  #                   continue
  #               # self.vis_detections(image_name, im, CLASSES[cls_ind], dets,0.5)
  #               for i in inds:
  #                   latent_box = dets[i, :4]
  #                   latent_box_score = dets[i, -1]
  #                   detected_dets = np.vstack((detected_dets, latent_box))
  #                   detected_scores = np.hstack((detected_scores, latent_box_score))
  #                   detected_img_inds = np.hstack((detected_img_inds, i))
  #                   detected_labels = np.hstack((detected_labels, cls_ind))
#--------------------------------------end --------------------------------------
        
        score_sortedIndex = np.argsort(detected_scores,axis=0)[::-1] 
        # all_length=int(len(score_sortedIndex)*0.9)
        # score_sortedIndex=score_sortedIndex[0:all_length]
        sorted_boxes= detected_dets[score_sortedIndex].astype(int) 
        sorted_img_inds= detected_img_inds[score_sortedIndex].astype(int)  
        sorted_img_labels= detected_labels[score_sortedIndex].astype(int)
        sorted_scores = detected_scores[score_sortedIndex].astype(float);  
        n_pruned_items= len(score_sortedIndex)
        return sorted_boxes[0:n_pruned_items-1], sorted_img_inds[0:n_pruned_items-1], sorted_img_labels[0:n_pruned_items-1],sorted_scores[0:n_pruned_items-1]# after sorting, the last element corresponds to the dummy initialization
 
    def Roidb_selection(self, sorted_boxes, sorted_img_inds, sorted_img_labels,sorted_scores):
        
        new_roidb = []
        n_imgs_to_select=  len(sorted_img_inds)      
 
        n= 0 # n. copied images so far
        i= 0 # n. items of sorted_img_inds processed so far
 
        while n < n_imgs_to_select: 
            k= sorted_img_inds[i]
            curr_img_roidb = copy.deepcopy(self._Present_roidb[k])
            if sorted_scores[i]>=curr_img_roidb['scores']:
                curr_img_roidb['scores']=sorted_scores[i]
            else:
                continue
            
            new_roidb.append(curr_img_roidb)

            # im_name = curr_img_roidb['image']
            # image_name=im_name.split("/")[-1]
            # im = cv2.imread(im_name)	
            # if roidb[i]['flipped']:
            #     im = im[:, ::-1, :]
            # vis_detections(im,sorted_img_labels[i],sorted_boxes[i],image_name)    
            # add the pseudo-ground truth:
            curr_img_roidb['boxes'] = np.vstack((curr_img_roidb['boxes'],sorted_boxes[i]))
            curr_img_roidb['max_classes'] = np.hstack((curr_img_roidb['max_classes'],sorted_img_labels[i]))
            curr_img_roidb['gt_classes'] = np.hstack((curr_img_roidb['gt_classes'],sorted_img_labels[i]))
            curr_img_roidb['max_overlaps'] = np.hstack((curr_img_roidb['max_overlaps'],np.array([1])))
            
            i+= 1 
            n+= 1 

        return new_roidb
    
    def update_roidb(self,roidb): 
 
        n_imgs = len(roidb)

        update_roidb=[]
        for i in xrange(n_imgs):
            bb_boxes= roidb[i]['boxes'] # the pseudo-ground truth box is included (in position len(bb_boxes))
            g_ind= len(bb_boxes) - 1
            g_box = roidb[i]['boxes'][[g_ind]]  
            g_class = roidb[i]['gt_classes'][g_ind] 


            max_overlaps = bbox_overlaps(bb_boxes.astype(np.float), g_box.astype(np.float))
            roidb[i]['max_overlaps']= max_overlaps
            nonzero_inds = np.where(max_overlaps > 0)[0]
            zero_inds = np.where(max_overlaps == 0)[0]
            roidb[i]['max_classes'][zero_inds]= 0
            roidb[i]['max_classes'][nonzero_inds]= g_class
            update_roidb.append(roidb[i]) 
        return update_roidb 

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
 
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)


            if self.solver.iter % 5000 ==0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

                curr_roidb= self.get_curr_roidb()
                check_roidb(curr_roidb, False)
                self.solver.net.layers[0].set_roidb(curr_roidb)    



        if last_snapshot_iter != self.solver.iter: # last snapshot before exiting
            self.snapshot()
            


def check_roidb(roidb, general_roidb):
    num_images = len(roidb)
    for im_i in xrange(num_images):
        g_inds= np.where(roidb[im_i]['gt_classes'] > 0)[0] 

        max_classes= roidb[im_i]['max_classes']
        max_overlaps= roidb[im_i]['max_overlaps']
        
        if general_roidb:
            curr_img_labels= roidb[im_i]['img_labels']

        else:
            bb_inds= np.where(roidb[im_i]['gt_classes'] == 0)[0] 
            
        # other sanity checks ---
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)



def get_training_roidb(imdb): 
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb,roidb_w,output_dir,
              pretrained_model=None, max_iters=70000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb,roidb_w, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
