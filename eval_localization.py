# --*- coding: utf-8 -*-
"""
Created on Wed N////ov 02 21:48:58 2016

@author: yxp
"""
import numpy as np
import os
import xml.dom.minidom as minidom

classes = ( 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

folder_annotations="/home/yarley/yanxp/dataset/VOCdevkit2007/VOC2007/Annotations/"
test_file="/home/yarley/yanxp/dataset/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt"
fs=open(test_file,'r')
fp=[0]*20
tp=[0]*20
acc=[0]*20
for filename in fs.readlines():
    xmlname=folder_annotations+filename.strip('\n')+'.xml'
    if os.path.isfile(xmlname):
	def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(xmlname) as f:
            data = minidom.parseString(f.read())
	objs = data.getElementsByTagName('object')
        num_objs = len(objs)
	for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin')) 
            y1 = float(get_data_from_tag(obj, 'ymin')) 
            x2 = float(get_data_from_tag(obj, 'xmax')) 
            y2 = float(get_data_from_tag(obj, 'ymax')) 
            diff = float(get_data_from_tag(obj, 'difficult'))
	    if diff == 1.0:
		continue 
            cls = str(get_data_from_tag(obj, "name")).lower().strip()
	    for ind in xrange(20):
            	if cls == classes[ind]:
		    prefix=filename.strip('\n')
		    res=open('results_all.txt','r')
		    for line in res.readlines():
		    	line=line.strip('\n').split(" ")
		    	if line[0] == prefix :
			    x3=float(line[2])
			    y3=float(line[3])
			    x4=float(line[4])
			    y4=float(line[5])
	                    xmin=max(x1,x3)
			    ymin=max(y1,y3)
			    xmax=min(x2,x4)
			    ymax=min(y2,y4)
			    iw=xmax-xmin+1
			    ih=ymax-ymin+1
			    if ih>0 and iw >0:
			        ua=((x2-x1+1)*(y2-y1+1)+(x4-x3+1)*(y4-y3+1))-iw*ih
			        ov=iw*ih/ua
			        if ov>0.5:
			            tp[ind] +=1
			        elif ov<0.15:
				    fp[ind] +=1
		
for i in xrange(20):
    acc[i]=round(float(tp[i])/(tp[i]+fp[i]),3)*100
    print(acc[i])
print (round(sum(acc)/20,1))
