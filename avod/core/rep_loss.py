import tensorflow as tf
import numpy as np

def cal_interarea(bbox1,bbox2):
  tx1,ty1,tz1,dx1,dy1,dz1 = tf.split(bbox1,6,axis=2)
  tx2,ty2,tz2,dx2,dy2,dz2 = tf.split(bbox2,6,axis=2)

  xI1 = tf.maximum(tx1-0.5*dx1,tx2-0.5*dx2)
  yI1 = tf.maximum(ty1-0.5*dy1,ty2-0.5*dy2)
  zI1 = tf.maximum(tz1-0.5*dz1,tz2-0.5*dz2)

  xI2 = tf.minimum(tx1+0.5*dx1,tx2+0.5*dx2)
  yI2 = tf.minimum(ty1+0.5*dy1,ty2+0.5*dy2)
  zI2 = tf.minimum(tz1+0.5*dz1,tz2+0.5*dz2)

  #if (xI2>xI1) and (yI2>yI1) and (zI2>zI1):
  #    inter_area = (xI2-xI1+1)*(yI2-yI1+1)*(zI2-zI1+1)
  #else:
  #    inter_area = -1
  inter_area = (xI2-xI1+1)*(yI2-yI1+1)*(zI2-zI1+1)
  return  tf.maximum(inter_area,0.0)

def cal_iou(bbox1,bbox2):
  tx1,ty1,tz1,dx1,dy1,dz1 = tf.split(bbox1,6,axis=2)
  tx2,ty2,tz2,dx2,dy2,dz2 = tf.split(bbox2,6,axis=2)

  inter = cal_interarea(bbox1,bbox2)
  union = dx1*dy1*dz1+dx2*dy2*dz2-inter
  return tf.maximum(inter/union,0.0)

def cal_iog(gtbox,prebox):
  gtx,gty,gtz,gdx,gdy,gdz = tf.split(gtbox,6,axis=2)

  inter = cal_interarea(gtbox,prebox)
  gt_area = gdx*gdy*gdz
  return inter/gt_area

def smooth_l1(pre,tar):
  diff = pre - tar
  abs_diff = tf.abs(diff)
  abs_diff_lt_1 = tf.less(abs_diff,1.0)
  return tf.reduce_sum(tf.where(abs_diff_lt_1,
    0.5*tf.pow(abs_diff,2),
    abs_diff-0.5),axis=1)

def smooth_ln(x,smooth):
  return tf.where(
    tf.less_equal(x,smooth),
    -tf.log(1-x),
    ((x-smooth)/(1-smooth))-tf.log(1-smooth))
  

def attraction_term(pre,tar,iou):
  iou_max_indices = tf.argmax(iou,axis=1)   #shape[len_of_pre,1]

  gt_boxes_with_max_ious = tf.gather_nd(tar,iou_max_indices)  #shape[len_of_gt,6]

  l1_distances = smooth_l1(pre,gt_boxes_with_max_ious)

  return tf.reduce_sum(tf.cast(l1_distances,tf.float32))/tf.cast(tf.shape(pre)[0],dtype=tf.float32)



def rep_term_gt(pre,tar,iou,smooth):
  iou = tf.reduce_sum(iou,axis=2)  #shape[len_of_pre,len_of_tar,1]  ---> shape[len_of_pre,len_of_tar]
  _,indices_2highest_iou = tf.nn.top_k(iou,k=2)  #shape[len_of_pre,2]

  indice_0 = indices_2highest_iou[...,1]
  indice_1 = indices_2highest_iou[...,1]
  indice = tf.stack([indice_0,indice_1],axis=1)#用indices_2highest_iou的第二列替换第一列

  iou_indices,_= tf.nn.top_k(indice,k=1) #shape[len_of_pre,1] 得到第二大iou对应得到tar索引

  gt_boxes_with_max_ious = tf.gather_nd(tar,iou_indices) #shape[len_of_tar,6] 得到第二大iou对应tar的对应编码

  #为方便计算iog，增加pre和tar的维度
  gt_boxes_with_max_ious = tf.expand_dims(gt_boxes_with_max_ious,axis=1)
  pre = tf.expand_dims(pre,axis=1)

  ln_distances = smooth_ln(cal_iog(gt_boxes_with_max_ious,pre),smooth)
  return tf.reduce_sum(ln_distances)/tf.cast(tf.shape(pre)[0],dtype=tf.float32)


def rep_term_box(iou,smooth):
  iou_over_pre_indices = tf.where(tf.less(iou,1.0))
  iou = tf.gather_nd(iou,iou_over_pre_indices)

  dist_sum = tf.reduce_sum(smooth_ln(iou,smooth))
  iou_sum = tf.reduce_sum(iou)

  return dist_sum/tf.maximum(iou_sum,0.000001)