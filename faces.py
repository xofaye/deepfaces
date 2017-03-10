from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import cm
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

from collections import defaultdict

import hashlib


# t = int(time.time())
t = 1454219613
print "t=", t
random.seed(t)


M = loadmat("mnist_all.mat")

import tensorflow as tf


CROPPED_DIR = 'cropped'
UNCROPPED_DIR = 'uncropped'
ALL_ACTORS = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
SKIP = ['baldwin132.jpg', 'chenoweth85.jpg''drescher92.jpg', 'hader101.jpg']
    
def crop(image, boundingBox):
  # Crop an image given the bounding box
  x1 = int(boundingBox[0])
  y1 = int(boundingBox[1])
  x2 = int(boundingBox[2])
  y2 = int(boundingBox[3])
  return image[y1:y2, x1:x2]

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()      

def rgb2gray(rgb):
    """
    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
    
def resize(image):
    # Resize an image to 64x64
    return imresize(image, (64, 64))

def createDataset(actors):
  """
  Separate data into 3 separate sets and convert to grayscale and resize.
  Requires images of actors to exist in the data/cropped folder
  Output is in the format of:
  {
    train_actor1name: [image1_array, ...],
    validation_actor1name: [image1_array, ...],
    test_actor1name: [image1_array, ...],
    train_actor2name: [image1_array, ...],
    validation_actor2name: [image1_array, ...],
    test_actor2name: [image1_array, ...]
  }
  """
  images_by_actors = defaultdict(list)
  # Find all image files ignoring hidden files that start with '.'
  cropped_images = [f for f in os.listdir(CROPPED_DIR) if os.path.isfile(os.path.join(CROPPED_DIR, f)) and not f.startswith('.')]
  cropped_images.sort()
  for filename in cropped_images:
    actor = filename[: [char.isdigit() for char in filename].index(True)]

    if actor not in actors:
      continue
      
    if filename in SKIP:
        continue

    try:
      # Read the image, transform to greyscale, and resize
      image_array = imread(CROPPED_DIR + '/' + filename)
      image_array = rgb2gray(image_array)
      image_array = resize(image_array)

      images_by_actors[actor].append(image_array)

    except:
      pass
  
  output = {}
  for actor in actors:
    # Randomize the input by shuffling the data
    output['train_'+actor] = array(images_by_actors[actor][0:20]).reshape(20, 64*64)
    output['test_'+actor] = array(images_by_actors[actor][90:120]).reshape(30, 64*64)
    output['validation_'+actor] = array(images_by_actors[actor][120:135]).reshape(15, 64*64)

  return output


def downloadData(actor_names=None):
  """
  Download all images from hardcoded urls.
  """
  file_urls = [
    'http://www.cs.toronto.edu/~guerzhoy/411/proj1/facescrub_actors.txt',
    'http://www.cs.toronto.edu/~guerzhoy/411/proj1/facescrub_actresses.txt'
  ]

  all_data = []
  for file_url in file_urls:
    for line in urllib.urlopen(file_url).readlines():
      all_data.append(line)
  act = actor_names or list(set([a.split("\t")[0] for a in all_data]))

  if not os.path.exists(UNCROPPED_DIR):
    os.makedirs(UNCROPPED_DIR)
  if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)

  for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in all_data:
      if (i == 150):
        # Do not download more than 150 pictures per actor
        break
      if a in line:
        filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
        #timeout is used to stop downloading images which take too long to download
        timeout(testfile.retrieve, (line.split()[4], UNCROPPED_DIR +"/"+filename), {}, 10)
        if not os.path.isfile(UNCROPPED_DIR +"/"+filename):
          continue

        # Once the original image is saved, open it up and crop
        try :
          original_image = imread(UNCROPPED_DIR +"/"+filename)
          temp = open(UNCROPPED_DIR +"/"+filename).read()
          m = hashlib.sha256();
          m.update(temp);
          if (line.split()[-1] == m.hexdigest()):
              altered_image = crop(original_image, line.split()[5].split(','))
              imsave(CROPPED_DIR+ '/'+filename, altered_image)
              print filename
              i += 1
        except:
            pass


def get_train_batch(M, N):
    n = N/6
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train_"+actor_name for actor_name in ALL_ACTORS]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(6):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test_"+actor_name for actor_name in ALL_ACTORS]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_validation(M):
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["validation_"+actor_name for actor_name in ALL_ACTORS]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train_"+actor_name for actor_name in ALL_ACTORS]

    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        


#downloadData(['Fran Drescher', 'Alec Baldwin', 'Bill Hader', 'Steve Carell', 'America Ferrera', 'Kristin Chenoweth'])

dataset = createDataset(ALL_ACTORS)

x = tf.placeholder(tf.float32, [None, 4096])


nhid = 300
W0 = tf.Variable(tf.random_normal([4096, nhid], stddev=0.00001))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.00001))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 0.00005
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(dataset)
validation_x, validation_y = get_validation(dataset)

a1=[]
a2=[]
a3=[]
for i in range(6000):
  #print i  
  batch_xs, batch_ys = get_train_batch(dataset, 60)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 100 == 0:
    print "i=",i
    print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    batch_xs, batch_ys = get_train(dataset)
    print "Validation:", sess.run(accuracy, feed_dict={x: validation_x, y_: validation_y})
    print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print "Penalty:", sess.run(decay_penalty)
    a1.append(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    a2.append(sess.run(accuracy, feed_dict={x: validation_x, y_: validation_y}))
    a3.append(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


  # elif i % 5999 == 0:
  #   snapshot = {}
  #   snapshot["W0"] = sess.run(W0)
  #   snapshot["W1"] = sess.run(W1)
  #   snapshot["b0"] = sess.run(b0)
  #   snapshot["b1"] = sess.run(b1)
  #   X, Y = meshgrid(arange(0, 64), arange(0, 64))
  #   for index in range(6):

  #       max_actor_index = snapshot["W1"].T[index].tolist().index(max(snapshot["W1"].T[index]))
  #       W__ = snapshot["W0"].T[max_actor_index]
  #       W__ = W__.reshape(64, 64)
  #       fig = figure(index)
  #       ax = fig.gca()
  #       heatmap = ax.imshow(W__, cmap = cm.coolwarm)
  #       title(ALL_ACTORS[index])
  #       fig.colorbar(heatmap, shrink = 0.5, aspect = 5)
  #       savefig(ALL_ACTORS[index] + '.png')



