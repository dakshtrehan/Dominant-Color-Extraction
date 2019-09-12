#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import cv2


# In[5]:


im=cv2.imread("elephant.jpg")
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(im)


# In[50]:


original_pixels=im.shape
print(im.shape)


# In[38]:


#Flattening each channel of image
pixel = im.reshape((330*500,3))
print(pixel.shape)


# In[39]:


from sklearn.cluster import KMeans


# In[66]:


dominant_color= 5
km= KMeans(n_clusters= dominant_color)
km.fit(pixel)


# In[67]:


centers = km.cluster_centers_
print(centers)


# In[68]:


centers = np.array(centers, dtype='uint8')


# In[69]:


print(centers)


# In[70]:


#Plotting these colors

i =1
plt.figure(0,figsize=(8,2))


color = []
for x in centers:
    plt.subplot(1,5,i)
    plt.axis("off")
    i+=1
    color.append(x)
    
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = x
    plt.imshow(a)
plt.show()


# In[71]:


#Segmenting the original image
new_pixel= np.zeros(((330*500),3),dtype='uint8')


# In[72]:


color


# In[73]:


km.labels_


# In[74]:


for i in range(new_pixel.shape[0]):
    new_pixel[i]=color[km.labels_[i]]
new_pixel=new_pixel.reshape(original_pixels)


# In[75]:


plt.imshow(im)
plt.title("Before segmentation")
plt.show()


# In[76]:


plt.imshow(new_pixel)
plt.title("After segmentation")
plt.show()


# In[ ]:





# In[ ]:




