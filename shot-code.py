import cv2
import numpy as np
import ellipses as el
from matplotlib.patches import Ellipse


##########################################################Clustering by morphology##############################################################

#### reading the image, cropping the region of interest, change it to grayscale, and resizing it
    
img_path = input('image path?')
img = cv2.imread(img_path)
img_draw = img
s = img.shape
img = cv2.cvtColor(img[700:s[2]-400, 1000:s[1]-820], cv2.COLOR_BGR2GRAY)
img_prime = img
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
img_draw = cv2.resize(img_draw[700:s[2]-400, 1000:s[1]-820], (0,0), fx=0.25, fy=0.25)



#### binarizing the image

img_1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img_prime = cv2.threshold(img_prime, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


#### changing the black foreground to white

img_1 = cv2.bitwise_not(img_1)


#### denoising with the openning operation

kernel = np.ones((2,2),np.uint8)
img_2 = cv2.morphologyEx(img_1, cv2.MORPH_OPEN, kernel)


#### dilation repeatedly until all separated white parts of the targets are merged as a whole

kernel = np.ones((3,3),np.uint8)
img_3 = cv2.dilate(img_2,kernel,iterations = 4)


#### Filling small gaps in the image resulted from last step using the closing operation

img_4 = cv2.morphologyEx(img_3, cv2.MORPH_CLOSE, kernel)


#### Extracting the edge of each subregion

img_5 = cv2.Canny(img_4,0,200)


########################################################Annotation##############################################################################
Dic = {}
with open('shot-code-recognition/annotation.txt', 'r') as f:
   A = [line.split() for line in f]
   for j in range(0,len(A)):
       Ant = np.zeros((1,len(A[j])))
       for i in range(0,len(A[j])):
           Ant[0][i] = float(A[j][i])
       Ant = map(tuple,Ant)
       Dic.update({j+1:Ant[0]})
    
##########################################Representation and classification of the coded targets################################################

#### finding the ROI of each shot-code

_, contours, _ = cv2.findContours(img_5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for (counter,cnt) in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(cnt)
    ROI = img_prime[4*y-5:4*(y+h)+5,4*x-5:4*(x+w)+5]
   
#### obtain all the subedges of the ROI, if there is just one subregion, ignore it
 
    EDG = cv2.Canny(ROI,0,255)
    _, subcontours, _ = cv2.findContours(EDG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(subcontours)==1:
       break


#### fitting an ellipse to each subedge and finding the one with minimum fitting error

    I = np.zeros(ROI.shape) 
    E = []
    CENTER = []
    WIDTH = []
    HEIGHT = []
    THETA = [] 

    for subcnt in subcontours:
        xdata = []
        ydata = []
        for i in range(0,len(subcnt)):
  	    xdata.append(subcnt[i][0][1])
	    ydata.append(subcnt[i][0][0])
        data = [np.asarray(xdata), np.asarray(ydata)]
        lsqe = el.LSqEllipse()
        lsqe.fit(data)
        center, width, height, theta, Error = lsqe.parameters()
        E.append(Error/len(subcnt))
        CENTER.append(center)
        WIDTH.append(width)
        HEIGHT.append(height)
        THETA.append(theta)

    m_er = E.index(min(E))

#### constructing the minimum-fitting-error ellipse 

    if WIDTH[m_er]>HEIGHT[m_er]:
       a = int(2.78*WIDTH[m_er])
       b = int(2.78*HEIGHT[m_er])
    else:
       b = int(2.78*WIDTH[m_er])
       a = int(2.78*HEIGHT[m_er])

    ellps = cv2.ellipse(I,(int(CENTER[m_er][1]),int(CENTER[m_er][0])),(a,b),-THETA[m_er]*180/np.pi,0,360,255,1)

#### omitting the central circle from the subedeges
   
    del subcontours[m_er]


#### finding the intersections between the ellipse and all remaining subedges

    p = []
    
    intrsec = np.zeros(ROI.shape)

    for i in range(1,I.shape[0]-1):
        for j in range(1,I.shape[1]-1):

            if (I[i,j]==255 and (EDG[i,j]==255 or 
                EDG[i-1,j]==255 or EDG[i+1,j]==255 or
                EDG[i,j-1]==255 or EDG[i,j+1]==255)):

                intrsec[i,j]=255

    intrsec = np.array(intrsec, dtype=np.uint8)
    _, intrseccon, _ = cv2.findContours(intrsec.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#separating intersection areas

    for k in range(0,len(intrseccon)):
        p.append([intrseccon[k][0][0][1],intrseccon[k][0][0][0]])


#### rotating the intersection points and scaling them by the factor a/b

    im_test = np.zeros(ROI.shape)
    rows,cols = ROI.shape

    for k in range (0,len(p)):
        im_test[p[k][0], p[k][1]]=255

    alpha = -THETA[m_er]*180/(np.pi)

    M = cv2.getRotationMatrix2D((int(CENTER[m_er][1]),int(CENTER[m_er][0])),alpha,1)
    im_test = cv2.warpAffine(im_test,M,(cols,rows))

    for i in range(0, im_test.shape[0]):
        for j in range(0, im_test.shape[1]):
            if im_test[i,j]!=0:
               im_test[i,j]=255

    im_res = cv2.resize(im_test, (0,0), fx=1, fy=float(a)/float(b))

    im_res = np.array(im_res, dtype=np.uint8)
    
    p_prime = np.zeros((len(p),2))
    _, testcon, _ = cv2.findContours(im_res.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  
    for k in range(0,len(testcon)):
        p_prime[k] = [testcon[k][0][0][1], testcon[k][0][0][0]] 
 

#### converting to polar coordinate

    gama = np.zeros((1,p_prime.shape[0]))
    o = np.array((int(CENTER[m_er][0]*float(a)/float(b)),int(CENTER[m_er][1])))
 
    for k in range(0, p_prime.shape[0]):            
        v0 = np.array((o[0],ROI.shape[1]))-np.array(o)
        v1 = np.array(p_prime[k])- np.array(o)
        cosine_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        if (p_prime[k][0]<= o[0]):        
           gama[0][k] = np.arccos(cosine_angle)
        else:
           gama[0][k] = 2*np.pi-np.arccos(cosine_angle)

    srt = np.argsort(gama)
    gama = np.sort(gama)
  
    angle = np.zeros((1, p_prime.shape[0]))

    for k in range(0, p_prime.shape[0]-1):
        angle[0][k] = gama[0][k+1]-gama[0][k]

    angle[0][-1] = gama[0][0]+2*np.pi-gama[0][-1]
    
    n = np.floor(14*angle/(2*np.pi)+0.5)
    n_prime = 14*angle/(2*np.pi)+0.5
    
    diff = n_prime - n
    argdiff = np.argsort(diff)
    
    i = 0
    print(14*angle/(2*np.pi)+0.5)
    while (np.sum(n)>14):
        n[0][argdiff[0][i]] = n[0][argdiff[0][i]] - 1
        i = i+1

    j = n.shape[1] -1
    while (np.sum(n)<14):
        n[0][argdiff[0][j]] = n[0][argdiff[0][j]] + 1
        j = j-1

    n = np.asarray([n[np.nonzero(n)]])

    n_t = map(tuple,n)
    N = n_t[0]
 
    count = 0
    EDG_n = cv2.bitwise_not(EDG)
    
    EDG_rot = cv2.warpAffine(EDG_n,M,(cols,rows))
    EDG_rot = EDG_rot[3:EDG_rot.shape[0]-3, 3:EDG_rot.shape[1]-3] 
    o[0] = o[0]-3
    o[1] = o[1]-3 
    D = WIDTH[m_er] + o[1] + 7

    print(gama[0]) 
    o = np.array((int(CENTER[m_er][0]),int(CENTER[m_er][1])))
    if gama[0][0] <= 0.09 or gama[0][-1] >= 6.2:
       MM =  cv2.getRotationMatrix2D((int(CENTER[m_er][1]),int(CENTER[m_er][0])), 7,1)
       EDG_rot = cv2.warpAffine(EDG_rot,MM,(cols - 6,rows - 6))
       EDG_rot = EDG_rot[3:EDG_rot.shape[0]-3, 3:EDG_rot.shape[1]-3]
       o[0] = o[0]-3
       o[1] = o[1]-3 

    d = int(D)
    while d < EDG_rot.shape[1]:
          if EDG_rot[o[0]][d]!=255:
             count = count + 1 
             d = d + 2
          else:
             d = d + 1
	
    if gama[0][-1] >= 6.2 and count <= 1:
       N = N[1:len(N)] + (N[0],)
    if count<8 and count>1 and gama[0][-1]< 6.2: 
       N = N[1:len(N)] + (N[0],)

    print(count)
    print 'sequence:', N    
    print(counter)

    flag = 0
    L = 0
    while(flag==0 and L<len(N)/2):

        for k in range(1,len(Dic)+1):
           if Dic[k]==N:
              print 'code:', k
              font = cv2.FONT_HERSHEY_SIMPLEX
              cv2.putText(img_draw, str(k), (x,y), font, 0.5, 255,2,cv2.LINE_AA)
              flag = 1
        N = N[2:len(N)] + (N[0],) + (N[1],)
        L = L+1

    if flag ==0:
       print('could not find the code')
   
    EDG_rot = cv2.resize(EDG_rot, (0,0), fx=1, fy=float(a)/float(b))

cv2.imwrite('6659.jpg',img_draw)
cv2.imshow('result', img_draw)
cv2.waitKey(0)
cv2.destroyAllWnidows()




 


