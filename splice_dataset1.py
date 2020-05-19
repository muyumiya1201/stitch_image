##############################從右邊拼照片集##############################
import cv2
import numpy as np

frame_num = 0
total_frame = 81

goal = 'splice_dataset1.mp4'
width = (1920)//4*2
height = (1080)//4
fps = 30
videoWriter = cv2.VideoWriter(goal, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width, height))


def set_frame_number(x):
    global frame_num
    frame_num = x
    return

result = np.zeros((int(1080)//4,int(1920)//4*2,3))
count  = np.zeros((int(1080)//4,int(1920)//4*2))
ones = np.ones(((int(1080)//4,int(1920)//4)))

cv2.namedWindow('matching')
cv2.createTrackbar('frame no.','matching',0,total_frame-1,set_frame_number)

kpdetector = cv2.xfeatures2d.SIFT_create() 
#kpdetector = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
i = 0
while i in range(81):
    frame2=cv2.imread(".\\dataset1\\dataset_1_%d.jpg"%(i))
    print(".\\dataset1\\dataset_1_%d.jpg"%(i))
    frame2 = cv2.resize(frame2,(frame2.shape[1]//4,frame2.shape[0]//4))
    
#kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]
    if frame_num == 0:
        T      = np.eye(3)
        T[0,2] = result.shape[1]-frame2.shape[1]
        T[1,2] = 0
        result = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        count = t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
    
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
    else:
        # Match descriptors.
        matches = bf.match(dt2,dt1)
        print('{}, # of matches:{}'.format(frame_num,len(matches)))

        # Sort in ascending order of distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        src = []
        dst = []
        for m in matches:
            src.append(kp2[m.queryIdx].pt + (1,))
            dst.append(kp1[m.trainIdx].pt + (1,))
            
        src = np.array(src,dtype=np.float)
        dst = np.array(dst,dtype=np.float)
    
        # find a homography to map src to dst
        A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
        
        # map to the first frame
        T = T.dot(A)
        warp_img = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        result+= warp_img
        count += t_count.astype(np.float)

        t_count= count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()
        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count

        cv2.imshow('stitched image',disp.astype(np.uint8))
   
        cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1,kp1,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        
        
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
    
    videoWriter.write(disp.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    frame_num += 1
    i+=1
cv2.waitKey()    
videoWriter.release()
cv2.destroyAllWindows()