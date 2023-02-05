import cv2

name = "myfire"
vidcap = cv2.VideoCapture('tests/'+name+'.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite('tests/'+name+"/"+name+"_%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1