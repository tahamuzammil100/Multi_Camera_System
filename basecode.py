import cv2
import numpy as np
import cv2
import numpy as np
from PIL import Image

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    return points


def load_yolo():
    net = cv2.dnn.readNet("Project/darknet/backup/yolov3.weights", "Project/darknet/cfg/yolov3.cfg")
    classes = []
    with open("Project/darknet/data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers
def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels
def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            if class_id==0:
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height) 
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
    return boxes, confs, class_ids
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    points=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])+" "+str(confs)[1:5]
            # if i>2:
            #     i=1
            color = colors[1]
            # cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
            centerx = (x+x+w)/2
            centery = (y+h) 
            points.append([int(centerx), int(centery)])
            # cv2.circle(img, (int(centerx), int(centery)), 3, (255,0,0), 3)
            # cv2.putText(img, label, (x, y - 5), font, 2, (0,0,255), 2)
    return img, points
def detection(frame, model, output_layers, colors, classes):
    height, width, channels = frame.shape
    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img, points = draw_labels(boxes, confs, colors, class_ids, classes, frame)
    return img, points, boxes, confs, class_ids 

def no_detection(frame, colors, classes, boxes, confs, class_ids):
    img, points = draw_labels(boxes, confs, colors, class_ids, classes, frame)
    return img, points
    
def points_homography(points1, h1, top_view):
    homography_points = []
    for p in points1: # doing homogrphy of person detection points
        product = np.dot(h1, [p[0],p[1],1])
        k, l, m = (product / product[2] + 0.5).astype(int)
        if k >= 0 and k < top_view.shape[0] and l >= 0 and l < top_view.shape[1]:
            homography_points.append([k, l, 9999999]) 
    return homography_points

def main():

    print("Press 1 for pre-recorded videos, 2 for live stream, and 0 Top-View: ")
    option = int(input())
    if option==0:
        print("**********   TOP VIEW   **********")
        print("Press 1 for pre-recorded videos, 2 for live stream")
        suboption = int(input())
        if suboption == 1:
            dot_color = (255, 0, 0)
            windowName0 = "Top View of all Camera's"
            cv2.namedWindow(windowName0)
            windowName1 = "Top View of Camera 1"
            cv2.namedWindow(windowName1)
            windowName2 = "Top View of Camera 2"
            cv2.namedWindow(windowName2)
            windowName3 = "Top View of Camera 3"
            cv2.namedWindow(windowName3)
            windowName4 = "Static Heat Map Video"
            cv2.namedWindow(windowName4)
            windowName5 = "Animated Heat Map Video"
            cv2.namedWindow(windowName5)
            
            top_view = cv2.imread("topview1.jpg")
            output_size = (top_view.shape[1], top_view.shape[0]) 
            print("top view shape", top_view.shape)
            print("output_size", output_size)
            optputFile00 = cv2.VideoWriter(
                'homography/vid0.avi', cv2.VideoWriter_fourcc(*'XVID'), 4, (top_view.shape[1], top_view.shape[0]))
            optputFile1 = cv2.VideoWriter(
                'homography/vid1.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile2 = cv2.VideoWriter(
                'homography/vid2.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile3 = cv2.VideoWriter(
                'homography/vid3.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile4 = cv2.VideoWriter(
                'homography/heatmapsvid.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile5 = cv2.VideoWriter(
                'homography/heatmapavid.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)

            capture1 = cv2.VideoCapture("Dataset/one.mp4")
            capture2 = cv2.VideoCapture("Dataset/two.mp4")
            capture3 = cv2.VideoCapture("Dataset/three.mp4")

            if capture1.isOpened():
                ret1, frame1 = capture1.read()
                # pts_src1 = get_four_points(frame1)
                # pts_dst1 = get_four_points(top_view)
                pts_src1 = np.float32([[255, 265], [1070, 206], [1200, 540], [340, 565]])
                pts_dst1 = np.float32([[470, 120], [780, 430], [350, 470], [270, 310]])
                h1, status = cv2.findHomography(pts_src1, pts_dst1)
                im1 = cv2.warpPerspective(frame1, h1, output_size)
                cv2.imwrite("homography/pic1.jpg", im1)
            else:
                ret1 = False
    #   *************************************************************************************************
            if capture2.isOpened():
                ret2, frame2 = capture2.read()
                # pts_src2 = get_four_points(frame2)
                # pts_dst2 = get_four_points(top_view)
                pts_src2 = np.float32([[30, 440], [1425, 406], [1500, 900], [57, 910]])
                pts_dst2 = np.float32([[152, 427], [470, 112], [472, 588], [350, 628]])
                h2, status = cv2.findHomography(pts_src2, pts_dst2)
                im2 = cv2.warpPerspective(frame2, h2, output_size)
                cv2.imwrite("homography/pic2.jpg", im2)
            else:
                ret2 = False
    #   *************************************************************************************************
            if capture3.isOpened():
                ret3, frame3 = capture3.read()
                # pts_src3 = get_four_points(frame3)
                # pts_dst3 = get_four_points(top_view)
                pts_src3 = np.float32([[440, 147], [1830, 300], [1585, 820], [125, 670]])
                pts_dst3 = np.float32([[470, 750], [150, 430], [355, 230], [510, 310]])
                h3, status = cv2.findHomography(pts_src3, pts_dst3)
                im3 = cv2.warpPerspective(frame3, h3, output_size)
                cv2.imwrite("homography/pic3.jpg", im3)  

                a = im1 / 255
                b = im2 / 255
                c = im3 / 255
                final = np.zeros((a.shape))
                final = np.where(np.logical_and(a != 0, b != 0, c != 0), (a + b + c) / 3, final)
                final = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final)
                final = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final)

                flag, encoded_image = cv2.imencode(".jpg", (final * 255).astype(np.float32))     
                    
                cv2.imwrite("homography/stiched.jpg", final*255 )  
            else:
                ret3 = False

            frame_no=0
            model, classes, colors, output_layers = load_yolo()
            
            person_points = []
            person_points2 = []
            k = 21
            gauss = cv2.getGaussianKernel(k, np.sqrt(64))
            gauss = gauss * gauss.T
            gauss = (gauss / gauss[int(k/2), int(k/2)])

            heatmap_image = np.zeros((top_view.shape[0], top_view.shape[1], 3)).astype(np.float32)
            heatmap_image2 = np.zeros((top_view.shape[0], top_view.shape[1], 3)).astype(np.float32)
            jj = cv2.cvtColor(cv2.applyColorMap(((gauss)*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            flagg = 0
            time_steps = 10

            while ret1 and ret2 and ret3:
                skip = 10 # Skip frames

                ret1, frame1 = capture1.read()
                img1, points1, boxes1, confs1, class_ids1 = detection(frame1, model, output_layers, colors, classes) 
                homography_points = points_homography(points1, h1, top_view)
                size = len(homography_points)
                
                lst=[]
                if size!=1:
                    for i in range(0, size):
                        lst.append(i)
                        for j in range(0, size):
                            if j not in lst:
                                point1 = homography_points[i]
                                point2 = homography_points[j]        
                                dist = np.sqrt(np.square(np.subtract(point1[0], point2[0])) + np.square(np.subtract(point1[1], point2[1])))
                                if dist < homography_points[i][2]:
                                    homography_points[i][2] = dist
                                if dist < homography_points[j][2]:
                                    homography_points[j][2] = dist
                    

                    
                im_temp1 = cv2.warpPerspective(img1, h1, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    if int(p[2])>100:
                        cv2.circle(im_temp1, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    else:
                        cv2.circle(im_temp1, (int(p[0]), int(p[1])), 3, (0,0, 255), 3)

                optputFile1.write(im_temp1)
                person_points = person_points + homography_points
                person_points2 = person_points2 + homography_points
                cv2.imshow(windowName1, im_temp1)
                for i in range(skip):
                    ret1, frame1 = capture1.read()
                    # img1, points1 = no_detection(frame1, colors, classes, boxes1, confs1, class_ids1)
                    # homography_points = points_homography(points1, h1, top_view)
                    # im_temp1 = cv2.warpPerspective(img1, h1, (top_view.shape[1],top_view.shape[0]))
                    # for p in homography_points:
                    #     cv2.circle(im_temp1, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    # optputFile1.write(im_temp1)

                ret2, frame2 = capture2.read()
                img2, points2, boxes2, confs2, class_ids2 = detection(frame2, model, output_layers, colors, classes)
                homography_points = points_homography(points2, h2, top_view)
                size = len(homography_points)
                
                lst=[]
                if size!=1:
                    for i in range(0, size):
                        lst.append(i)
                        for j in range(0, size):
                            if j not in lst:
                                point1 = homography_points[i]
                                point2 = homography_points[j]        
                                dist = np.sqrt(np.square(np.subtract(point1[0], point2[0])) + np.square(np.subtract(point1[1], point2[1])))
                                if dist < homography_points[i][2]:
                                    homography_points[i][2] = dist
                                if dist < homography_points[j][2]:
                                    homography_points[j][2] = dist


                im_temp2 = cv2.warpPerspective(img2, h2, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    if int(p[2])>100:
                        cv2.circle(im_temp2, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    else:
                        cv2.circle(im_temp2, (int(p[0]), int(p[1])), 3, (0,0, 255), 3)

                optputFile2.write(im_temp2)
                person_points = person_points + homography_points
                person_points2 = person_points2 + homography_points
                cv2.imshow(windowName2, im_temp2)
                for i in range(skip):
                    ret2, frame2 = capture2.read()
                    # img2, points2 = no_detection(frame2, colors, classes, boxes2, confs2, class_ids2)
                    # homography_points = points_homography(points2, h2, top_view)
                    # im_temp2 = cv2.warpPerspective(img2, h2, (top_view.shape[1],top_view.shape[0]))
                    # for p in homography_points:
                    #     cv2.circle(im_temp2, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    # optputFile2.write(im_temp2)

                ret3, frame3 = capture3.read()
                img3, points3, boxes3, confs3, class_ids3 = detection(frame3, model, output_layers, colors, classes)
                homography_points = points_homography(points3, h3, top_view)
                size = len(homography_points)
                lst=[]
                if size!=1:
                    for i in range(0, size):
                        lst.append(i)
                        for j in range(0, size):
                            if j not in lst:
                                point1 = homography_points[i]
                                point2 = homography_points[j]        
                                dist = np.sqrt(np.square(np.subtract(point1[0], point2[0])) + np.square(np.subtract(point1[1], point2[1])))
                                if dist < homography_points[i][2]:
                                    homography_points[i][2] = dist
                                if dist < homography_points[j][2]:
                                    homography_points[j][2] = dist
                im_temp3 = cv2.warpPerspective(img3, h3, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    if int(p[2])>100:
                        cv2.circle(im_temp3, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    else:
                        cv2.circle(im_temp3, (int(p[0]), int(p[1])), 3, (0,0, 255), 3)

                optputFile3.write(im_temp3)
                person_points = person_points + homography_points
                person_points2 = person_points2 + homography_points
                cv2.imshow(windowName3, im_temp3)
                for i in range(skip):
                    ret3, frame3 = capture3.read()
                    # img3, points3 = no_detection(frame3, colors, classes, boxes3, confs3, class_ids3)
                    # homography_points = points_homography(points3, h3, top_view)
                    # im_temp3 = cv2.warpPerspective(img3, h3, (top_view.shape[1],top_view.shape[0]))
                    # for p in homography_points:
                    #     cv2.circle(im_temp3, (int(p[0]), int(p[1])), 3, dot_color, 3)
                #     optputFile3.write(im_temp3)


                a = im_temp1 / 255
                b = im_temp2 / 255
                c = im_temp3 / 255
                final = np.zeros((top_view.shape))
                final1 = np.where(np.logical_and(a != 0, b != 0, c != 0), a, final)
                final2 = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final1)
                final3 = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final2)


                # # print(np.max(im_temp1))
                # # print(np.max(final))
                cv2.imshow(windowName0, final3)
                if flagg == 0:
                    stitched_image = final3
                flagg = 1
                # flag, encoded_image = cv2.imencode(".jpg", (final * 255).astype(np.float32))
                # cv2.imwrite("homography/frames/academics_block/"+str(frame_no)+".jpg", final*255)
                # frame_no+=1
                frame_no+=1
                
                # save_vid = ((final2*255).astype(int)).copy()
                
                cv2.imwrite("homography/frames/topview_vid/"+str(frame_no)+".jpg", final3*255)
                optputFile00.write(cv2.imread("homography/frames/topview_vid/"+str(frame_no)+".jpg"))


                heatmap_image = np.zeros((top_view.shape[0], top_view.shape[1], 3)).astype(np.float32)
                
                for p in person_points:
                    bb = heatmap_image[p[1] - int(k/2):p[1]+int(k/2)+1, p[0]-int(k/2):p[0]+int(k/2)+1, :]
                    cc = jj + bb
                    heatmap_image[p[1] - int(k/2):p[1]+int(k/2)+1, p[0]-int(k/2):p[0]+int(k/2)+1, :] = cc



                m = np.max(heatmap_image, axis = (0,1)) + 0.0001
                heatmap_image = heatmap_image / m
                heatmap_image = np.where(heatmap_image == 0.0, stitched_image, heatmap_image)
                cv2.imshow(windowName4, heatmap_image)

                # cv2.imwrite("homography/frames/heatmap_vid/"+str(frame_no)+".jpg", heatmap_image)
                # cv2.imread("homography/frames/heatmap_vid/"+str(frame_no)+".jpg")

                optputFile4.write(heatmap_image)


                # heatmap_image2 = np.zeros((top_view.shape[0], top_view.shape[1], 3)).astype(np.float32)
                # if len(person_points2) > time_steps:
                #     person_points2 = person_points2.pop()
                # for p in person_points2:
                #     bb2 = heatmap_image2[p[1] - int(k/2):p[1]+int(k/2)+1, p[0]-int(k/2):p[0]+int(k/2)+1, :]
                #     cc2 = jj + bb2
                #     heatmap_image2[p[1] - int(k/2):p[1]+int(k/2)+1, p[0]-int(k/2):p[0]+int(k/2)+1, :] = cc2

                # m = np.max(heatmap_image2, axis = (0,1)) + 0.0001
                # heatmap_image2 = heatmap_image2 / m
                # heatmap_image2 = np.where(heatmap_image2 == 0.0, stitched_image, heatmap_image2)
                # cv2.imshow(windowName5, heatmap_image2)
                # optputFile5.write(heatmap_image2)



                if cv2.waitKey(1) == 27:
                    break

        

            # capture1.release()
            # optputFile1.release()
            # cv2.destroyAllWindows()


        elif suboption == 12:
            print("Live Video")
            dot_color = (0, 0, 255)
            windowName0 = "Sample Feed from Camera 0"
            cv2.namedWindow(windowName0)
            # windowName1 = "Sample Feed from Camera 1"
            # cv2.namedWindow(windowName1)
            # windowName2 = "Sample Feed from Camera 2"
            # cv2.namedWindow(windowName2)
            # windowName3 = "Sample Feed from Camera 3"
            # cv2.namedWindow(windowName3)
            
            top_view = cv2.imread("topview1.jpg")
            output_size = (top_view.shape[1], top_view.shape[0]) 
            
            optputFile0 = cv2.VideoWriter(
                'homography/vid00.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile1 = cv2.VideoWriter(
                'homography/vid11.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile2 = cv2.VideoWriter(
                'homography/vid22.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)
            optputFile3 = cv2.VideoWriter(
                'homography/vid33.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, output_size)

            # capture1 = cv2.VideoCapture("Dataset/one.mp4")
            # capture2 = cv2.VideoCapture("Dataset/two.mp4")
            # capture3 = cv2.VideoCapture("Dataset/three.mp4")
            capture2 = cv2.VideoCapture("http://10.104.6.198:8080/video")  # laptop's camera alishba
            capture3 = cv2.VideoCapture("http://10.104.6.247:8080/video")   # sample code for mobile camera video capture using IP camera taha
            capture1 = cv2.VideoCapture("http://10.104.2.33:8080/video")    # sample code for mobile camera video  ujala

            if capture1.isOpened():
                ret1, frame1 = capture1.read()
                # pts_src1 = get_four_points(frame1)
                # pts_dst1 = get_four_points(top_view)
                pts_src1 = np.float32([[255, 265], [1070, 206], [1200, 540], [340, 565]])
                pts_dst1 = np.float32([[470, 120], [780, 430], [350, 470], [270, 310]])
                h1, status = cv2.findHomography(pts_src1, pts_dst1)
                im1 = cv2.warpPerspective(frame1, h1, output_size)
                cv2.imwrite("homography/pic1.jpg", im1)
            else:
                ret1 = False
    #   *************************************************************************************************
            if capture2.isOpened():
                ret2, frame2 = capture2.read()
                # pts_src2 = get_four_points(frame2)
                # pts_dst2 = get_four_points(top_view)
                pts_src2 = np.float32([[30, 440], [1425, 406], [1500, 900], [57, 910]])
                pts_dst2 = np.float32([[152, 427], [470, 112], [472, 588], [350, 628]])
                h2, status = cv2.findHomography(pts_src2, pts_dst2)
                im2 = cv2.warpPerspective(frame2, h2, output_size)
                cv2.imwrite("homography/pic2.jpg", im2)
            else:
                ret2 = False
    #   *************************************************************************************************
            if capture3.isOpened():
                ret3, frame3 = capture3.read()
                # pts_src3 = get_four_points(frame3)
                # pts_dst3 = get_four_points(top_view)
                pts_src3 = np.float32([[440, 147], [1830, 300], [1585, 820], [125, 670]])
                pts_dst3 = np.float32([[470, 750], [150, 430], [355, 230], [510, 310]])
                h3, status = cv2.findHomography(pts_src3, pts_dst3)
                im3 = cv2.warpPerspective(frame3, h3, output_size)
                cv2.imwrite("homography/pic3.jpg", im3)  

                a = im1 / 255
                b = im2 / 255
                c = im3 / 255
                final = np.zeros((a.shape))
                final = np.where(np.logical_and(a != 0, b != 0, c != 0), (a + b + c) / 3, final)
                final = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final)
                final = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final)

                flag, encoded_image = cv2.imencode(".jpg", (final * 255).astype(np.float32))     
                    
                cv2.imwrite("homography/stiched.jpg", final*255 )  
            else:
                ret3 = False

            frame_no=0
            model, classes, colors, output_layers = load_yolo()
            while ret1 and ret2 and ret3:
                skip = 10 # Skip frames

                ret1, frame1 = capture1.read()
                img1, points1, boxes1, confs1, class_ids1 = detection(frame1, model, output_layers, colors, classes) 
                homography_points = points_homography(points1, h1, top_view)
                im_temp1 = cv2.warpPerspective(img1, h1, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    cv2.circle(im_temp1, (int(p[0]), int(p[1])), 3, dot_color, 3)
                optputFile1.write(im_temp1)
                for i in range(skip):
                    ret1, frame1 = capture1.read()
                    img1, points1 = no_detection(frame1, colors, classes, boxes1, confs1, class_ids1)
                    homography_points = points_homography(points1, h1, top_view)
                    im_temp1 = cv2.warpPerspective(img1, h1, (top_view.shape[1],top_view.shape[0]))
                    for p in homography_points:
                        cv2.circle(im_temp1, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    optputFile1.write(im_temp1)

                ret2, frame2 = capture2.read()
                img2, points2, boxes2, confs2, class_ids2 = detection(frame2, model, output_layers, colors, classes)
                homography_points = points_homography(points2, h2, top_view)
                im_temp2 = cv2.warpPerspective(img2, h2, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    cv2.circle(im_temp2, (int(p[0]), int(p[1])), 3, dot_color, 3)
                optputFile2.write(im_temp2)
                for i in range(skip):
                    ret2, frame2 = capture2.read()
                    img2, points2 = no_detection(frame2, colors, classes, boxes2, confs2, class_ids2)
                    homography_points = points_homography(points2, h2, top_view)
                    im_temp2 = cv2.warpPerspective(img2, h2, (top_view.shape[1],top_view.shape[0]))
                    for p in homography_points:
                        cv2.circle(im_temp2, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    optputFile2.write(im_temp2)

                ret3, frame3 = capture3.read()
                img3, points3, boxes3, confs3, class_ids3 = detection(frame3, model, output_layers, colors, classes)
                homography_points = points_homography(points3, h3, top_view)
                im_temp3 = cv2.warpPerspective(img3, h3, (top_view.shape[1],top_view.shape[0]))
                for p in homography_points:
                    cv2.circle(im_temp3, (int(p[0]), int(p[1])), 3, dot_color, 3)
                optputFile3.write(im_temp3)
                for i in range(skip):
                    ret3, frame3 = capture3.read()
                    img3, points3 = no_detection(frame3, colors, classes, boxes3, confs3, class_ids3)
                    homography_points = points_homography(points3, h3, top_view)
                    im_temp3 = cv2.warpPerspective(img3, h3, (top_view.shape[1],top_view.shape[0]))
                    for p in homography_points:
                        cv2.circle(im_temp3, (int(p[0]), int(p[1])), 3, dot_color, 3)
                    optputFile3.write(im_temp3)


                a = im_temp1 / 255
                b = im_temp2 / 255
                c = im_temp3 / 255
                final = np.zeros((a.shape))
                final = np.where(np.logical_and(a != 0, b != 0, c != 0), (a + b + c) / 3, final)
                final = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final)
                final = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                            np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                            np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final)

                cv2.imshow(windowName0, final)
                flag, encoded_image = cv2.imencode(".jpg", (final * 255).astype(np.float32))
                # cv2.imwrite("homography/frames/academics_block/"+str(frame_no)+".jpg", final*255)
                frame_no+=1
                optputFile0.write(np.array(final*255))




                if cv2.waitKey(1) == 27:
                    break

        

            # capture1.release()
            # optputFile1.release()
            # cv2.destroyAllWindows()


    if option == 1:
        # Record video
        windowName1 = "Sample Feed from Camera 1"
        windowName2 = "Sample Feed from Camera 2"
        windowName3 = "Sample Feed from Camera 3"
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)

        capture1 = cv2.VideoCapture("Dataset/one.mp4")  # laptop's camera
        capture2 = cv2.VideoCapture("Dataset/two.mp4")   # sample code for mobile camera video capture using IP camera
        capture3 = cv2.VideoCapture("Dataset/three.mp4")    # sample code for mobile camera video 

        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)

        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            'detections/result11.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, size1)

        optputFile2 = cv2.VideoWriter(
            'detections/result22.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, size2)

        optputFile3 = cv2.VideoWriter(
            'detections/result33.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, size3)
 

        # check if feed exists or not for camera 1
        if capture1.isOpened():
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
         
        if capture2.isOpened():
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        model, classes, colors, output_layers = load_yolo()
        while ret1 and ret2 and ret3:
            skip = 10
            
            # for i in range(0,10):
            ret1, frame1 = capture1.read()
            img1, boxes1, confs1, class_ids1 = detection(frame1, model, output_layers, colors, classes)
            cv2.imshow(windowName1, img1)
            optputFile1.write(img1)
            for i in range(skip):
                ret1, frame1 = capture1.read()
                img1 = no_detection(frame1, colors, classes, boxes1, confs1, class_ids1)
                cv2.imshow(windowName1, img1)
                optputFile1.write(img1)
            
            ret2, frame2 = capture2.read()
            img2, boxes2, confs2, class_ids2 = detection(frame2, model, output_layers, colors, classes)
            cv2.imshow(windowName2, img2)
            optputFile2.write(img2)
            for i in range(skip):
                ret2, frame2 = capture2.read()
                img2 = no_detection(frame2, colors, classes, boxes2, confs2, class_ids2)
                cv2.imshow(windowName2, img2)
                optputFile2.write(img2)

            ret3, frame3 = capture3.read()
            img3, boxes3, confs3, class_ids3 = detection(frame3, model, output_layers, colors, classes)
            cv2.imshow(windowName3, img3)
            optputFile3.write(img3)
            for i in range(skip):
                ret3, frame3 = capture3.read()
                img3 = no_detection(frame3, colors, classes, boxes3, confs3, class_ids3)
                cv2.imshow(windowName3, img3)
                optputFile3.write(img3)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    elif option == 2:
        # live stream
        windowName1 = "Live Stream Camera 1"
        windowName2 = "Live Stream Camera 2"
        windowName3 = "Live Stream Camera 3"
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)

        capture1 = cv2.VideoCapture(0)  # laptop's camera
        capture2 = cv2.VideoCapture("http://10.104.6.198:8080/video")   # sample code for mobile camera video capture using IP camera
        capture3 = cv2.VideoCapture("http://10.104.6.247:8080/video")    # sample code for mobile camera video 


        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        if capture2.isOpened():  # check if feed exists or not for camera 2
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        if capture3.isOpened():  # check if feed exists or not for camera 3
            ret3, frame3 = capture3.read()
        else:
            ret3 = False

        model, classes, colors, output_layers = load_yolo()
        while ret1 and ret2:
            ret1, frame1 = capture1.read()
            height, width, channels = frame1.shape
            blob, outputs = detect_objects(frame1, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            img1 = draw_labels(boxes, confs, colors, class_ids, classes, frame1)
            cv2.imshow(windowName1, img1)

            ret2, frame2 = capture2.read()
            height, width, channels = frame2.shape
            blob, outputs = detect_objects(frame2, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            img2 = draw_labels(boxes, confs, colors, class_ids, classes, frame2)
            cv2.imshow(windowName2, img2)

            ret3, frame3 = capture3.read()
            height, width, channels = frame3.shape
            blob, outputs = detect_objects(frame3, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            img3 = draw_labels(boxes, confs, colors, class_ids, classes, frame3)
            cv2.imshow(windowName3, img3) 

            if cv2.waitKey(1) == 27:
                break
        capture1.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid option entered. Exiting...")

    
main()
