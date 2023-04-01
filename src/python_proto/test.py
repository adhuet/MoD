import cv2 as cv
import numpy as np

# See https://www.freedomvc.com/index.php/2021/06/26/contours-and-bounding-boxes/

# This function allows us to create a descending sorted list of contour areas.


def contour_area(contours):

    # create an empty list
    cnt_area = []

    # loop through all the contours
    for i in range(0, len(contours), 1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv.contourArea(contours[i]))

    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area


def draw_bounding_box(contours, image, number_of_boxes=1):
    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)

    # Loop through each contour of our image
    for i in range(0, len(contours), 1):
        cnt = contours[i]

        # Only draw the the largest number of boxes
        if (cv.contourArea(cnt) > cnt_area[number_of_boxes]):

            # Use OpenCV boundingRect function to get the details of the contour
            x, y, w, h = cv.boundingRect(cnt)

            # Draw the bounding box
            image = cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return image


if __name__ == "__main__":
    bgd = cv.imread("background.png")
    cv.imshow("Background", bgd)
    cv.imwrite("test_Background.png", bgd)

    frame = cv.imread("source_example.png")

    cv.imshow("Frame", frame)
    cv.imwrite("test_Frame.png", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bgd = cv.cvtColor(bgd, cv.COLOR_BGR2GRAY)

    cv.imshow("Grayscale", gray)
    cv.imwrite("test_Grayscale.png", gray)

    blur = cv.GaussianBlur(gray, (5, 5), 0.2)
    bgd = cv.GaussianBlur(bgd, (5, 5), 0)

    cv.imshow("Blurred", blur)
    cv.imwrite("test_Blurred.png", blur)
    cv.imwrite("test_bgd_Blurred.png", bgd)

    # diff = cv.subtract(bgd, frame)
    # Conv_hsv_Gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    # ret, mask = cv.threshold(Conv_hsv_Gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # diff[mask != 255] = [0, 0, 255]

    diff = cv.absdiff(bgd, blur)
    cv.imshow("Diff", diff)
    cv.imwrite("test_Diff.png", diff)

    opening = cv.morphologyEx(diff, cv.MORPH_OPEN, (15, 15))
    cv.imshow("Opening", opening)
    cv.imwrite("test_Opening.png", opening)

    ret, thresh = cv.threshold(opening, 20, 255, cv.THRESH_BINARY)
    cv.imshow("Thresholded", thresh)
    cv.imwrite("test_Thresholded.png", thresh)

    contours, hierarchy = cv.findContours(thresh, 1, 2)
    output = draw_bounding_box(contours, frame)
    cv.imshow("Boxed", output)
    cv.imwrite("test_Boxed.png", output)
    cv.waitKey(0)
