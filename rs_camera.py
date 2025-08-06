import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        depth_value = depth_frame.get_distance(cx, cy)
        depth_mm = int(depth_value * 1000)
        cv2.circle(color_image, (cx, cy), 3, (255, 0, 0), 3, 1, )
        # cv2.imshow("Depth Image", depth_color)
        # cv2.imshow("Color Image", color_image)
        print(f"Depth value: {depth_mm}")
        cv2.putText(color_image, f"z: {round(depth_mm, 2)} mm", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        combined_image = np.hstack((color_image, depth_color))
        cv2.imshow('RGB + Depth', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()