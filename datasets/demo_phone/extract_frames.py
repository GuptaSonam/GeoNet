import cv2
import os

# Parameters
# TODO: use inline arguments
# TODO: add arguments for resize
videos_dir = './videos'
videos = os.listdir(videos_dir)

for video in videos:
    # Open video
    cap = cv2.VideoCapture(os.path.join(videos_dir, video))
    frame_id = 0

    if not cap.isOpened():
        print('Could not open video file %s.' % video)

    # Use video name as folder name
    sequence_name = video.split('.')[0]
    sequence_dir = os.path.join('.', 'sequences', sequence_name)
    print('Extracting frames from %s' % sequence_dir)

    if os.path.isdir(sequence_dir):
        # Assume frames are already extracted for this video
        print('Directory %s already exists, skipping frame extraction.' % sequence_dir)
    else:
        # Extract frames for this video
        os.makedirs(sequence_dir)

        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()

            if ret:
                # TODO: resize frame to target size
                # Store image
                frame_path = os.path.join(sequence_dir, ('%.3d.jpg') % frame_id)
                cv2.imwrite(frame_path, frame)
                frame_id = frame_id + 1
            else:
                break
        print('Done. Extracted %d frames.' % frame_id)
