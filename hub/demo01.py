import paddlehub as hub
# module = hub.Module(name="openpose_body_estimation", version="1.0.0")
module = hub.Module(name="openpose_body_estimation")

res = module.predict(
    img="./test_image.jpg",
    visualization=True,
    save_path='keypoint_output')
