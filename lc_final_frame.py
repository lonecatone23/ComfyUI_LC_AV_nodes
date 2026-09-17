"""
LC Final Frame
--------------
Pick one frame from an IMAGE batch (video frames).
"""


class LCFinalFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frame batch. Last frame is returned."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    CATEGORY = "LC AV/video"
    DESCRIPTION = "Last frame of an IMAGE batch. Same contract as FinalFrameSelector."

    def pick(self, images):
        n = int(images.shape[0]) if hasattr(images, "shape") else len(images)
        if n <= 0:
            raise ValueError("LC Final Frame: empty batch.")
        return (images[n - 1:n],)


NODE_CLASS_MAPPINGS = {
    "LCFinalFrame": LCFinalFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCFinalFrame": "LC Final Frame 🖼️",
}
