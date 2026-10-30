from io import BytesIO

from PIL import Image

from backend.app.preprocessing.quality import decode_image


def test_decode_image_applies_exif_orientation_before_reporting_dimensions():
    image = Image.new("RGB", (240, 160), "green")
    exif = image.getexif()
    exif[274] = 6
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)

    decoded, quality = decode_image(output.getvalue())

    assert decoded.size == (160, 240)
    assert (quality.width, quality.height) == (160, 240)
